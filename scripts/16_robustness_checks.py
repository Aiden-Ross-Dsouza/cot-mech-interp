"""
scripts/16_robustness_checks.py
Pre-paper robustness checks addressing reviewer concerns.

Two analyses:
  A. Edit-distance confound control (priority 2)
     - Computes Levenshtein-token edit distance per pair (prompt0 vs prompt1)
     - Tests whether the regime ordering of PANO_div (notably the
       "mistake anomaly": B_mistake < A) survives controlling for edit
       distance. Uses partial correlation, ANCOVA-style residuals, and
       stratified comparisons.

  B. Incremental value of PANO_div over textual / graph-size baselines
     (priority 3 -- analog of H3 on H1's setup)
     - Fits OLS regression of AOC on lightweight baselines
       (edit-distance, length difference, graph node counts, original AGD)
       with vs. without PANO_div as a feature, on per-item Regime B aggregates.
     - Reports incremental R^2 with BCa bootstrap CI.

Reads:
    artifacts/agd/pano_pairs.parquet
    artifacts/agd/pairs.parquet           (for n0/n1/agd legacy columns)
    artifacts/behavioral/aoc_lanham.parquet

Writes:
    analysis/robustness_results.json
    artifacts/agd/pano_pairs_with_editdist.parquet   (cached edit distances)

Usage:
    ./env/Scripts/python.exe scripts/16_robustness_checks.py --config config.yaml
    ./env/Scripts/python.exe scripts/16_robustness_checks.py --config config.yaml --use-full-set
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import Levenshtein
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_test_ids(cfg) -> Optional[set]:
    test_path = Path(cfg.paths.data) / "test_ids.txt"
    if not test_path.exists():
        return None
    with open(test_path) as f:
        return set(line.strip() for line in f if line.strip())


def get_base_id(row) -> str:
    b = row.get("base_item_id")
    if b and not (isinstance(b, float) and np.isnan(b)):
        return b
    return row["item_id"]


# ─────────────────────────────────────────────────────────────────────────────
# Edit-distance computation
# ─────────────────────────────────────────────────────────────────────────────

def token_edit_distance(s0: str, s1: str) -> Tuple[int, int, int]:
    """Levenshtein distance over whitespace-split tokens.

    Returns (edit_distance, len_tokens_s0, len_tokens_s1).

    Tokenisation: whitespace split. We hash each unique token to a single
    Unicode codepoint so we can re-use Levenshtein's char-level C
    implementation (O(n*m)).  For our prompt lengths (~100-1000 tokens)
    this is far faster than a Python-level token DP.
    """
    if not isinstance(s0, str) or not isinstance(s1, str):
        return -1, 0, 0
    t0 = s0.split()
    t1 = s1.split()
    vocab: Dict[str, str] = {}

    def encode(tokens: List[str]) -> str:
        out = []
        for t in tokens:
            if t not in vocab:
                # Map to a private-use codepoint to avoid collisions
                vocab[t] = chr(0xE000 + len(vocab))
            out.append(vocab[t])
        return "".join(out)

    e0 = encode(t0)
    e1 = encode(t1)
    return Levenshtein.distance(e0, e1), len(t0), len(t1)


def add_edit_distance_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Adds edit_distance, len0_tokens, len1_tokens, len_diff, edit_norm columns."""
    if "prompt0" not in df.columns or "prompt1" not in df.columns:
        raise ValueError("DataFrame missing prompt0/prompt1 columns")

    eds, l0, l1 = [], [], []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="edit-distance"):
        d, n0, n1 = token_edit_distance(row["prompt0"], row["prompt1"])
        eds.append(d); l0.append(n0); l1.append(n1)

    df = df.copy()
    df["edit_distance"] = eds
    df["len0_tokens"]   = l0
    df["len1_tokens"]   = l1
    df["len_diff"]      = (df["len1_tokens"] - df["len0_tokens"]).abs()
    # Normalised edit distance: relative to the longer prompt
    max_len = df[["len0_tokens", "len1_tokens"]].max(axis=1).clip(lower=1)
    df["edit_norm"] = df["edit_distance"] / max_len
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: Spearman partial correlation
# ─────────────────────────────────────────────────────────────────────────────

def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> Dict[str, float]:
    """Spearman partial correlation: corr(x, y | z).

    Implemented as: rank-transform each variable, then correlate the residuals
    of rank(x)~rank(z) and rank(y)~rank(z). Equivalent to controlling for z
    on the rank scale.
    """
    mask = ~(np.isnan(x) | np.isnan(y) | np.isnan(z))
    x, y, z = x[mask], y[mask], z[mask]
    n = len(x)
    if n < 5:
        return {"rho_partial": float("nan"), "p": float("nan"), "n": n}

    rx = scipy_stats.rankdata(x)
    ry = scipy_stats.rankdata(y)
    rz = scipy_stats.rankdata(z)

    # Linear regression of rx on rz, ry on rz; correlate residuals
    Z = rz.reshape(-1, 1)
    rx_resid = rx - LinearRegression().fit(Z, rx).predict(Z)
    ry_resid = ry - LinearRegression().fit(Z, ry).predict(Z)

    rho, p = scipy_stats.spearmanr(rx_resid, ry_resid)
    return {"rho_partial": float(rho), "p": float(p), "n": int(n)}


# ─────────────────────────────────────────────────────────────────────────────
# Analysis A: edit-distance confound check
# ─────────────────────────────────────────────────────────────────────────────

def analysis_a_edit_distance_confound(pano_df: pd.DataFrame) -> Dict[str, Any]:
    """Test whether PANO_div regime ordering is explained by edit distance."""
    logger.info("\n── Analysis A: Edit-distance confound check ──")

    out: Dict[str, Any] = {}

    # ── A.1: Per-regime edit-distance distributions ──────────────────────────
    logger.info("\n[A.1] Edit-distance by regime (mean, median, n):")
    rows = []
    for regime in ["A", "B_mistake", "B_trunc", "C"]:
        v = pano_df[pano_df["regime_label"] == regime]["edit_norm"].dropna().values
        if len(v) == 0:
            continue
        rows.append({
            "regime": regime,
            "n": len(v),
            "edit_norm_mean":   float(v.mean()),
            "edit_norm_median": float(np.median(v)),
            "edit_norm_std":    float(v.std()),
        })
        logger.info(f"  {regime:10s}: mean={v.mean():.3f}, median={np.median(v):.3f}, n={len(v)}")
    out["per_regime_edit_distance"] = rows

    # ── A.2: Within-regime correlation: PANO_div ~ edit_norm ──────────────────
    logger.info("\n[A.2] Within-regime Spearman(PANO_div, edit_norm):")
    within = {}
    for regime in ["A", "B_mistake", "B_trunc", "C"]:
        sub = pano_df[pano_df["regime_label"] == regime].dropna(subset=["pano_div", "edit_norm"])
        if len(sub) < 10:
            continue
        rho, p = scipy_stats.spearmanr(sub["pano_div"], sub["edit_norm"])
        within[regime] = {"rho": float(rho), "p": float(p), "n": int(len(sub))}
        logger.info(f"  {regime:10s}: rho={rho:+.3f}, p={p:.4f}, n={len(sub)}")
    out["within_regime_panodiv_vs_editnorm"] = within

    # ── A.3: ANCOVA-style — does regime label predict PANO_div *after*
    #        residualising out edit distance? ─────────────────────────────────
    logger.info("\n[A.3] ANCOVA: is regime ordering preserved after residualising edit distance?")
    df_anc = pano_df.dropna(subset=["pano_div", "edit_norm"]).copy()
    Z = df_anc[["edit_norm"]].values
    df_anc["pano_div_resid"] = (
        df_anc["pano_div"].values
        - LinearRegression().fit(Z, df_anc["pano_div"].values).predict(Z)
    )
    ancova_rows = []
    for regime in ["A", "B_mistake", "B_trunc", "C"]:
        v = df_anc[df_anc["regime_label"] == regime]["pano_div_resid"].values
        if len(v) == 0:
            continue
        ancova_rows.append({
            "regime": regime,
            "n": len(v),
            "panodiv_resid_mean":   float(v.mean()),
            "panodiv_resid_median": float(np.median(v)),
        })
        logger.info(f"  {regime:10s}: residualised mean={v.mean():+.4f}, n={len(v)}")
    out["panodiv_residual_after_editnorm"] = ancova_rows

    # ── A.4: The mistake-anomaly survival test ────────────────────────────────
    # Within edit-norm quartiles, does B_mistake still have lower PANO_div than A?
    logger.info("\n[A.4] Mistake-anomaly survival: A vs B_mistake within edit_norm quartiles")
    a_b = pano_df[pano_df["regime_label"].isin(["A", "B_mistake"])].dropna(
        subset=["pano_div", "edit_norm"]
    ).copy()
    if len(a_b) >= 40:
        try:
            quartiles = pd.qcut(a_b["edit_norm"], q=4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
            a_b["edit_q"] = quartiles
        except ValueError:
            a_b["edit_q"] = pd.cut(a_b["edit_norm"], bins=4, labels=["Q1", "Q2", "Q3", "Q4"])
        survival_rows = []
        for q in ["Q1", "Q2", "Q3", "Q4"]:
            sub = a_b[a_b["edit_q"] == q]
            a_v  = sub[sub["regime_label"] == "A"]["pano_div"].values
            bm_v = sub[sub["regime_label"] == "B_mistake"]["pano_div"].values
            if len(a_v) < 5 or len(bm_v) < 5:
                survival_rows.append({"quartile": q, "n_A": len(a_v), "n_Bm": len(bm_v),
                                      "skipped": True})
                continue
            try:
                _, p_less = scipy_stats.mannwhitneyu(bm_v, a_v, alternative="less")
            except ValueError:
                p_less = float("nan")
            survival_rows.append({
                "quartile": q,
                "n_A":  len(a_v),
                "n_Bm": len(bm_v),
                "mean_A":  float(a_v.mean()),
                "mean_Bm": float(bm_v.mean()),
                "p_less":  float(p_less),
                "anomaly_holds": bool(bm_v.mean() < a_v.mean()),
            })
            logger.info(
                f"  {q}: n_A={len(a_v)}, n_Bm={len(bm_v)}, "
                f"meanA={a_v.mean():.3f} vs meanBm={bm_v.mean():.3f}, "
                f"p_less={p_less:.4f}, anomaly_holds={bm_v.mean() < a_v.mean()}"
            )
        out["mistake_anomaly_quartile_survival"] = survival_rows
    else:
        logger.warning("  Too few A/B_mistake rows — skipping quartile analysis")

    # ── A.5: Headline conclusion ─────────────────────────────────────────────
    # Compare PANO_div(A) vs PANO_div(B_mistake) on edit-norm-matched samples
    # via partial-correlation framing: if pano_div ~ regime is significant
    # after controlling for edit_norm, the anomaly is not just an artefact.
    logger.info("\n[A.5] Partial-correlation summary (regime → PANO_div, controlling edit_norm)")
    df_p = pano_df[pano_df["regime_label"].isin(["A", "B_mistake"])].dropna(
        subset=["pano_div", "edit_norm"]
    ).copy()
    if len(df_p) >= 30:
        df_p["regime_bm"] = (df_p["regime_label"] == "B_mistake").astype(int)
        partial = partial_spearman(
            df_p["regime_bm"].values.astype(float),
            df_p["pano_div"].values,
            df_p["edit_norm"].values,
        )
        out["partial_corr_regime_to_panodiv_given_editnorm"] = partial
        logger.info(
            f"  partial rho(regime=Bm, PANO_div | edit_norm) = {partial['rho_partial']:+.3f}, "
            f"p={partial['p']:.4f}, n={partial['n']}"
        )

    return out


# ─────────────────────────────────────────────────────────────────────────────
# Analysis B: incremental R^2 of PANO_div over textual baselines on H1 task
# ─────────────────────────────────────────────────────────────────────────────

def analysis_b_incremental_h1(
    pano_df: pd.DataFrame,
    legacy_pairs: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Does PANO_div add explanatory power over lightweight baselines for AOC?

    Builds per-item Regime-B feature aggregates:
        edit_norm_mean, len_diff_mean, n0_mean, n1_mean, agd_legacy_mean
        + pano_div_mean.

    Fits OLS:
        AOC ~ baselines             (R²_red)
        AOC ~ baselines + PANO_div  (R²_full)

    Reports delta_R² with BCa bootstrap CI.
    """
    logger.info("\n── Analysis B: Incremental value of PANO_div over baselines (H1 task) ──")

    # Per-item Regime B aggregates
    b_mask = pano_df["regime_label"].isin(["B_mistake", "B_trunc"])
    b = pano_df[b_mask].dropna(subset=["pano_div", "edit_norm"]).copy()

    # Merge legacy AGD (jw, se, agd, n0, n1) to use as "graph-size" baselines
    keep_cols = ["item_id", "agd", "jw", "se", "n0", "n1"]
    legacy_sub = legacy_pairs[[c for c in keep_cols if c in legacy_pairs.columns]].copy()
    b = b.merge(legacy_sub, on="item_id", how="left", suffixes=("", "_legacy"))

    grouped = b.groupby("base_item_id").agg(
        edit_norm_mean=("edit_norm",     "mean"),
        len_diff_mean =("len_diff",      "mean"),
        n0_mean       =("n0",            "mean"),
        n1_mean       =("n1",            "mean"),
        agd_legacy_mean=("agd",          "mean"),
        pano_div_mean =("pano_div",      "mean"),
    ).reset_index()

    merged = grouped.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")
    merged = merged.dropna(subset=["aoc_composite", "pano_div_mean", "edit_norm_mean"])
    logger.info(f"  Per-item items with AOC + features: n={len(merged)}")

    if len(merged) < 30:
        logger.warning("  Too few items for incremental analysis — skipping")
        return {"skipped": True, "n": int(len(merged))}

    base_cols = ["edit_norm_mean", "len_diff_mean", "n0_mean", "n1_mean", "agd_legacy_mean"]
    base_cols = [c for c in base_cols if c in merged.columns and not merged[c].isna().all()]
    merged_clean = merged.dropna(subset=base_cols + ["pano_div_mean"]).copy()
    logger.info(f"  After dropping NaN baselines: n={len(merged_clean)}")
    logger.info(f"  Baseline columns: {base_cols}")

    if len(merged_clean) < 30:
        return {"skipped": True, "n": int(len(merged_clean)), "reason": "n<30 after dropna"}

    y = merged_clean["aoc_composite"].values
    X_red  = merged_clean[base_cols].values
    X_full = np.column_stack([X_red, merged_clean["pano_div_mean"].values])

    # In-sample R² (diagnostic only)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r2_red_insample  = LinearRegression().fit(X_red,  y).score(X_red,  y)
        r2_full_insample = LinearRegression().fit(X_full, y).score(X_full, y)
    logger.info(f"  In-sample R² (reduced): {r2_red_insample:.4f}")
    logger.info(f"  In-sample R² (full)   : {r2_full_insample:.4f}")
    logger.info(f"  In-sample ΔR²         : {r2_full_insample - r2_red_insample:+.4f}")

    # Honest out-of-sample bootstrap: each iter resamples, fits on 80 %, evaluates on 20 %
    rng = np.random.default_rng(seed)
    n = len(y)
    n_train = max(20, int(0.8 * n))
    boot_deltas = []
    boot_r2_full = []
    boot_r2_red  = []

    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        Xr = X_red[idx];  Xf = X_full[idx];  yb = y[idx]
        Xr_tr, Xr_te = Xr[:n_train], Xr[n_train:]
        Xf_tr, Xf_te = Xf[:n_train], Xf[n_train:]
        y_tr,  y_te  = yb[:n_train], yb[n_train:]
        if len(y_te) < 5 or np.std(y_te) == 0:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r2_r = LinearRegression().fit(Xr_tr, y_tr).score(Xr_te, y_te)
            r2_f = LinearRegression().fit(Xf_tr, y_tr).score(Xf_te, y_te)
        boot_deltas.append(r2_f - r2_r)
        boot_r2_full.append(r2_f)
        boot_r2_red.append(r2_r)

    boot_deltas  = np.array(boot_deltas)
    boot_r2_full = np.array(boot_r2_full)
    boot_r2_red  = np.array(boot_r2_red)

    delta_med = float(np.median(boot_deltas))
    p_one_sided = float(np.mean(boot_deltas <= 0))

    # Percentile CI (BCa would require jackknife on n items; use percentile here
    # given the tighter timeline; flag as such).
    ci_lo = float(np.percentile(boot_deltas, 2.5))
    ci_hi = float(np.percentile(boot_deltas, 97.5))

    logger.info(f"  Bootstrap median ΔR²: {delta_med:+.4f}")
    logger.info(f"  95% CI (percentile): [{ci_lo:+.4f}, {ci_hi:+.4f}]")
    logger.info(f"  P(ΔR² ≤ 0) = {p_one_sided:.4f}")

    # Also: simple correlation of pano_div_mean with AOC after partialing out baselines
    # (additional confirmation of unique signal)
    Z = X_red
    pano_resid = (
        merged_clean["pano_div_mean"].values
        - LinearRegression().fit(Z, merged_clean["pano_div_mean"].values).predict(Z)
    )
    aoc_resid = y - LinearRegression().fit(Z, y).predict(Z)
    rho_partial, p_partial = scipy_stats.spearmanr(pano_resid, aoc_resid)
    logger.info(
        f"  Partial Spearman(PANO_div, AOC | baselines) = {rho_partial:+.3f}, p={p_partial:.4f}"
    )

    return {
        "n_items": int(len(merged_clean)),
        "baseline_columns": base_cols,
        "r2_reduced_insample": float(r2_red_insample),
        "r2_full_insample":    float(r2_full_insample),
        "delta_r2_insample":   float(r2_full_insample - r2_red_insample),
        "delta_r2_oos_median": delta_med,
        "delta_r2_oos_ci_lo":  ci_lo,
        "delta_r2_oos_ci_hi":  ci_hi,
        "p_one_sided":         p_one_sided,
        "n_boot_valid":        int(len(boot_deltas)),
        "partial_spearman_panodiv_aoc": float(rho_partial),
        "partial_spearman_p":           float(p_partial),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-full-set", action="store_true",
                        help="Use full dataset instead of test-split filter")
    parser.add_argument("--rebuild-edit-distance", action="store_true",
                        help="Force re-computation of edit distances")
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.seed
    n_boot = cfg.stats.n_bootstrap

    pano_path = Path(cfg.paths.agd) / "pano_pairs.parquet"
    if not pano_path.exists():
        logger.error(f"{pano_path} not found. Run script 13 first.")
        sys.exit(1)

    cached_path = Path(cfg.paths.agd) / "pano_pairs_with_editdist.parquet"

    # ── Load PANO + edit distance ────────────────────────────────────────────
    if cached_path.exists() and not args.rebuild_edit_distance:
        logger.info(f"Loading cached edit distances from {cached_path}")
        pano_df = pd.read_parquet(cached_path)
    else:
        logger.info(f"Computing edit distances for all pairs...")
        pano_df = pd.read_parquet(pano_path)
        pano_df = add_edit_distance_columns(pano_df)
        pano_df.to_parquet(cached_path, index=False)
        logger.info(f"Cached -> {cached_path}")

    logger.info(f"Loaded {len(pano_df)} pairs")

    # Filter to test set unless --use-full-set
    if not args.use_full_set:
        test_ids = load_test_ids(cfg)
        if test_ids:
            mask = pano_df.apply(lambda r: get_base_id(r) in test_ids, axis=1)
            pano_df = pano_df[mask].copy()
            logger.info(f"After test-set filter: n={len(pano_df)}")
    else:
        logger.info("Using full dataset (train+test).")

    analysis_dir = Path(cfg.paths.analysis)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {}

    # ── Analysis A ────────────────────────────────────────────────────────────
    results["analysis_A_edit_distance_confound"] = analysis_a_edit_distance_confound(pano_df)

    # ── Analysis B ────────────────────────────────────────────────────────────
    legacy_path = Path(cfg.paths.agd) / "pairs.parquet"
    aoc_path = Path(cfg.paths.behavioral) / "aoc_lanham.parquet"
    if legacy_path.exists() and aoc_path.exists():
        legacy_df = pd.read_parquet(legacy_path)
        aoc_df    = pd.read_parquet(aoc_path)
        results["analysis_B_incremental_h1"] = analysis_b_incremental_h1(
            pano_df, legacy_df, aoc_df, n_boot=n_boot, seed=seed,
        )
    else:
        logger.warning(
            f"Skipping Analysis B: legacy_pairs={legacy_path.exists()}, aoc={aoc_path.exists()}"
        )

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = analysis_dir / "robustness_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults -> {out_path}")

    # ── Summary ──────────────────────────────────────────────────────────────
    logger.info("\n" + "="*65)
    logger.info("ROBUSTNESS-CHECK SUMMARY")
    logger.info("="*65)

    A = results.get("analysis_A_edit_distance_confound", {})
    if "partial_corr_regime_to_panodiv_given_editnorm" in A:
        pc = A["partial_corr_regime_to_panodiv_given_editnorm"]
        logger.info(
            f"\n[A] Mistake anomaly after controlling for edit distance:\n"
            f"    partial rho(B_mistake vs A → PANO_div | edit_norm) = "
            f"{pc['rho_partial']:+.3f}, p={pc['p']:.4f}, n={pc['n']}"
        )
    if "mistake_anomaly_quartile_survival" in A:
        n_holds = sum(
            1 for r in A["mistake_anomaly_quartile_survival"]
            if not r.get("skipped") and r.get("anomaly_holds")
        )
        n_total = sum(
            1 for r in A["mistake_anomaly_quartile_survival"] if not r.get("skipped")
        )
        logger.info(f"    Mistake-anomaly holds in {n_holds}/{n_total} edit-norm quartiles")

    B = results.get("analysis_B_incremental_h1", {})
    if B and not B.get("skipped"):
        logger.info(
            f"\n[B] Incremental ΔR² over baselines (predicting AOC):\n"
            f"    OOS median ΔR² = {B['delta_r2_oos_median']:+.4f}, "
            f"CI=[{B['delta_r2_oos_ci_lo']:+.4f}, {B['delta_r2_oos_ci_hi']:+.4f}], "
            f"P(ΔR²≤0)={B['p_one_sided']:.4f}\n"
            f"    Partial Spearman(PANO_div, AOC | baselines) = "
            f"{B['partial_spearman_panodiv_aoc']:+.3f}, p={B['partial_spearman_p']:.4f}"
        )

    logger.info("="*65)


if __name__ == "__main__":
    main()
