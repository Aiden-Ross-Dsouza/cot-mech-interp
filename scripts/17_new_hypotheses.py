"""
scripts/17_new_hypotheses.py
Replacement H2 and H3 hypotheses extending H1, plus cross-dataset generalization.

Replaces the hint-flip H2 (Regime C) and incremental-AUROC H3, both of which
failed or had CIs crossing zero.  All three analyses use only existing artifacts
-- no new GPU graph generation required.

H2-new (Truncation-Depth Monotonicity):
    For faithful items, PANO_div should increase monotonically as more of the
    CoT is truncated (25% → 50% → 75%).  For unfaithful items the model's
    final answer doesn't depend on the CoT, so PANO_div stays flat.
    Test: per-item Kendall τ(truncation_fraction, PANO_div), then
          Spearman ρ(τ_per_item, AOC_composite) ≥ 0.30 on test split.

H3-new (Head-to-Head Baseline Comparison):
    PANO_div has a strictly higher Spearman ρ(., AOC) than each of four
    textual / graph-size baselines: edit_norm, len_diff, n_concepts_0,
    n_concepts_1.  Reported as paired-bootstrap 95% CI of (ρ_PANO - ρ_baseline).
    CI excluding zero = PANO_div uniquely outperforms that baseline.

Supporting (Cross-Dataset Generalization):
    Per-dataset breakdown of H1 ρ on the test split: BBH / MMLU / GSM8K.
    Summarised as descriptive evidence that H1 is not dataset-specific.

Reads:
    artifacts/agd/pano_pairs_with_editdist.parquet
    artifacts/behavioral/aoc_lanham.parquet
    data/test_ids.txt

Writes:
    analysis/results_new_h2_h3.json

Usage:
    python scripts/17_new_hypotheses.py --config config.yaml
    python scripts/17_new_hypotheses.py --config config.yaml --use-full-set
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci, holm_bonferroni


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_test_ids(cfg) -> Optional[set]:
    test_path = Path(cfg.paths.data) / "test_ids.txt"
    if not test_path.exists():
        logger.error("test_ids.txt not found.  Run data/split.py first.")
        sys.exit(1)
    with open(test_path) as f:
        return set(line.strip() for line in f if line.strip())


def get_base_id(row) -> str:
    b = row.get("base_item_id")
    if b and not (isinstance(b, float) and np.isnan(b)):
        return b
    return row["item_id"]


def infer_dataset(item_id: str) -> str:
    s = str(item_id).lower()
    if s.startswith("bbh"):
        return "BBH"
    if s.startswith("mmlu"):
        return "MMLU"
    if s.startswith("gsm"):
        return "GSM8K"
    if s.startswith("turpin"):
        return "Turpin"
    return "Unknown"


# ─────────────────────────────────────────────────────────────────────────────
# Paired-bootstrap CI for ρ_A - ρ_B (Spearman difference)
# ─────────────────────────────────────────────────────────────────────────────

def paired_spearman_diff_ci(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """BCa bootstrap CI for (Spearman(x_a, y) - Spearman(x_b, y)).

    All three arrays are paired observations of equal length.
    NaNs are dropped row-wise across all three.
    """
    mask = ~(np.isnan(x_a) | np.isnan(x_b) | np.isnan(y))
    x_a, x_b, y = x_a[mask], x_b[mask], y[mask]
    n = len(y)

    rho_a = scipy_stats.spearmanr(x_a, y)[0]
    rho_b = scipy_stats.spearmanr(x_b, y)[0]
    observed = rho_a - rho_b

    rng = np.random.default_rng(seed)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        d_a = scipy_stats.spearmanr(x_a[idx], y[idx])[0]
        d_b = scipy_stats.spearmanr(x_b[idx], y[idx])[0]
        boot_diffs.append(d_a - d_b)
    boot_diffs = np.array(boot_diffs)

    # BCa
    z0 = scipy_stats.norm.ppf(np.mean(boot_diffs < observed) + 1e-12)
    jack = np.array([
        scipy_stats.spearmanr(np.delete(x_a, i), np.delete(y, i))[0]
        - scipy_stats.spearmanr(np.delete(x_b, i), np.delete(y, i))[0]
        for i in range(n)
    ])
    jm = np.mean(jack)
    num = np.sum((jm - jack) ** 3)
    denom = 6 * (np.sum((jm - jack) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo = float(np.percentile(boot_diffs, np.clip(_adj(z_lo), 0, 100)))
    ci_hi = float(np.percentile(boot_diffs, np.clip(_adj(z_hi), 0, 100)))

    return {
        "rho_pano": float(rho_a),
        "rho_baseline": float(rho_b),
        "diff": float(observed),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": bool(ci_lo > 0),
        "n": int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2-new: Truncation-depth monotonicity
# ─────────────────────────────────────────────────────────────────────────────

def run_h2_new_monotonicity(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
    rho_threshold: float = 0.30,
) -> Dict:
    """Per-item Kendall τ(trunc_fraction, PANO_div) then Spearman(τ, AOC)."""
    b_trunc = pano_df[
        (pano_df["regime_label"] == "B_trunc") &
        pano_df["truncation_fraction"].notna() &
        pano_df["pano_div"].notna()
    ].copy()

    logger.info(f"H2-new: B_trunc rows with truncation_fraction: {len(b_trunc)}")

    # Per-item: compute Kendall tau between truncation_fraction and pano_div
    records = []
    for base_id, grp in b_trunc.groupby("base_item_id"):
        grp = grp.sort_values("truncation_fraction")
        fracs = grp["truncation_fraction"].values
        divs = grp["pano_div"].values

        if len(fracs) < 2:
            continue  # need at least 2 points for tau

        tau, p_tau = scipy_stats.kendalltau(fracs, divs)
        records.append({
            "base_item_id": base_id,
            "kendall_tau": float(tau),
            "p_tau": float(p_tau),
            "n_trunc_points": len(fracs),
        })

    tau_df = pd.DataFrame(records)
    logger.info(
        f"H2-new: {len(tau_df)} items with per-item Kendall τ "
        f"(≥2 truncation points)"
    )

    # Merge with AOC
    merged = tau_df.merge(
        aoc_df[["item_id", "aoc_composite"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["kendall_tau", "aoc_composite"])

    logger.info(f"H2-new: {len(merged)} items with both τ and AOC")

    if len(merged) < 20:
        logger.warning("H2-new: too few items — skipping.")
        return {"status": "skipped", "reason": f"n={len(merged)} < 20"}

    result = spearman_with_ci(
        merged["kendall_tau"].values,
        merged["aoc_composite"].values,
        n_boot=n_boot, seed=seed,
    )
    result["threshold"] = rho_threshold
    result["passed"] = bool(result["rho"] >= rho_threshold)

    # Descriptive: fraction with τ > 0 (monotonically increasing)
    result["frac_items_positive_tau"] = float((merged["kendall_tau"] > 0).mean())
    result["mean_tau"] = float(merged["kendall_tau"].mean())
    result["n_trunc_points_dist"] = merged["n_trunc_points"].value_counts().to_dict()

    logger.info(
        f"H2-new: ρ(τ, AOC) = {result['rho']:.4f}, p={result['p']:.4f}, "
        f"CI=[{result['ci_lo']:.4f}, {result['ci_hi']:.4f}], n={result['n']}"
    )
    logger.info(
        f"H2-new: mean τ={result['mean_tau']:.4f}, "
        f"frac τ>0={result['frac_items_positive_tau']:.3f}"
    )
    logger.info(f"H2-new passed (ρ ≥ {rho_threshold}): {result['passed']}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# H3-new: Head-to-head baseline comparison
# ─────────────────────────────────────────────────────────────────────────────

def run_h3_new_headtohead(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict:
    """Paired bootstrap CI: ρ(PANO_div, AOC) - ρ(baseline, AOC) per baseline."""
    b_df = pano_df[pano_df["regime_label"].isin(["B_mistake", "B_trunc"])].copy()

    baseline_cols = {
        "edit_norm": "Normalised edit distance",
        "len_diff": "CoT length difference (tokens)",
        "n_concepts_0": "Clean graph node count",
        "n_concepts_1": "Perturbed graph node count",
    }
    # Keep only available and non-constant cols (n_concepts = k=64 always)
    available = {
        k: v for k, v in baseline_cols.items()
        if k in b_df.columns and b_df[k].nunique() > 1
    }
    if not available:
        return {"status": "skipped", "reason": "No baseline columns found in parquet"}

    # Per-item aggregation: mean over truncation variants
    agg_dict = {"pano_div": "mean"}
    for col in available:
        agg_dict[col] = "mean"

    per_item = b_df.groupby("base_item_id").agg(agg_dict).reset_index()
    per_item.columns = ["base_item_id"] + [
        "pano_div_mean"] + [f"{c}_mean" for c in available]

    merged = per_item.merge(
        aoc_df[["item_id", "aoc_composite"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["pano_div_mean", "aoc_composite"])

    logger.info(f"H3-new: {len(merged)} items for head-to-head comparison")

    if len(merged) < 20:
        return {"status": "skipped", "reason": f"n={len(merged)} < 20"}

    y = merged["aoc_composite"].values
    x_pano = merged["pano_div_mean"].values

    rho_pano = spearman_with_ci(x_pano, y, n_boot=n_boot, seed=seed)
    logger.info(
        f"H3-new: ρ(PANO_div, AOC) = {rho_pano['rho']:.4f}, "
        f"CI=[{rho_pano['ci_lo']:.4f}, {rho_pano['ci_hi']:.4f}]"
    )

    comparisons = {}
    n_beats = 0
    for col, label in available.items():
        x_base = merged[f"{col}_mean"].values
        diff_result = paired_spearman_diff_ci(
            x_pano, x_base, y, n_boot=n_boot, seed=seed
        )
        diff_result["baseline_label"] = label
        comparisons[col] = diff_result
        beats = diff_result["ci_excludes_zero"]
        n_beats += int(beats)
        logger.info(
            f"  vs {col}: diff={diff_result['diff']:+.4f}, "
            f"CI=[{diff_result['ci_lo']:+.4f}, {diff_result['ci_hi']:+.4f}], "
            f"CI excl. 0: {beats}"
        )

    return {
        "pano_spearman": rho_pano,
        "baseline_comparisons": comparisons,
        "n_baselines_beaten": n_beats,
        "n_baselines_total": len(available),
        "summary": f"PANO_div strictly outperforms {n_beats}/{len(available)} baselines (CI excludes zero)",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Supporting: Cross-dataset H1 generalization
# ─────────────────────────────────────────────────────────────────────────────

def run_cross_dataset(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict:
    """Per-dataset Spearman ρ(PANO_div, AOC) on Regime B."""
    b_df = pano_df[pano_df["regime_label"].isin(["B_mistake", "B_trunc"])].copy()
    b_df["dataset"] = b_df["base_item_id"].apply(infer_dataset)

    per_item = (
        b_df.groupby(["base_item_id", "dataset"])["pano_div"]
        .mean().reset_index()
        .rename(columns={"pano_div": "pano_div_mean"})
    )
    merged = per_item.merge(
        aoc_df[["item_id", "aoc_composite"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["pano_div_mean", "aoc_composite"])

    results = {}
    for ds, grp in merged.groupby("dataset"):
        n = len(grp)
        if n < 10:
            logger.info(f"  Cross-dataset: {ds} skipped (n={n} < 10)")
            results[ds] = {"n": n, "status": "skipped"}
            continue
        r = spearman_with_ci(
            grp["pano_div_mean"].values,
            grp["aoc_composite"].values,
            n_boot=n_boot, seed=seed,
        )
        results[ds] = r
        logger.info(
            f"  {ds} (n={n}): ρ={r['rho']:.4f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}]"
        )

    all_rhos = [v["rho"] for v in results.values() if "rho" in v]
    all_positive = all(r > 0 for r in all_rhos)
    results["summary"] = {
        "all_datasets_positive_rho": all_positive,
        "datasets_with_n_ge_10": [k for k, v in results.items() if "rho" in v],
    }
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-full-set", action="store_true",
                        help="Use all items (train+test) instead of held-out test half only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_boot = cfg.stats.n_bootstrap

    # Load edit-dist parquet (has all columns we need)
    ed_path = Path(cfg.paths.agd) / "pano_pairs_with_editdist.parquet"
    pano_path = Path(cfg.paths.agd) / "pano_pairs.parquet"

    if ed_path.exists():
        pano_df = pd.read_parquet(ed_path)
        logger.info(f"Loaded pano_pairs_with_editdist: {len(pano_df)} rows")
    elif pano_path.exists():
        pano_df = pd.read_parquet(pano_path)
        logger.warning("Edit-dist parquet not found; falling back to pano_pairs (no edit_norm/len_diff)")
    else:
        logger.error("No pano_pairs parquet found. Run scripts 13 and 16 first.")
        sys.exit(1)

    # Test-set filter
    if not args.use_full_set:
        test_ids = load_test_ids(cfg)
        test_mask = pano_df.apply(lambda r: get_base_id(r) in test_ids, axis=1)
        pano_df = pano_df[test_mask].copy()
        logger.info(f"After test-set filter: {len(pano_df)} rows")
    else:
        logger.info("Using full dataset (train+test) — discovery/sensitivity mode.")

    aoc_df = pd.read_parquet(Path(cfg.paths.behavioral) / "aoc_lanham.parquet")
    logger.info(f"AOC items: {len(aoc_df)}")

    results: Dict = {}
    p_values: List[float] = []
    test_names: List[str] = []

    # ── H2-new: Truncation-depth monotonicity ─────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("H2-new: Truncation-depth monotonicity")
    logger.info("="*60)
    h2_result = run_h2_new_monotonicity(
        pano_df, aoc_df, n_boot=n_boot, seed=cfg.seed,
        rho_threshold=cfg.stats.h1_rho_threshold,
    )
    results["H2_new_monotonicity"] = h2_result
    if "p" in h2_result:
        p_values.append(h2_result["p"])
        test_names.append("H2_new_monotonicity")

    # ── H3-new: Head-to-head baseline comparison ──────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("H3-new: Head-to-head baseline comparison")
    logger.info("="*60)
    h3_result = run_h3_new_headtohead(
        pano_df, aoc_df, n_boot=n_boot, seed=cfg.seed,
    )
    results["H3_new_headtohead"] = h3_result

    # ── Supporting: Cross-dataset generalization ──────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Supporting: Cross-dataset H1 generalization")
    logger.info("="*60)
    cross_result = run_cross_dataset(
        pano_df, aoc_df, n_boot=n_boot, seed=cfg.seed,
    )
    results["cross_dataset_h1"] = cross_result

    # ── Holm-Bonferroni ───────────────────────────────────────────────────────
    if p_values:
        hb = holm_bonferroni(p_values, alpha=cfg.stats.alpha_family)
        results["holm_bonferroni"] = {
            "test_names": test_names,
            "raw_p": p_values,
            **hb,
        }
        for name, rp, cp, rej in zip(
            test_names, p_values, hb["corrected_p"], hb["rejected"]
        ):
            logger.info(
                f"  HB: {name}: raw_p={rp:.4f}, corr_p={cp:.4f}, reject={rej}"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("SUMMARY — New H2 / H3")
    logger.info("="*60)
    if "rho" in h2_result:
        logger.info(
            f"  H2-new PASSED: {h2_result['passed']}  "
            f"(ρ={h2_result['rho']:.4f}, threshold={h2_result['threshold']})"
        )
    if "n_baselines_beaten" in h3_result:
        logger.info(f"  H3-new: {h3_result['summary']}")
    if "summary" in cross_result:
        logger.info(
            f"  Cross-dataset: all datasets positive ρ = "
            f"{cross_result['summary']['all_datasets_positive_rho']}"
        )

    out_path = Path(cfg.paths.analysis) / "results_new_h2_h3.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    main()
