"""
scripts/23_day4_analysis.py
Day 4 analysis per research_plan_v3 (v2.2 framing).

Tasks (all on existing Gemma graphs — no new GPU compute needed):

  Task A — GRM exploration (§2.5):
    Compute Global Reorganization Magnitude on all Regime-B mistake pairs.
    Correlate with aoc_mistake. Decision gate: ρ ≥ 0.20 → GRM enters paper;
    ρ < 0.20 → GRM dropped. Decision logged to results JSON.

  Task B — ED failure characterization (§4, Day 4):
    Expand locality sanity check from n=10 to full corpus.
    Compute locality fractions at f ∈ {0.05, 0.10, 0.20, 0.30}.
    Compute layer-band shift profile: where does the diffuse shift live?
    Output: table for paper Appendix D.

  Task C — HTIR anti-prediction characterization (§4, Day 4):
    Bootstrap CI on HTIR_flipped − HTIR_nonflipped.
    Per-subtask AUROC breakdown.
    Tail analysis: is anti-prediction global or driven by outliers?
    Output: material for Appendix E.

  Task D — Headline BCa CIs + Holm-Bonferroni:
    Lock the family of hypothesis tests:
      H1a: ρ(GRACE-T, aoc_truncate_50) > 0
      H1b: ρ(GRACE-T, aoc_composite) > 0
      H2a: ρ(GRACE-T, aoc_mistake) ≈ 0 (null)
      H3: ρ(GRM, aoc_mistake) > 0 [only if GRM survived Task A]
    BCa CI (5000 resamples) on each. Holm-Bonferroni across family.
    Output: Table 1 numbers with CIs.

Reads:
    data/pairs/regime_B_addmistake.jsonl
    data/pairs/regime_C_hint.jsonl
    artifacts/graphs/*.json
    artifacts/agd/pano_pairs_with_editdist.parquet
    artifacts/agd/ed_results.parquet         (from script 22)
    artifacts/agd/htir_results.parquet       (from script 22)
    artifacts/behavioral/aoc_lanham.parquet
    artifacts/behavioral/turpin_flips.parquet

Writes:
    artifacts/agd/grm_results.parquet        — GRM per B_mistake pair
    analysis/results_day4.json               — all Task A–D results
    analysis/figures/ed_locality_profile.png — locality vs f, by band
    analysis/figures/htir_subtask_auroc.png  — per-subtask AUROC bar chart

Usage:
    python scripts/23_day4_analysis.py --config config.yaml
    python scripts/23_day4_analysis.py --config config.yaml --grm-threshold 0.20
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph_utils import load_graph
from src.pano import (
    compute_grm,
    get_layer_shift_profile,
    compute_ed,
    get_position_influence,
    get_node_position,
)
from src.stats import (
    spearman_with_ci,
    auroc_with_ci,
    holm_bonferroni,
    bootstrap_bca_2sample,
    cliffs_delta,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

import jsonlines

MISTAKE_MARKER = "Wait, actually that's wrong."
CHARS_PER_TOKEN = 3.8


# ─────────────────────────────────────────────────────────────────────────────
# Task A — GRM computation + correlation with aoc_mistake
# ─────────────────────────────────────────────────────────────────────────────

def compute_grm_for_pairs(
    mistake_pairs: List[Dict],
    graph_dir: Path,
) -> pd.DataFrame:
    rows = []
    for pair in tqdm(mistake_pairs, desc="GRM computation"):
        iid = pair["item_id"]
        p0 = graph_dir / f"{iid}_clean.json"
        p1 = graph_dir / f"{iid}_addmistake.json"
        row = {"item_id": iid, "base_item_id": pair.get("base_item_id"), "task": pair.get("task")}
        if not p0.exists() or not p1.exists():
            row["grm"] = np.nan
            rows.append(row)
            continue
        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            row["grm"] = compute_grm(g0, g1)
            band_profile = get_layer_shift_profile(g0, g1)
            row.update({f"shift_frac_{k}": v for k, v in band_profile.items()})
        except Exception as e:
            logger.warning(f"GRM failed {iid}: {e}")
            row["grm"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def task_a_grm(
    mistake_pairs: List[Dict],
    graph_dir: Path,
    aoc_df: Optional[pd.DataFrame],
    grm_threshold: float,
    agd_dir: Path,
) -> Dict[str, Any]:
    logger.info("\n=== Task A: GRM exploration ===")
    grm_df = compute_grm_for_pairs(mistake_pairs, graph_dir)

    agd_dir.mkdir(parents=True, exist_ok=True)
    grm_df.to_parquet(agd_dir / "grm_results.parquet", index=False)
    logger.info(f"GRM parquet → {agd_dir}/grm_results.parquet  ({len(grm_df)} rows)")

    valid_grm = grm_df["grm"].dropna()
    summary = {
        "n_valid": int(len(valid_grm)),
        "n_nan":   int(grm_df["grm"].isna().sum()),
        "mean":    float(valid_grm.mean()),
        "std":     float(valid_grm.std()),
        "median":  float(valid_grm.median()),
    }
    logger.info(f"GRM stats: mean={summary['mean']:.4f}, median={summary['median']:.4f}, std={summary['std']:.4f}")

    correlations = {}
    if aoc_df is not None:
        aoc_merge = aoc_df.copy()
        if "base_item_id" not in aoc_merge.columns:
            aoc_merge["base_item_id"] = aoc_merge["item_id"]
        merged = grm_df.merge(aoc_merge, on="base_item_id", how="inner")
        aoc_cols = [c for c in merged.columns if c.startswith("aoc_")]
        for aoc_c in aoc_cols:
            valid = merged[["grm", aoc_c]].dropna()
            if len(valid) < 10:
                continue
            res = spearman_with_ci(valid["grm"].values, valid[aoc_c].values)
            correlations[aoc_c] = res
            logger.info(f"  ρ(GRM, {aoc_c}) = {res['rho']:.3f} [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}], p={res['p']:.4f}, n={res['n']}")

    # Decision gate
    mistake_rho = correlations.get("aoc_mistake", {}).get("rho", 0.0)
    grm_survives = float(mistake_rho) >= grm_threshold
    decision = "KEEP" if grm_survives else "DROP"
    logger.info(f"\nGRM decision: ρ(GRM, aoc_mistake)={mistake_rho:.3f} {'≥' if grm_survives else '<'} {grm_threshold} → {decision}")

    # Layer-band shift profile summary
    band_cols = [c for c in grm_df.columns if c.startswith("shift_frac_")]
    band_means = {c: float(grm_df[c].mean()) for c in band_cols if grm_df[c].notna().any()}
    logger.info(f"Mean layer-band shift fractions: {band_means}")

    return {
        "summary": summary,
        "correlations": correlations,
        "grm_threshold": grm_threshold,
        "grm_survives": grm_survives,
        "decision": decision,
        "layer_band_shift_means": band_means,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task B — ED failure characterization (full corpus, multi-f, layer profile)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ed_full_characterization(
    mistake_pairs: List[Dict],
    graph_dir: Path,
    f_values: List[float] = (0.05, 0.10, 0.20, 0.30),
) -> pd.DataFrame:
    rows = []
    for pair in tqdm(mistake_pairs, desc="ED full characterization"):
        iid = pair["item_id"]
        p0 = graph_dir / f"{iid}_clean.json"
        p1 = graph_dir / f"{iid}_addmistake.json"
        row = {"item_id": iid, "base_item_id": pair.get("base_item_id"), "task": pair.get("task")}

        if not p0.exists() or not p1.exists():
            for f in f_values:
                row[f"ed_f{f:.2f}"] = np.nan
            rows.append(row)
            continue

        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            prompt0 = str(pair.get("prompt0", ""))
            prompt1 = str(pair.get("prompt1", ""))
            char_pos = prompt1.find(MISTAKE_MARKER)

            if char_pos < 0:
                for f in f_values:
                    row[f"ed_f{f:.2f}"] = np.nan
                row["mistake_found"] = False
            else:
                token_pos = int(char_pos / CHARS_PER_TOKEN)
                cot_len   = max(1, int(len(prompt0) / CHARS_PER_TOKEN))
                row["mistake_token_pos"] = token_pos
                row["cot_len_tokens"]   = cot_len
                row["mistake_found"]    = True
                for f in f_values:
                    row[f"ed_f{f:.2f}"] = compute_ed(g0, g1, token_pos, cot_len, f=f)

            # Layer-band shift profile
            band_profile = get_layer_shift_profile(g0, g1)
            row.update({f"shift_frac_{k}": v for k, v in band_profile.items()})

            # GRM (the total shift magnitude)
            row["grm"] = compute_grm(g0, g1)

        except Exception as e:
            logger.warning(f"ED characterization failed {iid}: {e}")
            for f in f_values:
                row[f"ed_f{f:.2f}"] = np.nan

        rows.append(row)
    return pd.DataFrame(rows)


def task_b_ed_failure(
    mistake_pairs: List[Dict],
    graph_dir: Path,
    f_values: List[float],
    fig_dir: Path,
) -> Dict[str, Any]:
    logger.info("\n=== Task B: ED failure characterization (full corpus) ===")
    df = compute_ed_full_characterization(mistake_pairs, graph_dir, f_values)

    results: Dict[str, Any] = {
        "n_pairs": len(df),
        "n_mistake_found": int(df.get("mistake_found", pd.Series(dtype=bool)).sum()),
    }

    # ED stats per f-value
    ed_stats = {}
    for f in f_values:
        col = f"ed_f{f:.2f}"
        if col not in df.columns:
            continue
        v = df[col].dropna()
        ed_stats[f"f={f}"] = {
            "n_valid": int(len(v)),
            "mean":    float(v.mean()),
            "median":  float(v.median()),
            "std":     float(v.std()),
            "pct_below_random_expectation": float((v < (1.0 / df["cot_len_tokens"].dropna().mean())).mean()) if "cot_len_tokens" in df.columns else float("nan"),
        }
        logger.info(f"  ED f={f}: n={len(v)}, mean={v.mean():.4f}, median={v.median():.5f}")

    results["ed_by_f"] = ed_stats

    # Layer-band shift profile (where does the diffuse shift live?)
    band_cols = [c for c in df.columns if c.startswith("shift_frac_")]
    band_summary = {}
    for col in band_cols:
        band = col.replace("shift_frac_", "")
        v = df[col].dropna()
        band_summary[band] = {"mean": float(v.mean()), "std": float(v.std()), "median": float(v.median())}
        logger.info(f"  Shift fraction in {band}: mean={v.mean():.3f} ± {v.std():.3f}")
    results["layer_band_shift"] = band_summary

    # Figure: ED median by f, and shift fractions by band
    if HAS_MPL and len(f_values) >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        # Left: ED median by f
        medians = [df.get(f"ed_f{f:.2f}", pd.Series()).median() for f in f_values]
        axes[0].bar([str(f) for f in f_values], medians, color="steelblue")
        axes[0].set_xlabel("Window fraction f")
        axes[0].set_ylabel("Median ED")
        axes[0].set_title("ED median by window size\n(all near 0 — locality assumption falsified)")
        axes[0].axhline(0, color="black", linewidth=0.5)

        # Right: Band shift fractions
        if band_summary:
            bands = list(band_summary.keys())
            means = [band_summary[b]["mean"] for b in bands]
            axes[1].bar(bands, means, color=["#4CAF50", "#FF9800", "#F44336", "#9E9E9E"][:len(bands)])
            axes[1].set_xlabel("Depth band")
            axes[1].set_ylabel("Mean fraction of total shift")
            axes[1].set_title("Where does the diffuse shift live?\n(mistake-injection, full corpus)")

        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "ed_locality_profile.png", dpi=150)
        plt.close(fig)
        logger.info(f"Figure → {fig_dir}/ed_locality_profile.png")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Task C — HTIR anti-prediction characterization
# ─────────────────────────────────────────────────────────────────────────────

def task_c_htir_characterization(
    htir_df: pd.DataFrame,
    n_boot: int = 5000,
    fig_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    logger.info("\n=== Task C: HTIR anti-prediction characterization ===")
    results: Dict[str, Any] = {}

    valid = htir_df[["htir", "flipped", "task"]].dropna(subset=["htir", "flipped"])
    valid = valid.copy()
    valid["flipped"] = valid["flipped"].astype(bool)

    flip_htir   = valid[valid["flipped"]]["htir"].values
    noflip_htir = valid[~valid["flipped"]]["htir"].values

    results["n_flipped"]    = int(len(flip_htir))
    results["n_not_flipped"] = int(len(noflip_htir))
    results["mean_htir_flip"]    = float(flip_htir.mean()) if len(flip_htir) else float("nan")
    results["mean_htir_no_flip"] = float(noflip_htir.mean()) if len(noflip_htir) else float("nan")

    # Bootstrap CI on the difference (non-flipped − flipped)
    if len(flip_htir) >= 5 and len(noflip_htir) >= 5:
        diff_obs, ci_lo, ci_hi = bootstrap_bca_2sample(
            lambda x, y: float(np.mean(y) - np.mean(x)),
            flip_htir, noflip_htir, n_boot=n_boot,
        )
        results["mean_diff_noflip_minus_flip"] = diff_obs
        results["ci_diff"] = [ci_lo, ci_hi]
        results["cliffs_delta"] = float(cliffs_delta(noflip_htir, flip_htir))
        logger.info(
            f"HTIR diff (no_flip − flip) = {diff_obs:.4f}  "
            f"BCa CI [{ci_lo:.4f}, {ci_hi:.4f}]  |  Cliff's δ={results['cliffs_delta']:.3f}"
        )

    # Overall AUROC with CI
    if valid["flipped"].nunique() == 2:
        auc_res = auroc_with_ci(valid["htir"].values, valid["flipped"].astype(int).values, n_boot=n_boot)
        results["auroc"] = auc_res
        logger.info(f"AUROC = {auc_res['auc']:.3f}  BCa CI [{auc_res['ci_lo']:.3f}, {auc_res['ci_hi']:.3f}]")

    # Per-subtask AUROC
    subtask_auroc = {}
    for task, grp in valid.groupby("task"):
        if grp["flipped"].nunique() < 2 or len(grp) < 10:
            continue
        try:
            from sklearn.metrics import roc_auc_score as _roc
            auc = _roc(grp["flipped"].astype(int), grp["htir"])
            subtask_auroc[str(task)] = round(float(auc), 4)
        except Exception:
            pass
    results["per_subtask_auroc"] = subtask_auroc
    if subtask_auroc:
        logger.info(f"Per-subtask AUROCs: {subtask_auroc}")

    # Tail analysis: is anti-prediction driven by high-HTIR non-flipped items?
    if len(noflip_htir) > 0:
        p75 = np.percentile(np.concatenate([flip_htir, noflip_htir]), 75)
        results["tail_analysis"] = {
            "p75_threshold": float(p75),
            "frac_flip_above_p75":    float((flip_htir > p75).mean()),
            "frac_noflip_above_p75":  float((noflip_htir > p75).mean()),
        }
        logger.info(f"Tail (>p75={p75:.4f}): flip={results['tail_analysis']['frac_flip_above_p75']:.3f}, "
                    f"no_flip={results['tail_analysis']['frac_noflip_above_p75']:.3f}")

    # Figure: per-subtask AUROC bar chart
    if HAS_MPL and subtask_auroc and fig_dir is not None:
        fig, ax = plt.subplots(figsize=(max(6, len(subtask_auroc) * 0.8), 4))
        tasks = list(subtask_auroc.keys())
        aucs  = [subtask_auroc[t] for t in tasks]
        colors = ["#F44336" if a < 0.5 else "#4CAF50" for a in aucs]
        ax.bar([t.replace("bbh_", "").replace("_", "\n")[:20] for t in tasks], aucs, color=colors)
        ax.axhline(0.5, color="black", linewidth=1, linestyle="--", label="random")
        ax.set_ylabel("AUROC")
        ax.set_title("HTIR AUROC per subtask (predicting hint-flip)\nRed < 0.5 = anti-predictive")
        ax.legend()
        fig_dir.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(fig_dir / "htir_subtask_auroc.png", dpi=150)
        plt.close(fig)
        logger.info(f"Figure → {fig_dir}/htir_subtask_auroc.png")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Task D — Headline BCa CIs + Holm-Bonferroni
# ─────────────────────────────────────────────────────────────────────────────

def task_d_headline_cis(
    pano_df: pd.DataFrame,
    aoc_df: Optional[pd.DataFrame],
    grm_results: Dict[str, Any],
    n_boot: int = 5000,
) -> Dict[str, Any]:
    logger.info("\n=== Task D: Headline BCa CIs + Holm-Bonferroni ===")
    if aoc_df is None:
        logger.warning("No AOC data — skipping headline CIs")
        return {}

    # Merge PANO with AOC — pano uses base_item_id; aoc uses item_id
    pano_b_trunc = pano_df[pano_df["regime_label"] == "B_trunc"].copy()
    aoc_cols = [c for c in aoc_df.columns if c.startswith("aoc_")]
    aoc_merge = aoc_df[["item_id"] + aoc_cols].copy()
    merged = pano_b_trunc.merge(
        aoc_merge, left_on="base_item_id", right_on="item_id", how="inner"
    )

    # Family of hypotheses
    hypothesis_family = []

    def add_h(name: str, x: np.ndarray, y: np.ndarray) -> Dict:
        res = spearman_with_ci(x, y, n_boot=n_boot)
        res["hypothesis"] = name
        hypothesis_family.append(res)
        logger.info(
            f"  {name}: ρ={res['rho']:.3f}  "
            f"BCa CI [{res['ci_lo']:.3f}, {res['ci_hi']:.3f}]  "
            f"p={res['p']:.5f}  n={res['n']}"
        )
        return res

    results: Dict[str, Any] = {"hypotheses": {}}

    # H1a: GRACE-T × aoc_truncate_50
    if "pano_div" in merged.columns and "aoc_truncate_50" in merged.columns:
        valid = merged[["pano_div", "aoc_truncate_50"]].dropna()
        h = add_h("H1a: rho(GRACE-T, aoc_truncate_50)", valid["pano_div"].values, valid["aoc_truncate_50"].values)
        results["hypotheses"]["H1a"] = h

    # H1b: GRACE-T × aoc_composite
    if "pano_div" in merged.columns and "aoc_composite" in merged.columns:
        valid = merged[["pano_div", "aoc_composite"]].dropna()
        h = add_h("H1b: rho(GRACE-T, aoc_composite)", valid["pano_div"].values, valid["aoc_composite"].values)
        results["hypotheses"]["H1b"] = h

    # H2a: GRACE-T × aoc_mistake (null hypothesis)
    pano_b_mistake = pano_df[pano_df["regime_label"] == "B_mistake"].copy()
    merged_m = pano_b_mistake.merge(
        aoc_merge, left_on="base_item_id", right_on="item_id", how="inner"
    )
    if "pano_div" in merged_m.columns and "aoc_mistake" in merged_m.columns:
        valid = merged_m[["pano_div", "aoc_mistake"]].dropna()
        h = add_h("H2a: rho(GRACE-T, aoc_mistake) [null]", valid["pano_div"].values, valid["aoc_mistake"].values)
        results["hypotheses"]["H2a"] = h

    # H3: GRM × aoc_mistake [only if GRM survived]
    if grm_results.get("grm_survives"):
        grm_corr = grm_results.get("correlations", {}).get("aoc_mistake", {})
        if grm_corr:
            results["hypotheses"]["H3"] = {**grm_corr, "hypothesis": "H3: rho(GRM, aoc_mistake) [conditional]"}
            hypothesis_family.append(results["hypotheses"]["H3"])
            logger.info(
                f"  H3: ρ(GRM, aoc_mistake)={grm_corr.get('rho', 0):.3f}  "
                f"BCa CI [{grm_corr.get('ci_lo', 0):.3f}, {grm_corr.get('ci_hi', 0):.3f}]"
            )

    # Holm-Bonferroni across the family
    if hypothesis_family:
        p_values = [h.get("p", 1.0) for h in hypothesis_family]
        hb = holm_bonferroni(p_values, alpha=0.05)
        results["holm_bonferroni"] = {
            "p_values_raw": p_values,
            "corrected_p": hb["corrected_p"],
            "rejected": hb["rejected"],
            "hypothesis_names": [h.get("hypothesis", f"H{i}") for i, h in enumerate(hypothesis_family)],
        }
        logger.info("\nHolm-Bonferroni corrected p-values:")
        for i, h in enumerate(hypothesis_family):
            logger.info(
                f"  {h.get('hypothesis', f'H{i}')}: "
                f"raw p={p_values[i]:.5f} → corrected={hb['corrected_p'][i]:.5f} "
                f"({'REJECT' if hb['rejected'][i] else 'retain'})"
            )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--grm-threshold", type=float, default=0.20,
                        help="Minimum ρ(GRM, aoc_mistake) for GRM to enter paper")
    parser.add_argument("--n-boot", type=int, default=5000)
    parser.add_argument("--skip-task-b", action="store_true",
                        help="Skip ED full characterization (saves ~5 min)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir    = Path(args.graph_dir or cfg.paths.graphs)
    agd_dir      = Path(cfg.paths.agd)
    analysis_dir = Path(cfg.paths.analysis)
    fig_dir      = analysis_dir / "figures"
    pairs_dir    = Path(cfg.paths.pairs)

    # ── Load data ─────────────────────────────────────────────────────────────
    aoc_df = None
    aoc_path = Path(cfg.paths.behavioral) / "aoc_lanham.parquet"
    if aoc_path.exists():
        raw_aoc = pd.read_parquet(aoc_path)
        # Normalize so both item_id and base_item_id are present
        aoc_df = raw_aoc.copy()
        if "base_item_id" not in aoc_df.columns:
            aoc_df["base_item_id"] = aoc_df["item_id"]
        logger.info(f"AOC loaded: {len(aoc_df)} items")

    pano_df = None
    for fname in ("pano_pairs_with_editdist.parquet", "pano_pairs.parquet"):
        p = agd_dir / fname
        if p.exists():
            pano_df = pd.read_parquet(p)
            logger.info(f"PANO pairs loaded from {fname}: {len(pano_df)} rows")
            break

    htir_df = None
    htir_path = agd_dir / "htir_results.parquet"
    if htir_path.exists():
        htir_df = pd.read_parquet(htir_path)
        logger.info(f"HTIR results loaded: {len(htir_df)} rows")

    # Load mistake pairs (with prompts for ED characterization)
    mistake_pairs = []
    mistake_file = pairs_dir / "regime_B_addmistake.jsonl"
    if mistake_file.exists():
        with jsonlines.open(mistake_file) as reader:
            mistake_pairs = list(reader)
    logger.info(f"Loaded {len(mistake_pairs)} B_mistake pairs")

    # ── Run tasks ─────────────────────────────────────────────────────────────
    all_results: Dict[str, Any] = {}

    # Task A: GRM
    grm_results = task_a_grm(
        mistake_pairs, graph_dir, aoc_df, args.grm_threshold, agd_dir
    )
    all_results["task_a_grm"] = grm_results

    # Task B: ED failure characterization
    if not args.skip_task_b:
        ed_results = task_b_ed_failure(
            mistake_pairs, graph_dir,
            f_values=[0.05, 0.10, 0.20, 0.30],
            fig_dir=fig_dir,
        )
        all_results["task_b_ed_failure"] = ed_results

    # Task C: HTIR anti-prediction
    if htir_df is not None:
        htir_results = task_c_htir_characterization(
            htir_df, n_boot=args.n_boot, fig_dir=fig_dir
        )
        all_results["task_c_htir"] = htir_results

    # Task D: Headline CIs
    if pano_df is not None:
        headline_results = task_d_headline_cis(
            pano_df, aoc_df, grm_results, n_boot=args.n_boot
        )
        all_results["task_d_headline"] = headline_results

    # ── Print decision summary ────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("=== DAY 4 DECISION SUMMARY ===")
    logger.info("=" * 60)
    logger.info(f"GRM decision: {grm_results.get('decision', 'N/A')}")
    logger.info(f"  ρ(GRM, aoc_mistake) = {grm_results.get('correlations', {}).get('aoc_mistake', {}).get('rho', 'N/A')}")
    logger.info(f"  Threshold: {args.grm_threshold}")
    if grm_results.get("grm_survives"):
        logger.info("  → GRM ENTERS the paper as 4th metric (with post-hoc caveat)")
    else:
        logger.info("  → GRM DROPPED; paper reports ED locality-failure only")

    h1a = all_results.get("task_d_headline", {}).get("hypotheses", {}).get("H1a", {})
    logger.info(f"\nH1a ρ(GRACE-T, aoc_truncate_50) = {h1a.get('rho', 'N/A'):.3f}  "
                f"BCa [{h1a.get('ci_lo', 0):.3f}, {h1a.get('ci_hi', 0):.3f}]")

    h2a = all_results.get("task_d_headline", {}).get("hypotheses", {}).get("H2a", {})
    logger.info(f"H2a ρ(GRACE-T, aoc_mistake)     = {h2a.get('rho', 'N/A'):.3f}  "
                f"BCa [{h2a.get('ci_lo', 0):.3f}, {h2a.get('ci_hi', 0):.3f}]")

    # ── Write results ─────────────────────────────────────────────────────────
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "results_day4.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nAll results → {out_path}")


if __name__ == "__main__":
    main()
