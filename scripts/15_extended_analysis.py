"""
scripts/15_extended_analysis.py
Extended pre-paper analyses building on PANO results.

Five analyses:
  1. Per-dataset breakdown of H1 Spearman rho (BBH / MMLU / GSM8K)
  2. H2 stratified by dataset and flip-confidence
  3. Feature-level qualitative analysis — which concepts appear/disappear
     in high-vs-low PANO_div B_mistake pairs
  4. Regime A vs B sanity check: Mann-Whitney U + Cliff's delta
  5. Null distribution: PANO_div distributions per regime (for a figure)

Reads:
    artifacts/agd/pano_pairs.parquet
    artifacts/behavioral/aoc_lanham.parquet
    artifacts/behavioral/turpin_flips.parquet
    artifacts/graphs/*.json        (for feature-level analysis)

Writes:
    analysis/extended_results.json
    analysis/regime_distributions.csv    (for plotting)
    analysis/feature_analysis.json       (qualitative examples)

Usage:
    python scripts/15_extended_analysis.py --config config.yaml
    python scripts/15_extended_analysis.py --config config.yaml --use-full-set
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci, auroc_with_ci, cliffs_delta
from src.pano import strip_position, graph_to_pano_node_set
from src.graph_utils import load_graph


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def infer_dataset(item_id: str) -> str:
    """Infer dataset label from item_id prefix convention."""
    s = str(item_id).lower()
    if s.startswith("bbh"):
        return "BBH"
    if s.startswith("mmlu"):
        return "MMLU"
    if s.startswith("gsm"):
        return "GSM8K"
    if s.startswith("turpin") or s.startswith("tur"):
        return "Turpin"
    # Fallback: try numeric ranges or unknown
    return "Unknown"


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
# Analysis 1: Per-dataset H1 breakdown
# ─────────────────────────────────────────────────────────────────────────────

def analysis_1_per_dataset_h1(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """Spearman rho(PANO_div, AOC) broken down by dataset."""
    logger.info("\n── Analysis 1: Per-dataset H1 breakdown ──")

    b_mask = pano_df["regime_label"].isin(["B_mistake", "B_trunc"])
    b_pano = pano_df[b_mask].dropna(subset=["pano_div"]).copy()
    b_pano["dataset"] = b_pano["base_item_id"].apply(infer_dataset)

    # Also tag by dataset via item_id when base_item_id is unavailable
    b_pano.loc[b_pano["dataset"] == "Unknown", "dataset"] = (
        b_pano.loc[b_pano["dataset"] == "Unknown", "item_id"].apply(infer_dataset)
    )

    results = {}

    # Overall (reference)
    overall = b_pano.groupby("base_item_id")["pano_div"].mean().reset_index()
    overall_merged = overall.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")
    if len(overall_merged) >= 20:
        r = spearman_with_ci(
            overall_merged["pano_div"].values,
            overall_merged["aoc_composite"].values,
            n_boot=n_boot, seed=seed,
        )
        results["overall"] = {**r, "n_items": len(overall_merged)}
        logger.info(
            f"  Overall: rho={r['rho']:.3f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], n={len(overall_merged)}"
        )

    # Per dataset
    for ds in sorted(b_pano["dataset"].unique()):
        ds_rows = b_pano[b_pano["dataset"] == ds]
        if len(ds_rows) < 10:
            logger.info(f"  {ds}: too few rows ({len(ds_rows)}) — skipped")
            continue

        per_item = ds_rows.groupby("base_item_id")["pano_div"].mean().reset_index()
        merged = per_item.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")

        if len(merged) < 10:
            logger.info(f"  {ds}: too few matched items ({len(merged)}) — skipped")
            results[ds] = {"skipped": True, "reason": f"n={len(merged)} < 10"}
            continue

        r = spearman_with_ci(
            merged["pano_div"].values,
            merged["aoc_composite"].values,
            n_boot=n_boot, seed=seed,
        )
        results[ds] = {**r, "n_items": len(merged)}
        logger.info(
            f"  {ds}: rho={r['rho']:.3f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], n={len(merged)}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 2: H2 stratified by dataset
# ─────────────────────────────────────────────────────────────────────────────

def analysis_2_h2_stratified(
    pano_df: pd.DataFrame,
    flip_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict[str, Any]:
    """AUROC(delta_AGD, hint_flip) broken down by dataset."""
    logger.info("\n── Analysis 2: H2 stratified by dataset ──")

    if "delta_agd" not in pano_df.columns:
        logger.warning("delta_agd column missing — skipping Analysis 2")
        return {}

    c_pano = pano_df[pano_df["regime_label"] == "C"].copy()
    if "unfaithful_flip" in c_pano.columns:
        c_pano = c_pano.drop(columns=["unfaithful_flip"])

    c_pano["dataset"] = c_pano["item_id"].apply(infer_dataset)
    merged = c_pano.merge(flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner")
    merged = merged.dropna(subset=["delta_agd"])

    results = {}

    # Overall reference
    n_flips = merged["unfaithful_flip"].sum()
    if len(merged) >= 30 and n_flips >= 10:
        r = auroc_with_ci(
            merged["delta_agd"].values,
            merged["unfaithful_flip"].astype(int).values,
            n_boot=n_boot, seed=seed,
        )
        results["overall"] = {**r}
        logger.info(
            f"  Overall: AUROC={r['auc']:.3f}, CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], "
            f"n_pos={r['n_pos']}, n_neg={r['n_neg']}"
        )

    # Per dataset
    for ds in sorted(merged["dataset"].unique()):
        ds_rows = merged[merged["dataset"] == ds]
        n_pos = ds_rows["unfaithful_flip"].sum()
        if len(ds_rows) < 20 or n_pos < 5:
            logger.info(f"  {ds}: too few samples or flips — skipped")
            results[ds] = {"skipped": True, "reason": f"n={len(ds_rows)}, n_pos={n_pos}"}
            continue

        r = auroc_with_ci(
            ds_rows["delta_agd"].values,
            ds_rows["unfaithful_flip"].astype(int).values,
            n_boot=n_boot, seed=seed,
        )
        results[ds] = {**r}
        logger.info(
            f"  {ds}: AUROC={r['auc']:.3f}, CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}], "
            f"n_pos={r['n_pos']}, n_neg={r['n_neg']}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 3: Feature-level qualitative analysis
# ─────────────────────────────────────────────────────────────────────────────

def analysis_3_feature_level(
    pano_df: pd.DataFrame,
    graph_dir: Path,
    k: int = 64,
    n_examples: int = 3,
) -> Dict[str, Any]:
    """For top-N high-divergence and low-divergence B_mistake pairs,
    show which concepts are unique to each graph (gained/lost features)."""
    logger.info("\n── Analysis 3: Feature-level qualitative analysis ──")

    bm = pano_df[pano_df["regime_label"] == "B_mistake"].dropna(subset=["pano_div"]).copy()
    if bm.empty:
        logger.warning("No B_mistake pairs — skipping Analysis 3")
        return {}

    bm_sorted = bm.sort_values("pano_div", ascending=False)
    high_div = bm_sorted.head(n_examples)
    low_div  = bm_sorted.tail(n_examples)

    def describe_pair(row) -> Optional[Dict]:
        item_id = row["item_id"]
        c0 = row.get("condition0", "clean")
        c1 = row.get("condition1", "mistake")
        p0 = graph_dir / f"{item_id}_{c0}.json"
        p1 = graph_dir / f"{item_id}_{c1}.json"

        if not p0.exists() or not p1.exists():
            return None

        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
        except Exception as e:
            logger.warning(f"Could not load graphs for {item_id}: {e}")
            return None

        n0 = graph_to_pano_node_set(g0, k=k)
        n1 = graph_to_pano_node_set(g1, k=k)

        shared   = set(n0) & set(n1)
        only_g0  = set(n0) - set(n1)   # in clean, not in mistake
        only_g1  = set(n1) - set(n0)   # in mistake, not in clean (gained)

        # Top-5 by influence for each bucket
        def top5(concept_set, source_dict) -> List[Dict]:
            ranked = sorted(
                [(c, source_dict[c]) for c in concept_set if c in source_dict],
                key=lambda x: x[1], reverse=True,
            )[:5]
            return [{"concept": c, "influence": round(v, 4)} for c, v in ranked]

        return {
            "item_id": item_id,
            "pano_div": round(float(row["pano_div"]), 4),
            "n_shared": len(shared),
            "n_lost_from_clean": len(only_g0),
            "n_gained_in_mistake": len(only_g1),
            "top5_lost":   top5(only_g0, n0),
            "top5_gained": top5(only_g1, n1),
            "top5_shared": top5(shared,  n0),
        }

    high_examples = []
    for _, row in high_div.iterrows():
        desc = describe_pair(row)
        if desc:
            high_examples.append(desc)
            logger.info(
                f"  HIGH pano_div={desc['pano_div']:.3f} | item={desc['item_id']} | "
                f"shared={desc['n_shared']}, lost={desc['n_lost_from_clean']}, "
                f"gained={desc['n_gained_in_mistake']}"
            )

    low_examples = []
    for _, row in low_div.iterrows():
        desc = describe_pair(row)
        if desc:
            low_examples.append(desc)
            logger.info(
                f"  LOW  pano_div={desc['pano_div']:.3f} | item={desc['item_id']} | "
                f"shared={desc['n_shared']}, lost={desc['n_lost_from_clean']}, "
                f"gained={desc['n_gained_in_mistake']}"
            )

    return {
        "high_divergence_examples": high_examples,
        "low_divergence_examples":  low_examples,
        "k": k,
        "n_examples_requested": n_examples,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 4: Regime A vs B Mann-Whitney U + Cliff's delta
# ─────────────────────────────────────────────────────────────────────────────

def analysis_4_regime_ab_sanity(pano_df: pd.DataFrame) -> Dict[str, Any]:
    """Mann-Whitney U test comparing each regime against Regime A (paraphrase).

    B_trunc and C are tested one-sided 'greater' (expected higher divergence).
    B_mistake is tested two-sided then reported with direction, because it is
    empirically lower than A — a directional one-sided 'less' test is also stored
    so the anomaly is citable with a proper p-value.
    """
    logger.info("\n── Analysis 4: Regime A vs B sanity check ──")

    a_vals  = pano_df[pano_df["regime_label"] == "A"]["pano_div"].dropna().values
    bm_vals = pano_df[pano_df["regime_label"] == "B_mistake"]["pano_div"].dropna().values
    bt_vals = pano_df[pano_df["regime_label"] == "B_trunc"]["pano_div"].dropna().values
    c_vals  = pano_df[pano_df["regime_label"] == "C"]["pano_div"].dropna().values

    results = {}

    def effect_label(abs_cd: float) -> str:
        if abs_cd < 0.11:
            return "negligible"
        elif abs_cd < 0.28:
            return "small"
        elif abs_cd < 0.43:
            return "medium"
        return "large"

    # B_trunc and C: test group > A (expected direction)
    for name, group in [("B_trunc", bt_vals), ("C", c_vals)]:
        if len(group) < 5:
            logger.info(f"  A vs {name}: too few samples — skipped")
            continue
        stat, p = scipy_stats.mannwhitneyu(group, a_vals, alternative="greater")
        cd = cliffs_delta(group.tolist(), a_vals.tolist())
        results[f"A_vs_{name}"] = {
            "mean_A": float(a_vals.mean()), "mean_other": float(group.mean()),
            "std_A": float(a_vals.std()),   "std_other": float(group.std()),
            "n_A": len(a_vals),             "n_other": len(group),
            "mann_whitney_U": float(stat),
            "p_value": float(p),
            "alternative": "greater",
            "cliffs_delta": float(cd),
            "effect_size": effect_label(abs(cd)),
            "significant": bool(p < 0.05),
        }
        logger.info(
            f"  A (mean={a_vals.mean():.3f}) vs {name} (mean={group.mean():.3f}): "
            f"U={stat:.0f}, p={p:.4f} [greater], Cliff's δ={cd:+.3f} ({effect_label(abs(cd))})"
        )

    # B_mistake: test two-sided first, then directional 'less' to capture the anomaly
    if len(bm_vals) >= 5:
        stat_2s, p_2s = scipy_stats.mannwhitneyu(bm_vals, a_vals, alternative="two-sided")
        stat_ls, p_ls = scipy_stats.mannwhitneyu(bm_vals, a_vals, alternative="less")
        cd = cliffs_delta(bm_vals.tolist(), a_vals.tolist())
        results["A_vs_B_mistake"] = {
            "mean_A": float(a_vals.mean()),   "mean_other": float(bm_vals.mean()),
            "std_A": float(a_vals.std()),     "std_other": float(bm_vals.std()),
            "n_A": len(a_vals),               "n_other": len(bm_vals),
            "mann_whitney_U_twosided": float(stat_2s),
            "p_value_twosided": float(p_2s),
            "mann_whitney_U_less": float(stat_ls),
            "p_value_less": float(p_ls),     # use this for the anomaly claim
            "cliffs_delta": float(cd),
            "effect_size": effect_label(abs(cd)),
            "direction": "B_mistake < A (anomaly)",
            "significant_less": bool(p_ls < 0.05),
        }
        logger.info(
            f"  A (mean={a_vals.mean():.3f}) vs B_mistake (mean={bm_vals.mean():.3f}): "
            f"p_twosided={p_2s:.4f}, p_less={p_ls:.4f}, "
            f"Cliff's δ={cd:+.3f} ({effect_label(abs(cd))}) ← ANOMALY: mistake < paraphrase"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Analysis 5: Null distribution — per-regime PANO_div stats for figures
# ─────────────────────────────────────────────────────────────────────────────

def analysis_5_null_distribution(pano_df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive stats + percentiles of PANO_div per regime.
    Returns a DataFrame suitable for box-plot / violin-plot generation.
    """
    logger.info("\n── Analysis 5: PANO_div distributions per regime ──")

    regime_order = ["A", "B_mistake", "B_trunc", "C"]
    regime_labels = {
        "A":         "Paraphrase (A)",
        "B_mistake": "Mistake (B)",
        "B_trunc":   "Truncation (B)",
        "C":         "Hint (C)",
    }

    rows = []
    for regime in regime_order:
        vals = pano_df[pano_df["regime_label"] == regime]["pano_div"].dropna().values
        if len(vals) == 0:
            continue

        row = {
            "regime":  regime,
            "label":   regime_labels.get(regime, regime),
            "n":       len(vals),
            "mean":    float(vals.mean()),
            "std":     float(vals.std()),
            "median":  float(np.median(vals)),
            "q1":      float(np.percentile(vals, 25)),
            "q3":      float(np.percentile(vals, 75)),
            "min":     float(vals.min()),
            "max":     float(vals.max()),
            "p5":      float(np.percentile(vals, 5)),
            "p95":     float(np.percentile(vals, 95)),
        }
        rows.append(row)
        logger.info(
            f"  {regime:10s}: mean={row['mean']:.3f} ± {row['std']:.3f}, "
            f"median={row['median']:.3f}, n={row['n']}"
        )

    df = pd.DataFrame(rows)
    logger.info("\n" + df.to_string(index=False))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-full-set", action="store_true",
                        help="Use all items (train+test) instead of test half only")
    parser.add_argument("--n-examples", type=int, default=3,
                        help="Number of qualitative feature examples per category")
    args = parser.parse_args()

    cfg  = load_config(args.config)
    seed = cfg.seed
    n_boot = cfg.stats.n_bootstrap

    # ── Load PANO pairs ───────────────────────────────────────────────────────
    pano_path = Path(cfg.paths.agd) / "pano_pairs.parquet"
    if not pano_path.exists():
        logger.error("pano_pairs.parquet not found. Run script 13 first.")
        sys.exit(1)
    pano_df = pd.read_parquet(pano_path)
    logger.info(f"Loaded pano_pairs: {len(pano_df)} rows")

    # Filter to test set unless --use-full-set
    if not args.use_full_set:
        test_ids = load_test_ids(cfg)
        if test_ids:
            test_mask = pano_df.apply(lambda r: get_base_id(r) in test_ids, axis=1)
            pano_df = pano_df[test_mask].copy()
            logger.info(f"After test-set filter: {len(pano_df)} rows")
    else:
        logger.info("Using full dataset (train+test).")

    behavior_dir = Path(cfg.paths.behavioral)
    graph_dir    = Path(cfg.paths.graphs)
    analysis_dir = Path(cfg.paths.analysis)
    analysis_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[str, Any] = {}

    # ── Analysis 1 ────────────────────────────────────────────────────────────
    aoc_path = behavior_dir / "aoc_lanham.parquet"
    if aoc_path.exists():
        aoc_df = pd.read_parquet(aoc_path)
        results["analysis_1_per_dataset_h1"] = analysis_1_per_dataset_h1(
            pano_df, aoc_df, n_boot=n_boot, seed=seed,
        )
    else:
        logger.warning("aoc_lanham.parquet not found — skipping Analysis 1")

    # ── Analysis 2 ────────────────────────────────────────────────────────────
    flip_path = behavior_dir / "turpin_flips.parquet"
    if flip_path.exists():
        flip_df = pd.read_parquet(flip_path)
        results["analysis_2_h2_stratified"] = analysis_2_h2_stratified(
            pano_df, flip_df, n_boot=n_boot, seed=seed,
        )
    else:
        logger.warning("turpin_flips.parquet not found — skipping Analysis 2")

    # ── Analysis 3 ────────────────────────────────────────────────────────────
    if graph_dir.exists():
        feat_results = analysis_3_feature_level(
            pano_df, graph_dir, k=cfg.agd.k, n_examples=args.n_examples,
        )
        results["analysis_3_feature_level"] = feat_results

        feat_path = analysis_dir / "feature_analysis.json"
        with open(feat_path, "w") as f:
            json.dump(feat_results, f, indent=2, default=str)
        logger.info(f"Feature analysis -> {feat_path}")
    else:
        logger.warning(f"Graph directory not found ({graph_dir}) — skipping Analysis 3")

    # ── Analysis 4 ────────────────────────────────────────────────────────────
    results["analysis_4_regime_ab_sanity"] = analysis_4_regime_ab_sanity(pano_df)

    # ── Analysis 5 ────────────────────────────────────────────────────────────
    dist_df = analysis_5_null_distribution(pano_df)
    results["analysis_5_distributions"] = dist_df.to_dict(orient="records")

    dist_csv_path = analysis_dir / "regime_distributions.csv"
    dist_df.to_csv(dist_csv_path, index=False)
    logger.info(f"Regime distributions -> {dist_csv_path}")

    # ── Save consolidated results ─────────────────────────────────────────────
    out_path = analysis_dir / "extended_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nAll extended results -> {out_path}")

    # ── Print summary table ───────────────────────────────────────────────────
    logger.info("\n" + "="*65)
    logger.info("EXTENDED ANALYSIS SUMMARY")
    logger.info("="*65)

    if "analysis_1_per_dataset_h1" in results:
        logger.info("\nAnalysis 1 — H1 per dataset (rho):")
        for ds, r in results["analysis_1_per_dataset_h1"].items():
            if isinstance(r, dict) and "rho" in r:
                sig = "✓" if r["p"] < 0.05 else "✗"
                logger.info(
                    f"  {ds:8s}: rho={r['rho']:+.3f}, p={r['p']:.4f} {sig}, n={r.get('n_items', r.get('n','?'))}"
                )

    if "analysis_2_h2_stratified" in results:
        logger.info("\nAnalysis 2 — H2 per dataset (AUROC):")
        for ds, r in results["analysis_2_h2_stratified"].items():
            if isinstance(r, dict) and "auc" in r:
                flag = "✓" if r["auc"] >= 0.65 else "✗"
                logger.info(
                    f"  {ds:8s}: AUROC={r['auc']:.3f}, CI=[{r['ci_lo']:.3f},{r['ci_hi']:.3f}] {flag}"
                )

    if "analysis_4_regime_ab_sanity" in results:
        logger.info("\nAnalysis 4 — Regime A vs Others (Mann-Whitney, Cliff's δ):")
        for comparison, r in results["analysis_4_regime_ab_sanity"].items():
            cd = r["cliffs_delta"]
            effect = r["effect_size"]
            if comparison == "A_vs_B_mistake":
                sig = "p<0.05 ✓ (less)" if r.get("significant_less") else "n.s."
                logger.info(
                    f"  {comparison}: δ={cd:+.3f} ({effect}), {sig} ← ANOMALY: mistake < paraphrase"
                )
            else:
                sig = "p<0.05 ✓" if r.get("significant") else "n.s. ✗"
                logger.info(
                    f"  {comparison}: δ={cd:+.3f} ({effect}), {sig}"
                )

    logger.info("="*65)


if __name__ == "__main__":
    main()
