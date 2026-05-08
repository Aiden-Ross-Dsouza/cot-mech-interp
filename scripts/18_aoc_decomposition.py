"""
scripts/18_aoc_decomposition.py
Three exploratory analyses extending H1 using only existing artifacts.

All run on the FULL 143-item set (Regime B with AOC labels). PANO has no
learned parameters, so the held-out split is a consistency check, not a
requirement; the full set provides the statistical power that n=50 lacks.

Idea 1 — AOC component breakdown:
    Spearman(PANO_div, AOC_component) for each of the 5 AOC sub-scores
    (early, truncate_25, truncate_50, truncate_75, mistake). Reveals which
    type of unfaithfulness PANO_div tracks best.

Idea 2 — Per-perturbation-type H1:
    Spearman(PANO_div, AOC) computed separately for B_mistake and B_trunc
    item subsets. Tests whether H1 generalises across perturbation types.

Idea 4 — Shared-count ablation:
    Spearman(n_shared_concepts, AOC) — does the raw shared-concept count
    (no influence weighting) match PANO_div's predictive power? If yes,
    the influence weighting is decorative. If no, it's essential.

Reads:
    artifacts/agd/pano_pairs_with_editdist.parquet
    artifacts/behavioral/aoc_lanham.parquet

Writes:
    analysis/results_decomposition.json

Usage:
    python scripts/18_aoc_decomposition.py --config config.yaml
    python scripts/18_aoc_decomposition.py --config config.yaml --test-only
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci


def get_base_id(row) -> str:
    b = row.get("base_item_id")
    if b and not (isinstance(b, float) and np.isnan(b)):
        return b
    return row["item_id"]


def load_test_ids(cfg) -> Optional[set]:
    p = Path(cfg.paths.data) / "test_ids.txt"
    if not p.exists():
        return None
    with open(p) as f:
        return set(line.strip() for line in f if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Idea 1: AOC component breakdown
# ─────────────────────────────────────────────────────────────────────────────

def run_aoc_decomposition(b_pano: pd.DataFrame, aoc_df: pd.DataFrame,
                          n_boot: int, seed: int) -> Dict:
    """Per-component Spearman ρ(PANO_div, AOC_subscore)."""
    components = [
        "aoc_early",
        "aoc_truncate_25",
        "aoc_truncate_50",
        "aoc_truncate_75",
        "aoc_mistake",
        "aoc_composite",  # for reference
    ]
    available = [c for c in components if c in aoc_df.columns]

    # Per-item PANO_div mean across all Regime B perturbations
    per_item = (b_pano.groupby("base_item_id")["pano_div"]
                .mean().reset_index()
                .rename(columns={"pano_div": "pano_div_mean"}))
    merged = per_item.merge(
        aoc_df[["item_id"] + available],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["pano_div_mean"])

    logger.info(f"Idea 1: {len(merged)} items for AOC decomposition")

    results = {}
    for comp in available:
        subset = merged.dropna(subset=[comp])
        if len(subset) < 20:
            results[comp] = {"status": "skipped", "n": len(subset)}
            continue
        r = spearman_with_ci(
            subset["pano_div_mean"].values,
            subset[comp].values,
            n_boot=n_boot, seed=seed,
        )
        results[comp] = r
        logger.info(
            f"  {comp:20s}: ρ={r['rho']:+.4f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], n={r['n']}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Idea 2: Per-perturbation-type H1
# ─────────────────────────────────────────────────────────────────────────────

def run_per_perturbation(b_pano: pd.DataFrame, aoc_df: pd.DataFrame,
                         n_boot: int, seed: int) -> Dict:
    """Spearman ρ(PANO_div, AOC_composite) split by B_mistake vs B_trunc."""
    results = {}
    for regime in ["B_mistake", "B_trunc"]:
        sub = b_pano[b_pano["regime_label"] == regime].copy()
        per_item = (sub.groupby("base_item_id")["pano_div"]
                    .mean().reset_index()
                    .rename(columns={"pano_div": "pano_div_mean"}))
        merged = per_item.merge(
            aoc_df[["item_id", "aoc_composite"]],
            left_on="base_item_id", right_on="item_id", how="inner"
        ).dropna(subset=["pano_div_mean", "aoc_composite"])

        if len(merged) < 20:
            results[regime] = {"status": "skipped", "n": len(merged)}
            continue
        r = spearman_with_ci(
            merged["pano_div_mean"].values,
            merged["aoc_composite"].values,
            n_boot=n_boot, seed=seed,
        )
        results[regime] = r
        logger.info(
            f"  {regime:12s}: ρ={r['rho']:+.4f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], n={r['n']}"
        )
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Idea 4: Shared-count ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_shared_count_ablation(b_pano: pd.DataFrame, aoc_df: pd.DataFrame,
                              n_boot: int, seed: int) -> Dict:
    """Compare ρ(PANO_div, AOC) vs ρ(1 - n_shared/k, AOC).

    n_shared_concepts is shared count (out of k=64). Convert to a divergence
    by 1 - n_shared/k so higher = more divergent (matching PANO_div direction).
    """
    sub = b_pano.dropna(subset=["pano_div", "n_shared_concepts",
                                 "n_concepts_0", "n_concepts_1"]).copy()
    # Approximate k as the max of n_concepts (typically 64)
    k_approx = max(sub["n_concepts_0"].max(), sub["n_concepts_1"].max())
    sub["unweighted_div"] = 1.0 - sub["n_shared_concepts"] / k_approx

    per_item = (sub.groupby("base_item_id")
                .agg(pano_div_mean=("pano_div", "mean"),
                     unweighted_div_mean=("unweighted_div", "mean"))
                .reset_index())
    merged = per_item.merge(
        aoc_df[["item_id", "aoc_composite"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["pano_div_mean", "unweighted_div_mean", "aoc_composite"])

    logger.info(f"Idea 4: {len(merged)} items for shared-count ablation")

    rho_pano = spearman_with_ci(
        merged["pano_div_mean"].values,
        merged["aoc_composite"].values,
        n_boot=n_boot, seed=seed,
    )
    rho_unweighted = spearman_with_ci(
        merged["unweighted_div_mean"].values,
        merged["aoc_composite"].values,
        n_boot=n_boot, seed=seed,
    )

    logger.info(
        f"  PANO_div (weighted)   : ρ={rho_pano['rho']:+.4f}, "
        f"CI=[{rho_pano['ci_lo']:+.4f}, {rho_pano['ci_hi']:+.4f}]"
    )
    logger.info(
        f"  Unweighted shared-div : ρ={rho_unweighted['rho']:+.4f}, "
        f"CI=[{rho_unweighted['ci_lo']:+.4f}, {rho_unweighted['ci_hi']:+.4f}]"
    )
    diff = rho_pano["rho"] - rho_unweighted["rho"]
    logger.info(f"  Δρ (weighted − unweighted): {diff:+.4f}")
    return {
        "weighted_pano": rho_pano,
        "unweighted_shared_div": rho_unweighted,
        "delta_rho": float(diff),
        "k_approx": float(k_approx),
        "interpretation": (
            "Influence weighting adds signal" if abs(diff) > 0.05
            else "Influence weighting may be decorative; raw shared-count nearly as predictive"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--test-only", action="store_true",
                        help="Restrict to held-out test split (default: full 143-item set)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_boot = cfg.stats.n_bootstrap

    pano_df = pd.read_parquet(
        Path(cfg.paths.agd) / "pano_pairs_with_editdist.parquet"
    )
    aoc_df = pd.read_parquet(
        Path(cfg.paths.behavioral) / "aoc_lanham.parquet"
    )

    if args.test_only:
        test_ids = load_test_ids(cfg)
        pano_df = pano_df[pano_df.apply(lambda r: get_base_id(r) in test_ids, axis=1)]
        logger.info(f"Test-only filter: {len(pano_df)} rows")
    else:
        logger.info(f"Full set (n={pano_df['base_item_id'].nunique()} unique items)")

    b_pano = pano_df[pano_df["regime_label"].isin(["B_mistake", "B_trunc"])].copy()

    results = {"split": "test" if args.test_only else "full"}

    logger.info("\n" + "="*60)
    logger.info("Idea 1: AOC component breakdown")
    logger.info("="*60)
    results["idea1_aoc_components"] = run_aoc_decomposition(
        b_pano, aoc_df, n_boot=n_boot, seed=cfg.seed
    )

    logger.info("\n" + "="*60)
    logger.info("Idea 2: Per-perturbation-type H1")
    logger.info("="*60)
    results["idea2_per_perturbation"] = run_per_perturbation(
        b_pano, aoc_df, n_boot=n_boot, seed=cfg.seed
    )

    logger.info("\n" + "="*60)
    logger.info("Idea 4: Shared-count ablation (is influence weighting essential?)")
    logger.info("="*60)
    results["idea4_shared_count_ablation"] = run_shared_count_ablation(
        b_pano, aoc_df, n_boot=n_boot, seed=cfg.seed
    )

    out = Path(cfg.paths.analysis) / (
        "results_decomposition_test.json" if args.test_only
        else "results_decomposition.json"
    )
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults -> {out}")


if __name__ == "__main__":
    main()
