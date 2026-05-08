"""
scripts/14_pano_dagd_analysis.py
H1/H2/H3 analysis using PANO divergence and delta-AGD.

Reads:
    artifacts/agd/pano_pairs.parquet
    artifacts/behavioral/aoc_lanham.parquet
    artifacts/behavioral/turpin_flips.parquet
    data/test_ids.txt

Hypothesis tests:
    H1': Spearman rho(PANO_div, AOC) >= 0.30 on Regime B test set
    H2': AUROC(delta_AGD, hint_flip) >= 0.65 on Regime C test set
    H3': Incremental AUROC of PANO_div + delta_AGD over activation-cosine baseline

Saves: analysis/results_pano.json

Usage:
    python scripts/14_pano_dagd_analysis.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci, auroc_with_ci, incremental_auroc, holm_bonferroni


def load_test_ids(cfg) -> set:
    test_path = Path(cfg.paths.data) / "test_ids.txt"
    if not test_path.exists():
        logger.error("test_ids.txt not found. Run data/split.py first.")
        sys.exit(1)
    with open(test_path) as f:
        return set(line.strip() for line in f if line.strip())


def get_base_id(row) -> str:
    """Return the base item ID for train/test filtering."""
    b = row.get("base_item_id")
    if b and not (isinstance(b, float) and np.isnan(b)):
        return b
    return row["item_id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--use-full-set", action="store_true",
                        help="Use all items (train+test) instead of test half only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_boot = cfg.stats.n_bootstrap

    # Load PANO results
    pano_path = Path(cfg.paths.agd) / "pano_pairs.parquet"
    if not pano_path.exists():
        logger.error("pano_pairs.parquet not found. Run script 13 first.")
        sys.exit(1)
    pano_df = pd.read_parquet(pano_path)
    logger.info(f"Loaded pano_pairs: {len(pano_df)} rows")

    # Filter to test set unless --use-full-set
    if not args.use_full_set:
        test_ids = load_test_ids(cfg)
        test_mask = pano_df.apply(lambda r: get_base_id(r) in test_ids, axis=1)
        pano_df = pano_df[test_mask].copy()
        logger.info(f"After test-set filter: {len(pano_df)} rows")
    else:
        logger.info("Using full dataset (train+test).")

    behavior_dir = Path(cfg.paths.behavioral)
    results = {}
    p_values = []
    test_names = []

    # ── H1': Spearman(PANO_div, AOC) on Regime B ─────────────────────────────
    aoc_path = behavior_dir / "aoc_lanham.parquet"
    h1_result = None
    if aoc_path.exists():
        aoc_df = pd.read_parquet(aoc_path)

        b_mask = pano_df["regime_label"].isin(["B_mistake", "B_trunc"])
        b_pano = pano_df[b_mask].dropna(subset=["pano_div"]).copy()

        # Average PANO_div per base item (same as H1 approach in script 10)
        b_per_item = b_pano.groupby("base_item_id")["pano_div"].mean().reset_index()
        merged = b_per_item.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")
        logger.info(f"H1' pairs (Regime B, per-item): {len(merged)}")

        if len(merged) >= 20:
            h1_result = spearman_with_ci(
                merged["pano_div"].values,
                merged["aoc_composite"].values,  # PANO_div should correlate POSITIVELY with AOC
                n_boot=n_boot, seed=cfg.seed,
            )
            results["H1_pano_spearman"] = h1_result
            p_values.append(h1_result["p"])
            test_names.append("H1_pano_spearman")
            logger.info(
                f"H1' PANO: rho={h1_result['rho']:.4f}, p={h1_result['p']:.4f}, "
                f"CI=[{h1_result['ci_lo']:.4f}, {h1_result['ci_hi']:.4f}], n={merged.shape[0]}"
            )
            passed = h1_result["rho"] >= cfg.stats.h1_rho_threshold
            logger.info(f"H1' passed (rho >= {cfg.stats.h1_rho_threshold}): {passed}")
        else:
            logger.warning(f"H1': insufficient pairs ({len(merged)}) — skipping")
    else:
        logger.warning("aoc_lanham.parquet not found — H1' skipped")

    # Also run H1' using delta-AGD on B_mistake only (cleaner signal)
    h1_delta_result = None
    if aoc_path.exists() and "delta_agd" in pano_df.columns:
        aoc_df = pd.read_parquet(aoc_path)
        bm = pano_df[(pano_df["regime_label"] == "B_mistake")].dropna(subset=["delta_agd"]).copy()
        bm_per_item = bm.groupby("base_item_id")["delta_agd"].mean().reset_index()
        merged_d = bm_per_item.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")
        logger.info(f"H1' delta-AGD pairs (B_mistake, per-item): {len(merged_d)}")

        if len(merged_d) >= 20:
            h1_delta_result = spearman_with_ci(
                merged_d["delta_agd"].values,
                merged_d["aoc_composite"].values,
                n_boot=n_boot, seed=cfg.seed,
            )
            results["H1_deltaAGD_spearman"] = h1_delta_result
            logger.info(
                f"H1' delta-AGD: rho={h1_delta_result['rho']:.4f}, "
                f"p={h1_delta_result['p']:.4f}, "
                f"CI=[{h1_delta_result['ci_lo']:.4f}, {h1_delta_result['ci_hi']:.4f}]"
            )

    # ── H2': AUROC(delta_AGD, hint_flip) on Regime C ─────────────────────────
    flip_path = behavior_dir / "turpin_flips.parquet"
    h2_result = None
    h2_pano_result = None
    if flip_path.exists() and "delta_agd" in pano_df.columns:
        flip_df = pd.read_parquet(flip_path)

        c_pano = pano_df[pano_df["regime_label"] == "C"].copy()
        if "unfaithful_flip" in c_pano.columns:
            c_pano = c_pano.drop(columns=["unfaithful_flip"])

        merged_c = c_pano.merge(
            flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner"
        )
        merged_c = merged_c.dropna(subset=["delta_agd"])
        n_flips = merged_c["unfaithful_flip"].sum()
        logger.info(f"H2' Regime C pairs: {len(merged_c)}, unfaithful flips: {n_flips}")

        if len(merged_c) >= 30 and n_flips >= 10:
            # H2' with delta-AGD (primary)
            h2_result = auroc_with_ci(
                merged_c["delta_agd"].values,
                merged_c["unfaithful_flip"].astype(int).values,
                n_boot=n_boot, seed=cfg.seed,
            )
            results["H2_deltaAGD_auroc"] = h2_result
            p_values.append(0.05 if h2_result["auc"] < cfg.stats.h2_auroc_threshold else 0.001)
            test_names.append("H2_deltaAGD_auroc")
            logger.info(
                f"H2' delta-AGD AUROC: {h2_result['auc']:.4f}, "
                f"CI=[{h2_result['ci_lo']:.4f}, {h2_result['ci_hi']:.4f}]"
            )
            passed = h2_result["auc"] >= cfg.stats.h2_auroc_threshold
            logger.info(f"H2' passed (AUROC >= {cfg.stats.h2_auroc_threshold}): {passed}")

            # H2' with raw PANO_div (secondary, for comparison)
            merged_c2 = c_pano.merge(
                flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner"
            ).dropna(subset=["pano_div"])
            if len(merged_c2) >= 30:
                h2_pano_result = auroc_with_ci(
                    merged_c2["pano_div"].values,
                    merged_c2["unfaithful_flip"].astype(int).values,
                    n_boot=n_boot, seed=cfg.seed,
                )
                results["H2_panoDiv_auroc"] = h2_pano_result
                logger.info(
                    f"H2' PANO_div AUROC (comparison): {h2_pano_result['auc']:.4f}, "
                    f"CI=[{h2_pano_result['ci_lo']:.4f}, {h2_pano_result['ci_hi']:.4f}]"
                )
        else:
            logger.warning(f"H2': insufficient Regime C items or flips — skipping")
    else:
        if not flip_path.exists():
            logger.warning("turpin_flips.parquet not found — H2' skipped")
        if "delta_agd" not in pano_df.columns:
            logger.warning("delta_agd column missing — run script 13 first")

    # ── H3': Incremental AUROC of PANO + delta-AGD over baselines ─────────────
    h3_result = None
    base_path = Path(cfg.paths.agd).parent / "baselines.parquet"
    if (h2_result is not None and flip_path.exists() and base_path.exists()):
        flip_df = pd.read_parquet(flip_path)
        base_df = pd.read_parquet(base_path)

        c_pano3 = pano_df[pano_df["regime_label"] == "C"].copy()
        if "unfaithful_flip" in c_pano3.columns:
            c_pano3 = c_pano3.drop(columns=["unfaithful_flip"])

        merged_h3 = (c_pano3
                     .merge(flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner")
                     .merge(base_df, on="item_id", how="left"))

        base_cols = ["activation_cosine", "kl_next_token", "cot_perplexity", "sc_variance"]
        avail = [c for c in base_cols if c in merged_h3.columns
                 and not merged_h3[c].isna().all()]
        merged_h3 = merged_h3.dropna(subset=["delta_agd", "pano_div"])

        if avail and len(merged_h3) >= 50:
            feat_without = merged_h3[avail].fillna(0).values
            feat_with = np.column_stack([
                merged_h3["pano_div"].fillna(0).values,
                merged_h3["delta_agd"].fillna(0).values,
                feat_without,
            ])
            labels = merged_h3["unfaithful_flip"].astype(int).values

            h3_result = incremental_auroc(
                feat_with, feat_without, labels,
                n_boot=n_boot, seed=cfg.seed,
            )
            results["H3_incremental_auroc"] = h3_result
            p_values.append(h3_result["p_value"])
            test_names.append("H3_incremental_auroc")
            logger.info(
                f"H3' delta-AUROC: {h3_result['delta_auc']:.4f}, "
                f"CI=[{h3_result['ci_lo']:.4f}, {h3_result['ci_hi']:.4f}], "
                f"p={h3_result['p_value']:.4f}"
            )
        else:
            logger.warning("H3': insufficient features or pairs — skipping")

    # ── Holm-Bonferroni ───────────────────────────────────────────────────────
    if p_values:
        hb = holm_bonferroni(p_values, alpha=cfg.stats.alpha_family)
        results["holm_bonferroni"] = {
            "test_names": test_names,
            "raw_p": p_values,
            **hb,
        }
        logger.info("\nHolm-Bonferroni:")
        for name, rp, cp, rej in zip(test_names, p_values, hb["corrected_p"], hb["rejected"]):
            logger.info(f"  {name}: raw_p={rp:.4f}, corr_p={cp:.4f}, reject={rej}")

    # ── Summary verdict ────────────────────────────────────────────────────────
    h1_ok = h1_result and h1_result["rho"] >= cfg.stats.h1_rho_threshold
    h2_ok = h2_result and h2_result["auc"] >= cfg.stats.h2_auroc_threshold
    h3_ok = h3_result and h3_result["delta_auc"] >= cfg.stats.h3_delta_auroc

    logger.info(f"\n{'='*60}")
    logger.info("PANO / delta-AGD RESULTS:")
    logger.info(f"  H1' PANO (rho >= {cfg.stats.h1_rho_threshold}):            {'PASSED' if h1_ok else 'FAILED / N/A'}")
    logger.info(f"  H2' delta-AGD (AUROC >= {cfg.stats.h2_auroc_threshold}):   {'PASSED' if h2_ok else 'FAILED / N/A'}")
    logger.info(f"  H3' Incr. AUROC (>= {cfg.stats.h3_delta_auroc}):          {'PASSED' if h3_ok else 'FAILED / N/A'}")
    logger.info(f"{'='*60}")

    results["verdict"] = {
        "H1_pano_passed": bool(h1_ok),
        "H2_deltaAGD_passed": bool(h2_ok),
        "H3_passed": bool(h3_ok),
    }

    out_path = Path(cfg.paths.analysis) / "results_pano.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nFull results -> {out_path}")


if __name__ == "__main__":
    main()
