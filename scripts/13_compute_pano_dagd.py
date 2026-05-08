"""
scripts/13_compute_pano_dagd.py
Compute PANO (Position-Agnostic Node Overlap) divergence and delta-AGD
for all graph pairs in artifacts/graphs/.

Reads:  data/pairs/*.jsonl  +  artifacts/graphs/*.json
Writes: artifacts/agd/pano_pairs.parquet

Usage:
    python scripts/13_compute_pano_dagd.py --config config.yaml
    python scripts/13_compute_pano_dagd.py --config config.yaml --k 64
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jsonlines
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.pano import batch_pano, compute_delta_agd

PAIR_FILES = {
    "A":         "regime_A_pairs.jsonl",
    "B_trunc":   "regime_B_truncate.jsonl",
    "B_mistake": "regime_B_addmistake.jsonl",
    "C":         "regime_C_hint.jsonl",
}


def load_all_pairs(cfg) -> pd.DataFrame:
    pairs_dir = Path(cfg.paths.pairs)
    all_rows = []
    for label, fname in PAIR_FILES.items():
        fpath = pairs_dir / fname
        if not fpath.exists():
            logger.warning(f"Pair file not found: {fpath}")
            continue
        with jsonlines.open(fpath) as reader:
            for row in reader:
                row["regime_label"] = label
                all_rows.append(row)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--k", type=int, default=None,
                        help="Top-k concepts for PANO (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    k = args.k if args.k is not None else cfg.agd.k

    logger.info(f"Computing PANO with k={k}")

    pairs_df = load_all_pairs(cfg)
    if pairs_df.empty:
        logger.error("No pairs found. Run scripts 01-03 first.")
        sys.exit(1)

    logger.info(f"Total pairs to process: {len(pairs_df)}")

    # ── Compute PANO for all pairs ────────────────────────────────────────────
    pano_df = batch_pano(
        pairs_df,
        graph_dir=cfg.paths.graphs,
        k=k,
        item_id_col="item_id",
        condition0_col="condition0",
        condition1_col="condition1",
    )

    valid = pano_df.dropna(subset=["pano_div"])
    logger.info(f"\nPANO summary ({len(valid)} valid pairs):")
    logger.info(f"  mean pano_div: {valid['pano_div'].mean():.4f}")
    logger.info(f"  std  pano_div: {valid['pano_div'].std():.4f}")
    logger.info(f"  min  pano_div: {valid['pano_div'].min():.4f}")
    logger.info(f"  max  pano_div: {valid['pano_div'].max():.4f}")

    by_regime = valid.groupby("regime_label")["pano_div"].agg(["mean", "std", "count"])
    logger.info(f"\nPANO divergence by regime:\n{by_regime.to_string()}")

    # Key sanity check: paraphrase should have LOWER divergence than mistake injection
    a_mean  = valid[valid["regime_label"] == "A"]["pano_div"].mean()
    bm_mean = valid[valid["regime_label"] == "B_mistake"]["pano_div"].mean()
    bt_mean = valid[valid["regime_label"] == "B_trunc"]["pano_div"].mean()
    c_mean  = valid[valid["regime_label"] == "C"]["pano_div"].mean()
    logger.info(f"\n[PANO SANITY CHECK]")
    logger.info(f"  Paraphrase (A):       {a_mean:.4f}  (should be lowest)")
    logger.info(f"  Truncation (B_trunc): {bt_mean:.4f}")
    logger.info(f"  Mistake (B_mistake):  {bm_mean:.4f}  (should be higher than A)")
    logger.info(f"  Hint (C):             {c_mean:.4f}")
    if a_mean < bm_mean:
        logger.info("  [PASS] Paraphrase < Mistake. Positive signal-of-life.")
    else:
        logger.warning("  [WARN] Paraphrase >= Mistake. Check graphs or k value.")

    # ── Compute delta-AGD (needs Regime A as baseline per item) ───────────────
    pano_df = compute_delta_agd(
        pano_df,
        regime_col="regime_label",
        base_id_col="base_item_id",
        item_id_col="item_id",
    )

    delta_valid = pano_df.dropna(subset=["delta_agd"])
    if not delta_valid.empty:
        logger.info(f"\ndelta-AGD summary ({len(delta_valid)} valid pairs):")
        logger.info(f"  mean: {delta_valid['delta_agd'].mean():.4f}")
        logger.info(f"  std:  {delta_valid['delta_agd'].std():.4f}")
        by_regime_d = delta_valid.groupby("regime_label")["delta_agd"].agg(["mean","std","count"])
        logger.info(f"\ndelta-AGD by regime:\n{by_regime_d.to_string()}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = Path(cfg.paths.agd) / "pano_pairs.parquet"
    pano_df.to_parquet(out_path, index=False)
    logger.info(f"\nSaved -> {out_path} ({len(pano_df)} rows)")

    # Print columns written
    logger.info(f"Columns: {list(pano_df.columns)}")


if __name__ == "__main__":
    main()
