"""
scripts/06_compute_agd.py
Compute AGD + components for all graph pairs → artifacts/agd/pairs.parquet

Reads all regime pair files, loads corresponding graphs from artifacts/graphs/,
calls batch_agd(), and writes a single parquet with:
  item_id, task, regime, condition0, condition1, agd, jw, se, n0, n1,
  gold_answer, target_token, [hint_choice, flipped (Regime C)]

Usage:
    python scripts/06_compute_agd.py --config config.yaml [--pilot]
    python scripts/06_compute_agd.py --config config.yaml --alpha 0.3 --k 32
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import jsonlines
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.agd import batch_agd


PAIR_FILES = {
    "A": "regime_A_pairs.jsonl",
    "B_trunc": "regime_B_truncate.jsonl",
    "B_mistake": "regime_B_addmistake.jsonl",
    "C": "regime_C_hint.jsonl",
}


def load_all_pairs(cfg, pilot: bool) -> pd.DataFrame:
    """Load all pair JSONL files into a single DataFrame."""
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

    df = pd.DataFrame(all_rows)
    if pilot and len(df):
        # Pilot: keep only the first n_items rows per regime
        n = cfg.pilot.n_items
        df = df.groupby("regime_label").head(n).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (default: from config)")
    parser.add_argument("--k", type=int, default=None,
                        help="Override k (default: from config)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    alpha = args.alpha if args.alpha is not None else cfg.agd.alpha
    k = args.k if args.k is not None else cfg.agd.k
    top_edges = cfg.agd.top_edges

    logger.info(f"Computing AGD: alpha={alpha}, k={k}, top_edges={top_edges}")

    pairs_df = load_all_pairs(cfg, args.pilot)
    if pairs_df.empty:
        logger.error("No pair data found. Run scripts 01–03 first.")
        sys.exit(1)

    logger.info(f"Total pairs: {len(pairs_df)}")

    # Compute AGD
    result_df = batch_agd(
        pairs_df=pairs_df,
        graph_dir=cfg.paths.graphs,
        alpha=alpha,
        k=k,
        top_edges=top_edges,
    )

    # Summary statistics
    valid = result_df.dropna(subset=["agd"])
    logger.info(f"\nAGD summary ({len(valid)} valid pairs):")
    logger.info(f"  mean: {valid['agd'].mean():.3f}")
    logger.info(f"  std:  {valid['agd'].std():.3f}")
    logger.info(f"  min:  {valid['agd'].min():.3f}")
    logger.info(f"  max:  {valid['agd'].max():.3f}")

    by_regime = valid.groupby("regime_label")["agd"].agg(["mean", "std", "count"])
    logger.info(f"\nBy regime:\n{by_regime.to_string()}")

    # Pilot gate check
    if args.pilot:
        paraphrase_agd = valid[valid["regime_label"] == "A"]["agd"].mean()
        mistake_agd = valid[valid["regime_label"].isin(["B_trunc", "B_mistake"])]["agd"].mean()
        logger.info(f"\n[PILOT GATE]")
        logger.info(f"  Paraphrase AGD (should be LOW):  {paraphrase_agd:.3f}")
        logger.info(f"  Mistake/Trunc AGD (should be higher): {mistake_agd:.3f}")
        if paraphrase_agd < mistake_agd:
            logger.info("  ✓ Signal-of-life check PASSED (paraphrase < mistake).")
        else:
            logger.warning("  ✗ Signal-of-life check FAILED. Inspect graphs.")

    # Save
    out_dir = Path(cfg.paths.agd)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "pairs.parquet"
    result_df.to_parquet(out_path, index=False)
    logger.info(f"\nSaved → {out_path} ({len(result_df)} rows)")


if __name__ == "__main__":
    main()
