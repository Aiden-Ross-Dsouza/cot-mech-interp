"""
scripts/08_behavioral_measures.py
Run Lanham AOC and Turpin hint-flip protocols on all items.

Outputs:
  artifacts/behavioral/aoc_lanham.parquet
    item_id, task, aoc_composite, aoc_early, aoc_truncate_25/50/75, aoc_mistake

  artifacts/behavioral/turpin_flips.parquet
    item_id, task, unbiased_answer, biased_answer, flipped, cot_mentions_hint,
    unfaithful_flip, cot_unbiased, cot_biased

Also updates regime_C_hint.jsonl with the resolved `flipped` column.

Usage:
    python scripts/08_behavioral_measures.py --config config.yaml [--pilot]
    python scripts/08_behavioral_measures.py --config config.yaml --task aoc
    python scripts/08_behavioral_measures.py --config config.yaml --task turpin
"""
from __future__ import annotations

import argparse
import json
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
from src.model_utils import load_main_model
from src.behavioral import compute_aoc, run_turpin_protocol


def run_aoc(cfg, model, tokenizer, pilot: bool):
    """Run Lanham AOC on all items from cots.jsonl."""
    cots_path = Path(cfg.paths.pairs) / "cots.jsonl"
    if not cots_path.exists():
        logger.error("Missing cots.jsonl. Run script 01 first.")
        return

    with jsonlines.open(cots_path) as reader:
        items = list(reader)

    if pilot:
        items = items[:cfg.pilot.n_items]

    # Run AOC
    df = compute_aoc(
        model=model,
        tokenizer=tokenizer,
        items=items,
        max_new_tokens=cfg.graph_gen.max_new_tokens,
        seed=cfg.seed,
        truncation_fractions=list(cfg.behavioral.truncation_fractions),
    )

    out_dir = Path(cfg.paths.behavioral)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "aoc_lanham.parquet"
    df.to_parquet(out_path, index=False)

    logger.info(f"\nAOC results ({len(df)} items):")
    logger.info(f"  Mean composite AOC: {df['aoc_composite'].mean():.3f}")
    logger.info(f"  Saved → {out_path}")


def run_turpin(cfg, model, tokenizer, pilot: bool):
    """Run Turpin hint-injection protocol on Regime C items."""
    c_path = Path(cfg.paths.pairs) / "regime_C_hint.jsonl"
    if not c_path.exists():
        logger.error("Missing regime_C_hint.jsonl. Run script 03 first.")
        return

    with jsonlines.open(c_path) as reader:
        all_rows = list(reader)

    # Build items list for Turpin protocol
    items = []
    for row in all_rows:
        items.append({
            "item_id": row["item_id"],
            "question": row.get("question", ""),
            "choices": row.get("choices", []),
            "answer": row.get("gold_answer", ""),
            "hint_choice": row.get("hint_choice", ""),
        })

    if pilot:
        items = items[:cfg.pilot.n_items]

    df = run_turpin_protocol(
        model=model,
        tokenizer=tokenizer,
        items=items,
        max_new_tokens=cfg.graph_gen.max_new_tokens,
        seed=cfg.seed,
    )

    out_dir = Path(cfg.paths.behavioral)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "turpin_flips.parquet"
    df.to_parquet(out_path, index=False)

    n_flipped = df["flipped"].sum()
    n_unfaithful = df["unfaithful_flip"].sum()
    logger.info(f"\nTurpin results ({len(df)} items):")
    logger.info(f"  Flipped: {n_flipped}/{len(df)} ({100*n_flipped/max(1,len(df)):.1f}%)")
    logger.info(f"  Unfaithful flips: {n_unfaithful}/{len(df)}")
    logger.info(f"  Saved → {out_path}")

    if n_flipped < 100:
        logger.warning(
            f"WARNING: Only {n_flipped} flipped items. "
            "H2 AUROC analysis may have insufficient power. "
            "Consider supplementing with synthetic hint pairs."
        )

    # Merge flip labels back into regime_C_hint.jsonl
    flip_map = dict(zip(df["item_id"], df["unfaithful_flip"]))
    updated_rows = []
    for row in all_rows:
        row["flipped"] = bool(flip_map.get(row["item_id"], False))
        updated_rows.append(row)

    with jsonlines.open(c_path, mode="w") as writer:
        for row in updated_rows:
            writer.write(row)
    logger.info(f"Updated {c_path} with flip labels.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--task", choices=["aoc", "turpin", "both"], default="both")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_main_model(cfg)

    if args.task in ("aoc", "both"):
        logger.info("=== Running Lanham AOC ===")
        run_aoc(cfg, model, tokenizer, args.pilot)

    if args.task in ("turpin", "both"):
        logger.info("=== Running Turpin hint-flip protocol ===")
        run_turpin(cfg, model, tokenizer, args.pilot)

    logger.info("\nScript 08 complete.")


if __name__ == "__main__":
    main()
