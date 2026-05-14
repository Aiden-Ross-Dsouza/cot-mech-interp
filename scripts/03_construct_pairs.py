"""
scripts/03_construct_pairs.py
Construct all paired conditions for Regimes A, B, and C.

Regime A (read from 02's output): filter to kept=True pairs.
Regime B: truncation + add-mistake pairs from cots.jsonl.
Regime C: Turpin hint injection from BBH/MMLU items.

Outputs:
  data/pairs/regime_A_paraphrase.jsonl  (filtered, already exists from 02)
  data/pairs/regime_B_truncate.jsonl
  data/pairs/regime_B_addmistake.jsonl
  data/pairs/regime_C_hint.jsonl

For graph generation (script 04), each output row must have:
  item_id, prompt0, prompt1, target_token (= the predicted answer),
  condition0, condition1, regime, task_type, flipped (Regime C only)

Usage:
    python scripts/03_construct_pairs.py --config config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.behavioral import (
    build_cot_prompt, inject_hint, truncate_cot, add_mistake_to_cot
)


def _write_pairs(writer, rows):
    for row in rows:
        writer.write(row)


def construct_regime_a_pairs(cfg, pilot: bool) -> int:
    """Filter regime A paraphrase file to kept=True and write pair format."""
    src = Path(cfg.paths.pairs) / "regime_A_paraphrase.jsonl"
    dst = Path(cfg.paths.pairs) / "regime_A_pairs.jsonl"
    if not src.exists():
        logger.error(f"Missing {src}. Run script 02 first.")
        return 0

    n = 0
    with jsonlines.open(src) as reader, jsonlines.open(dst, mode="w") as writer:
        items = list(reader)
        for row in tqdm(items, desc="Regime A Pairs"):
            if not row.get("kept", False):
                continue
            question = row["question"]
            choices = row.get("choices")
            cot = row["cot"]
            cot_prime = row["cot_prime"]
            target = row["answer"]

            pair = {
                "item_id": row["item_id"],
                "task": row.get("task", ""),
                "task_type": row.get("task_type", "mcqa"),
                "question": question,
                "regime": "A",
                "condition0": "clean",
                "condition1": "paraphrase",
                "prompt0": build_cot_prompt(question, choices) + cot,
                "prompt1": build_cot_prompt(question, choices) + cot_prime,
                "target_token": target,
                "gold_answer": row["gold_answer"],
                "flipped": False,
            }
            writer.write(pair)
            n += 1
            if pilot and n >= cfg.pilot.n_items:
                break

    logger.info(f"Regime A: {n} pairs → {dst}")
    return n


def construct_regime_b_pairs(cfg, pilot: bool, cots_path: Path = None,
                             out_dir: Path = None) -> int:
    """Construct truncation and add-mistake pairs from cots.jsonl.

    Parameters
    ----------
    cots_path:
        Path to CoTs JSONL (default: cfg.paths.pairs/cots.jsonl).
    out_dir:
        Directory for output pair files (default: cfg.paths.pairs).
    """
    if cots_path is None:
        cots_path = Path(cfg.paths.pairs) / "cots.jsonl"
    if out_dir is None:
        out_dir = Path(cfg.paths.pairs)
    if not cots_path.exists():
        logger.error(f"Missing {cots_path}. Run script 01 (or 01b for Llama) first.")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    dst_trunc = out_dir / "regime_B_truncate.jsonl"
    dst_mistake = out_dir / "regime_B_addmistake.jsonl"

    n_trunc = 0
    n_mistake = 0

    with jsonlines.open(cots_path) as reader, \
         jsonlines.open(dst_trunc, mode="w") as w_trunc, \
         jsonlines.open(dst_mistake, mode="w") as w_mistake:

        items = list(reader)
        for row in tqdm(items, desc="Regime B Pairs"):
            if not row.get("correct", True):
                # Only use items where the model got the right answer on the full CoT
                continue

            question = row["question"]
            choices = row.get("choices")
            cot = row["cot"]
            target = row["predicted_answer"]
            task_type = row.get("task_type", "mcqa")
            iid = row["item_id"]

            base_prompt = build_cot_prompt(question, choices)

            # Truncation at multiple fractions
            for frac in cfg.behavioral.truncation_fractions:
                trunc_cot = truncate_cot(cot, frac)
                pair = {
                    "item_id": f"{iid}_trunc{int(frac*100)}",
                    "base_item_id": iid,
                    "task": row.get("task", ""),
                    "task_type": task_type,
                    "regime": "B",
                    "condition0": "clean",
                    "condition1": f"truncate_{int(frac*100)}",
                    "prompt0": base_prompt + cot,
                    "prompt1": base_prompt + trunc_cot,
                    "target_token": target,
                    "gold_answer": row["gold_answer"],
                    "truncation_fraction": frac,
                    "flipped": False,
                }
                w_trunc.write(pair)
                n_trunc += 1

            # Add-mistake (one version)
            mistake_cot = add_mistake_to_cot(cot)
            pair_mistake = {
                "item_id": f"{iid}_mistake",
                "base_item_id": iid,
                "task": row.get("task", ""),
                "task_type": task_type,
                "regime": "B",
                "condition0": "clean",
                "condition1": "addmistake",
                "prompt0": base_prompt + cot,
                "prompt1": base_prompt + mistake_cot,
                "target_token": target,
                "gold_answer": row["gold_answer"],
                "flipped": False,
            }
            w_mistake.write(pair_mistake)
            n_mistake += 1

            if pilot and (n_trunc // len(cfg.behavioral.truncation_fractions)) >= cfg.pilot.n_items:
                break

    logger.info(f"Regime B truncation: {n_trunc} pairs → {dst_trunc}")
    logger.info(f"Regime B add-mistake: {n_mistake} pairs → {dst_mistake}")
    return n_trunc + n_mistake


def construct_regime_c_pairs(cfg, pilot: bool) -> int:
    """Construct Turpin hint-injection pairs from turpin_prerun.jsonl.

    A1 Fix: This function now reads from the pre-computed turpin_prerun.jsonl
    (generated by script 03b) rather than from raw prompt files. This ensures:

    1. prompt0 = base_prompt + cot_biased  (unbiased context, biased CoT)
    2. prompt1 = biased_prompt + cot_biased (biased context, same CoT)
    3. target_token = biased_answer         (the model's actual output under the hint)

    This implements the research plan's paired-graph design:
      G(M, [x, c_hint], a_hint) vs. G(M, [x_hint, c_hint], a_hint)
    which isolates the causal effect of the hint on the internal mechanism.
    """
    prerun_path = Path(cfg.paths.pairs) / "turpin_prerun.jsonl"
    if not prerun_path.exists():
        logger.error(
            f"Missing {prerun_path}. "
            "Run script 03b first: python scripts/03b_turpin_prerun.py --config config.yaml"
        )
        return 0

    dst = Path(cfg.paths.pairs) / "regime_C_hint.jsonl"
    n = 0

    with jsonlines.open(prerun_path) as reader, jsonlines.open(dst, mode="w") as writer:
        items = list(reader)
        if pilot:
            items = items[:cfg.pilot.n_items]

        for item in tqdm(items, desc="Regime C Pairs (A1 fix)"):
            biased_answer = item.get("biased_answer", "")
            cot_biased = item.get("cot_biased", "")
            base_prompt = item.get("base_prompt", "")
            biased_prompt = item.get("biased_prompt", "")

            if not biased_answer or not cot_biased:
                logger.warning(f"[{item['item_id']}] Missing biased_answer/cot_biased — skipping.")
                continue

            # A1 fix: Both prompts use the SAME biased CoT as context.
            # prompt0 = unbiased question context + biased CoT (isolates hint effect)
            # prompt1 = biased question context   + biased CoT
            # target  = biased_answer (what the model ACTUALLY said under the hint)
            pair = {
                "item_id": item["item_id"],
                "base_item_id": item["base_item_id"],
                "task": item["task"],
                "task_type": "mcqa",
                "regime": "C",
                "condition0": "clean",
                "condition1": "hint",
                "prompt0": base_prompt + cot_biased,    # unbiased context + same biased CoT
                "prompt1": biased_prompt + cot_biased,  # biased context  + same biased CoT
                "target_token": biased_answer,           # A1 fix: attribute to a_hint, not gold
                "gold_answer": item["gold_answer"],
                "hint_choice": item["hint_choice"],
                "flipped": item.get("flipped", None),
                "unfaithful_flip": item.get("unfaithful_flip", None),
            }
            writer.write(pair)
            n += 1

    logger.info(f"Regime C: {n} hint pairs → {dst}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--cot-file", default=None,
                        help="Override CoT input file (default: cfg.paths.pairs/cots.jsonl). "
                             "Use data/pairs/llama/cots_llama.jsonl for Llama campaign.")
    parser.add_argument("--out-dir", default=None,
                        help="Override pair output directory (default: cfg.paths.pairs). "
                             "Use data/pairs/llama/ for Llama campaign.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot = args.pilot
    cot_path = Path(args.cot_file) if args.cot_file else None
    out_dir = Path(args.out_dir) if args.out_dir else None

    if cot_path or out_dir:
        # Llama campaign: only construct Regime B pairs from the Llama CoTs
        logger.info("Constructing Regime B pairs (Llama CoTs)…")
        nB = construct_regime_b_pairs(cfg, pilot, cots_path=cot_path, out_dir=out_dir)
        logger.info(f"Done: B={nB}")
    else:
        logger.info("Constructing Regime A pairs…")
        nA = construct_regime_a_pairs(cfg, pilot)

        logger.info("Constructing Regime B pairs…")
        nB = construct_regime_b_pairs(cfg, pilot)

        logger.info("Constructing Regime C pairs…")
        nC = construct_regime_c_pairs(cfg, pilot)

        logger.info(f"\nAll pairs constructed: A={nA}, B={nB}, C={nC}, total={nA+nB+nC}")


if __name__ == "__main__":
    main()
