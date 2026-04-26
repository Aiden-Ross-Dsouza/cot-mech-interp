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
        for row in reader:
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


def construct_regime_b_pairs(cfg, pilot: bool) -> int:
    """Construct truncation and add-mistake pairs from cots.jsonl."""
    cots_path = Path(cfg.paths.pairs) / "cots.jsonl"
    if not cots_path.exists():
        logger.error(f"Missing {cots_path}. Run script 01 first.")
        return 0

    dst_trunc = Path(cfg.paths.pairs) / "regime_B_truncate.jsonl"
    dst_mistake = Path(cfg.paths.pairs) / "regime_B_addmistake.jsonl"

    n_trunc = 0
    n_mistake = 0

    with jsonlines.open(cots_path) as reader, \
         jsonlines.open(dst_trunc, mode="w") as w_trunc, \
         jsonlines.open(dst_mistake, mode="w") as w_mistake:

        for row in reader:
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
    """Construct Turpin hint-injection pairs from BBH/MMLU items."""
    prompt_dir = Path(cfg.paths.prompts)
    dst = Path(cfg.paths.pairs) / "regime_C_hint.jsonl"

    rng = random.Random(cfg.seed)
    n = 0

    sources = []
    for subtask in cfg.dataset.bbh_subtasks:
        p = prompt_dir / "bbh" / f"{subtask}.jsonl"
        if p.exists():
            sources.append(("bbh", subtask, p))

    for cat in cfg.dataset.mmlu_categories:
        p = prompt_dir / "mmlu" / f"{cat}.jsonl"
        if p.exists():
            sources.append(("mmlu", cat, p))

    with jsonlines.open(dst, mode="w") as writer:
        for (domain, subtask, p) in sources:
            with jsonlines.open(p) as reader:
                items = list(reader)

            for item in items:
                choices = item.get("choices", [])
                correct = item.get("answer", "")
                if not choices or not correct:
                    continue

                # Choose a WRONG choice as the hint
                wrong_choices = [
                    chr(65 + i) for i, c in enumerate(choices)
                    if chr(65 + i) != correct.upper()
                ]
                if not wrong_choices:
                    continue
                hint_choice = rng.choice(wrong_choices)

                iid = item["item_id"]
                question = item["question"]
                base_prompt = build_cot_prompt(question, choices)
                biased_prompt = inject_hint(base_prompt, hint_choice)

                pair = {
                    "item_id": f"{iid}_hint{hint_choice}",
                    "base_item_id": iid,
                    "task": f"{domain}_{subtask}",
                    "task_type": "mcqa",
                    "regime": "C",
                    "condition0": "clean",
                    "condition1": "hint",
                    "prompt0": base_prompt,           # unbiased (no CoT, model generates)
                    "prompt1": biased_prompt,          # with hint
                    "target_token": correct,           # we attribute to the correct answer
                    "gold_answer": correct,
                    "hint_choice": hint_choice,
                    "flipped": None,  # determined AFTER running script 08
                }
                writer.write(pair)
                n += 1

                if pilot and n >= cfg.pilot.n_items:
                    break
            if pilot and n >= cfg.pilot.n_items:
                break

    logger.info(f"Regime C: {n} hint pairs → {dst}")
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    pilot = args.pilot

    logger.info("Constructing Regime A pairs…")
    nA = construct_regime_a_pairs(cfg, pilot)

    logger.info("Constructing Regime B pairs…")
    nB = construct_regime_b_pairs(cfg, pilot)

    logger.info("Constructing Regime C pairs…")
    nC = construct_regime_c_pairs(cfg, pilot)

    logger.info(f"\nAll pairs constructed: A={nA}, B={nB}, C={nC}, total={nA+nB+nC}")


if __name__ == "__main__":
    main()
