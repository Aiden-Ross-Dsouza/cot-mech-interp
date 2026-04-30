"""
scripts/01_generate_cots.py
Generate model CoTs and extract predicted answers for all prompts.

For each item in data/prompts/{bbh,mmlu,gsm8k}/*.jsonl, runs Gemma-2-2B-it
with a CoT prompt template and records the generated chain of thought + answer.

Output: data/pairs/cots.jsonl
  {item_id, task, question, choices, gold_answer, cot, predicted_answer,
   prompt, correct, task_type}

Usage:
    python scripts/01_generate_cots.py --config config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_main_model, generate_text
from src.behavioral import build_cot_prompt, extract_answer


def iter_prompts(cfg, pilot: bool = False):
    """Yield item dicts from all prompt JSONL files."""
    prompt_dir = Path(cfg.paths.prompts)
    limit = cfg.pilot.n_items if pilot else None
    count = 0

    # BBH
    bbh_dir = prompt_dir / "bbh"
    for subtask in cfg.dataset.bbh_subtasks:
        p = bbh_dir / f"{subtask}.jsonl"
        if not p.exists():
            logger.warning(f"Missing BBH subtask file: {p}")
            continue
        with jsonlines.open(p) as reader:
            for item in reader:
                item["task"] = f"bbh_{subtask}"
                item["task_type"] = "mcqa"
                yield item
                count += 1
                if limit and count >= limit:
                    return

    # MMLU
    mmlu_dir = prompt_dir / "mmlu"
    for cat in cfg.dataset.mmlu_categories:
        p = mmlu_dir / f"{cat}.jsonl"
        if not p.exists():
            logger.warning(f"Missing MMLU category file: {p}")
            continue
        with jsonlines.open(p) as reader:
            for item in reader:
                item["task"] = f"mmlu_{cat}"
                item["task_type"] = "mcqa"
                yield item
                count += 1
                if limit and count >= limit:
                    return

    # GSM8K
    p = prompt_dir / "gsm8k.jsonl"
    if p.exists():
        with jsonlines.open(p) as reader:
            for item in reader:
                item["task"] = "gsm8k"
                item["task_type"] = "numeric"
                yield item
                count += 1
                if limit and count >= limit:
                    return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help=f"Only process the first N items (pilot run)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_main_model(cfg)

    out_path = Path(cfg.paths.pairs) / "cots.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-processed item_ids
    done_ids = set()
    if out_path.exists():
        with jsonlines.open(out_path) as r:
            for row in r:
                done_ids.add(row["item_id"])
        logger.info(f"Resuming: {len(done_ids)} items already done.")

    n_written = 0
    all_items = list(iter_prompts(cfg, pilot=args.pilot))
    with jsonlines.open(out_path, mode="a") as writer:
        for item in tqdm(all_items, desc="Generating CoTs"):
            iid = item["item_id"]
            if iid in done_ids:
                continue

            question = item["question"]
            choices = item.get("choices")
            gold = item.get("answer", item.get("gold_answer", ""))
            task_type = item.get("task_type", "mcqa")

            prompt = build_cot_prompt(question, choices)

            try:
                cot_full = generate_text(
                    model, tokenizer, prompt,
                    max_new_tokens=cfg.graph_gen.max_new_tokens,
                    seed=cfg.seed,
                )
                pred = extract_answer(cot_full, task_type)
                # Robust answer comparison (strip parentheses)
                def clean(s):
                    if not s: return ""
                    return str(s).replace("(", "").replace(")", "").strip().upper()

                correct = clean(pred) == clean(gold)

                writer.write({
                    "item_id": iid,
                    "task": item.get("task", ""),
                    "task_type": task_type,
                    "question": question,
                    "choices": choices,
                    "gold_answer": gold,
                    "cot": cot_full,
                    "predicted_answer": pred,
                    "prompt": prompt,
                    "correct": correct,
                })
                n_written += 1

            except Exception as e:
                logger.error(f"[{iid}] CoT generation failed: {e}")

    logger.info(f"Done. {n_written} CoTs written to {out_path}")


if __name__ == "__main__":
    main()
