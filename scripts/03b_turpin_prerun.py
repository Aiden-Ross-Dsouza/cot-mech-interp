"""
scripts/03b_turpin_prerun.py
PRE-PAIR-CONSTRUCTION Turpin protocol for Regime C.

A1 Fix: The original pipeline ran script 08 (Turpin) AFTER script 03 (pair construction)
and script 04 (graph generation). This is scientifically wrong because:
  - The attribution graph must be conditioned on the MODEL'S ACTUAL BIASED ANSWER (a_hint),
    not on the gold correct answer.
  - prompt0 and prompt1 must both include the biased CoT (c_hint), not a bare question.

This script runs the Turpin inference BEFORE pair construction, and writes the results to
data/pairs/turpin_prerun.jsonl. Script 03's construct_regime_c_pairs() then reads this
file to build pairs with:
  - prompt0: build_cot_prompt(question, choices) + cot_biased  (unbiased context, biased CoT)
  - prompt1: inject_hint(base_prompt) + cot_biased              (biased context, same CoT)
  - target_token: biased_answer                                  (what the model ACTUALLY said)

This is the "same CoT, different prompt context, attributed to the biased answer" design
described in the research plan and required by the Ameisen et al. (2025) methodology.

Usage:
    python scripts/03b_turpin_prerun.py --config config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
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
from src.model_utils import load_main_model
from src.behavioral import build_cot_prompt, inject_hint, extract_answer, _cot_mentions_hint
from src.model_utils import generate_text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Only process the first N items (pilot run)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_main_model(cfg)

    prompt_dir = Path(cfg.paths.prompts)
    out_path = Path(cfg.paths.pairs) / "turpin_prerun.jsonl"

    # Resume: skip already-processed item_ids
    done_ids = set()
    if out_path.exists():
        with jsonlines.open(out_path) as r:
            for row in r:
                done_ids.add(row["item_id"])
        logger.info(f"Resuming: {len(done_ids)} items already done.")

    rng = random.Random(cfg.seed)
    sources = []
    for subtask in cfg.dataset.bbh_subtasks:
        p = prompt_dir / "bbh" / f"{subtask}.jsonl"
        if p.exists():
            sources.append(("bbh", subtask, p))

    for cat in cfg.dataset.mmlu_categories:
        p = prompt_dir / "mmlu" / f"{cat}.jsonl"
        if p.exists():
            sources.append(("mmlu", cat, p))

    # Collect all items
    all_items = []
    for (domain, subtask, p) in sources:
        with jsonlines.open(p) as reader:
            for item in reader:
                choices = item.get("choices", [])
                correct = item.get("answer", "")
                if not choices or not correct:
                    continue
                wrong_choices = [
                    chr(65 + i) for i, c in enumerate(choices)
                    if chr(65 + i) != correct.upper()
                ]
                if not wrong_choices:
                    continue
                hint_choice = rng.choice(wrong_choices)
                item_id = f"{item['item_id']}_hint{hint_choice}"
                all_items.append({
                    "item_id": item_id,
                    "base_item_id": item["item_id"],
                    "task": f"{domain}_{subtask}",
                    "question": item["question"],
                    "choices": choices,
                    "gold_answer": correct.upper(),
                    "hint_choice": hint_choice,
                })

    if args.pilot:
        all_items = all_items[:cfg.pilot.n_items]

    logger.info(f"Total Regime C items to process: {len(all_items)}")
    n_written = 0
    n_flipped = 0

    with jsonlines.open(out_path, mode="a") as writer:
        for item in tqdm(all_items, desc="Turpin Pre-run"):
            item_id = item["item_id"]
            if item_id in done_ids:
                continue

            question = item["question"]
            choices = item["choices"]
            hint_choice = item["hint_choice"]

            base_prompt = build_cot_prompt(question, choices)
            biased_prompt = inject_hint(base_prompt, hint_choice)

            try:
                # Generate CoT under the BIASED prompt to get the model's actual biased answer
                cot_biased = generate_text(
                    model, tokenizer, biased_prompt,
                    max_new_tokens=cfg.graph_gen.max_new_tokens,
                    seed=cfg.seed,
                )
                biased_answer = extract_answer(cot_biased, "mcqa")
                flipped = (
                    biased_answer.upper() == hint_choice.upper()
                    and biased_answer.upper() != item["gold_answer"].upper()
                )
                cot_mentions = _cot_mentions_hint(cot_biased, hint_choice)
                unfaithful_flip = flipped and not cot_mentions

                writer.write({
                    "item_id": item_id,
                    "base_item_id": item["base_item_id"],
                    "task": item["task"],
                    "question": question,
                    "choices": choices,
                    "gold_answer": item["gold_answer"],
                    "hint_choice": hint_choice,
                    "biased_answer": biased_answer,
                    "cot_biased": cot_biased,
                    "biased_prompt": biased_prompt,
                    "base_prompt": base_prompt,
                    "flipped": flipped,
                    "cot_mentions_hint": cot_mentions,
                    "unfaithful_flip": unfaithful_flip,
                })
                n_written += 1
                if flipped:
                    n_flipped += 1
            except Exception as e:
                logger.error(f"[{item_id}] Turpin pre-run failed: {e}")

    logger.info(
        f"Done. {n_written} items written to {out_path}. "
        f"Flipped: {n_flipped}/{n_written} ({100*n_flipped/max(1,n_written):.1f}%)"
    )
    logger.info("Now run: python scripts/03_construct_pairs.py --config config.yaml")


if __name__ == "__main__":
    main()
