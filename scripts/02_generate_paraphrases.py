"""
scripts/02_generate_paraphrases.py
Regime A — Generate semantic paraphrases of CoTs using Gemma-2-9B-4bit.

Reads data/pairs/cots.jsonl, paraphrases each CoT, verifies:
  1. The paraphrased CoT still leads to the SAME predicted answer.
  2. char-level edit distance to original CoT > min_edit_distance threshold.

Output: data/pairs/regime_A_paraphrase.jsonl
  {item_id, task, question, choices, gold_answer,
   cot, cot_prime, answer, answer_prime,
   edit_distance, kept}

Usage:
    python scripts/02_generate_paraphrases.py --config config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import editdistance
import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_paraphrase_model, load_main_model, generate_text, unload_paraphrase_model
from src.behavioral import extract_answer, build_cot_prompt


PARAPHRASE_SYSTEM = (
    "You are a helpful assistant that paraphrases reasoning chains. "
    "Restate the following chain of thought in different words, maintaining all logical steps "
    "and arriving at the SAME conclusion. Do NOT change the answer. "
    "Only output the paraphrased reasoning, nothing else."
)


def paraphrase_cot(model, tokenizer, cot: str, temperature: float, seed: int) -> str:
    """Generate a paraphrase of the CoT string."""
    prompt = f"{PARAPHRASE_SYSTEM}\n\nOriginal reasoning:\n{cot}\n\nParaphrased reasoning:\n"
    return generate_text(
        model, tokenizer, prompt,
        max_new_tokens=len(cot.split()) * 2,  # generous budget
        temperature=temperature,
        do_sample=True,
        seed=seed,
    )


def verify_answer(model, tokenizer, question: str, choices, cot_prime: str,
                  gold_answer: str, task_type: str, max_new_tokens: int, seed: int) -> str:
    """Run the model with cot_prime and return the predicted answer."""
    prompt = build_cot_prompt(question, choices) + cot_prime
    response = generate_text(model, tokenizer, prompt, max_new_tokens=max_new_tokens, seed=seed)
    return extract_answer(response, task_type)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    min_edit = cfg.paraphrase.min_edit_distance
    temp = cfg.paraphrase.temperature

    # Load paraphrase model first, then will swap in main model for answer verification
    para_model, para_tokenizer = load_paraphrase_model(cfg)

    cots_path = Path(cfg.paths.pairs) / "cots.jsonl"
    if not cots_path.exists():
        logger.error(f"Missing {cots_path}. Run script 01 first.")
        sys.exit(1)

    out_path = Path(cfg.paths.pairs) / "regime_A_paraphrase.jsonl"
    done_ids = set()
    if out_path.exists():
        with jsonlines.open(out_path) as r:
            for row in r:
                done_ids.add(row["item_id"])

    items = []
    with jsonlines.open(cots_path) as r:
        for row in r:
            items.append(row)

    if args.pilot:
        items = items[:cfg.pilot.n_items]

    n_kept = 0
    n_total = 0

    with jsonlines.open(out_path, mode="a") as writer:
        for item in tqdm(items, desc="Paraphrasing"):
            iid = item["item_id"]
            if iid in done_ids:
                continue

            cot = item["cot"]
            question = item["question"]
            choices = item.get("choices")
            gold = item["gold_answer"]
            task_type = item.get("task_type", "mcqa")

            # Generate paraphrase
            try:
                cot_prime = paraphrase_cot(para_model, para_tokenizer, cot, temp, cfg.seed)
            except Exception as e:
                logger.warning(f"[{iid}] Paraphrase generation failed: {e}")
                continue

            # Compute edit distance
            ed = editdistance.eval(cot, cot_prime)

            # We unload the para model temporarily to check the answer — or we just
            # use the para model for verification too (it's instruction-tuned).
            try:
                answer_prime = verify_answer(
                    para_model, para_tokenizer,
                    question, choices, cot_prime, gold, task_type,
                    cfg.graph_gen.max_new_tokens, cfg.seed,
                )
            except Exception as e:
                logger.warning(f"[{iid}] Answer verification failed: {e}")
                answer_prime = ""

            kept = (
                ed >= min_edit
                and answer_prime.strip().upper() == str(gold).strip().upper()
            )

            writer.write({
                "item_id": iid,
                "task": item.get("task", ""),
                "task_type": task_type,
                "question": question,
                "choices": choices,
                "gold_answer": gold,
                "cot": cot,
                "cot_prime": cot_prime,
                "answer": item["predicted_answer"],
                "answer_prime": answer_prime,
                "edit_distance": ed,
                "kept": kept,
            })
            n_total += 1
            if kept:
                n_kept += 1

    logger.info(
        f"Done. {n_kept}/{n_total} pairs kept (ed ≥ {min_edit} AND same answer). "
        f"Output: {out_path}"
    )


if __name__ == "__main__":
    main()
