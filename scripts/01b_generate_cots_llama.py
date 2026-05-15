"""
scripts/01b_generate_cots_llama.py
Generate Llama-3.2-1B-Instruct CoTs for the cross-model Llama campaign.

Mirrors 01_generate_cots.py but uses cfg.models.robustness (Llama-3.2-1B-Instruct)
with its chat template instead of Gemma-2-2B-it.

Output: data/pairs/llama/cots_llama.jsonl
  {item_id, task, task_type, question, choices, gold_answer, cot, predicted_answer,
   prompt, correct}

Usage:
    python scripts/01b_generate_cots_llama.py --config config.yaml [--pilot]

Pilot: 10 BBH items (cfg.pilot.n_items) to verify non-degeneracy before full campaign.
Full:  all BBH + MMLU items from cfg.dataset (skip GSM8K; numeric format differs).
"""
from __future__ import annotations

# ── Offline mode — use cached HF files; avoids SSL errors on corporate networks
import os as _os
_os.environ["HF_HUB_OFFLINE"] = "1"
_os.environ["TRANSFORMERS_OFFLINE"] = "1"
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
from pathlib import Path

import jsonlines
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_robustness_model
from src.behavioral import extract_answer

SYSTEM_PROMPT = (
    "You are a helpful assistant that reasons carefully step by step. "
    "For multiple-choice questions, end your answer with 'Answer: X' "
    "where X is the correct letter."
)


def build_llama_cot_prompt(question: str, choices, tokenizer) -> str:
    """Build a Llama-chat-template CoT prompt."""
    if choices:
        choice_str = "\n".join(f"({chr(65 + i)}) {c}" for i, c in enumerate(choices))
        user_content = (
            f"Question: {question}\n{choice_str}\n\n"
            "Let's think step by step. Provide your reasoning, then end with "
            "'Answer: X' where X is the correct letter."
        )
    else:
        user_content = (
            f"Question: {question}\n\n"
            "Let's think step by step. Provide your reasoning, then end with "
            "'#### <number>' for the answer."
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    # apply_chat_template adds special tokens and the assistant turn prefix
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_prompts(cfg, pilot: bool = False):
    """Yield item dicts from BBH + MMLU (skip GSM8K for Llama campaign)."""
    prompt_dir = Path(cfg.paths.prompts)
    limit = cfg.pilot.n_items if pilot else None
    count = 0

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

    if limit and count >= limit:
        return

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


@torch.inference_mode()
def generate_llama(model, tokenizer, prompt: str, max_new_tokens: int, seed: int) -> str:
    """Generate text with Llama, returning only the newly generated tokens."""
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_len = inputs["input_ids"].shape[1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_ids = output_ids[0, input_len:]
    return tokenizer.decode(new_ids, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Only process cfg.pilot.n_items items")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model, tokenizer = load_robustness_model(cfg)

    out_dir = Path(cfg.paths.pairs) / "llama"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "cots_llama.jsonl"

    done_ids: set = set()
    if out_path.exists():
        with jsonlines.open(out_path) as r:
            for row in r:
                done_ids.add(row["item_id"])
        logger.info(f"Resuming: {len(done_ids)} items already done.")

    all_items = list(iter_prompts(cfg, pilot=args.pilot))
    logger.info(f"Generating CoTs for {len(all_items)} items with Llama-3.2-1B-Instruct…")
    n_written = 0

    with jsonlines.open(out_path, mode="a") as writer:
        for item in tqdm(all_items, desc="Llama CoTs"):
            iid = item["item_id"]
            if iid in done_ids:
                continue

            question = item["question"]
            choices = item.get("choices")
            gold = item.get("answer", item.get("gold_answer", ""))
            task_type = item.get("task_type", "mcqa")

            prompt = build_llama_cot_prompt(question, choices, tokenizer)

            try:
                cot_full = generate_llama(
                    model, tokenizer, prompt,
                    max_new_tokens=cfg.graph_gen.max_new_tokens,
                    seed=cfg.seed,
                )
                pred = extract_answer(cot_full, task_type)

                def clean(s):
                    if not s:
                        return ""
                    return str(s).replace("(", "").replace(")", "").strip().upper()

                correct = clean(pred) == clean(gold)
                writer.write({
                    "item_id": iid,
                    "task": item.get("task", ""),
                    "task_type": task_type,
                    "question": question,
                    "choices": choices,
                    "gold_answer": str(gold),
                    "cot": cot_full,
                    "predicted_answer": pred,
                    "prompt": prompt,
                    "correct": correct,
                })
                n_written += 1
            except Exception as e:
                logger.error(f"[{iid}] Failed: {e}")

    logger.info(f"Done. Wrote {n_written} new CoTs → {out_path}")


if __name__ == "__main__":
    main()
