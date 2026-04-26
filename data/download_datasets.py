"""
data/download_datasets.py
Download BBH, MMLU, GSM8K, and Turpin's hint dataset from HuggingFace
and convert to our JSONL schema under data/prompts/.

BBH schema per item:
  {item_id, question, choices, answer, task}

MMLU schema per item:
  {item_id, question, choices, answer, task}

GSM8K schema per item:
  {item_id, question, answer}  (open-ended numeric)

Turpin schema per item:
  {item_id, question, choices, answer, hint_choice, task}

Usage:
    python data/download_datasets.py --config config.yaml [--datasets bbh mmlu gsm8k turpin]
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


# ─────────────────────────────────────────────────────────────────────────────
# BBH (BIG-Bench Hard)
# ─────────────────────────────────────────────────────────────────────────────

def download_bbh(cfg):
    """Download selected BBH subtasks from lukaemon/bbh on HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed: pip install datasets")
        return

    out_dir = Path(cfg.paths.prompts) / "bbh"
    out_dir.mkdir(parents=True, exist_ok=True)

    for subtask in tqdm(cfg.dataset.bbh_subtasks, desc="BBH subtasks"):
        out_path = out_dir / f"{subtask}.jsonl"
        if out_path.exists():
            logger.info(f"  Skipping {subtask} (already exists)")
            continue
        try:
            ds = load_dataset("lukaemon/bbh", subtask, split="test")
            n_items = cfg.dataset.bbh_items_per_subtask
            items = list(ds)[:n_items]

            with jsonlines.open(out_path, mode="w") as writer:
                for i, item in enumerate(items):
                    # BBH items have 'input' and 'target'; multiple-choice items
                    # have choices embedded in 'input' as "(A) ... (B) ..."
                    question_raw = item.get("input", "")
                    target = item.get("target", "")
                    choices, question_clean = _parse_bbh_choices(question_raw, target)

                    writer.write({
                        "item_id": f"bbh_{subtask}_{i:04d}",
                        "question": question_clean,
                        "choices": choices,
                        "answer": target,
                        "task": f"bbh_{subtask}",
                        "task_type": "mcqa" if choices else "open",
                    })
            logger.info(f"  ✓ {subtask}: {len(items)} items")
        except Exception as e:
            logger.error(f"  ✗ {subtask}: {e}")


def _parse_bbh_choices(input_text: str, target: str):
    """Extract choices list and clean question from BBH input text."""
    import re
    choice_pattern = re.compile(r"\(([A-F])\)\s+(.+?)(?=\s*\([A-F]\)|$)", re.DOTALL)
    matches = choice_pattern.findall(input_text)
    if matches:
        choices = [m[1].strip() for m in matches]
        # Remove choice block from question
        question = re.sub(r"\([A-F]\).+", "", input_text, flags=re.DOTALL).strip()
        return choices, question
    return None, input_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# MMLU
# ─────────────────────────────────────────────────────────────────────────────

def download_mmlu(cfg):
    """Download selected MMLU categories from cais/mmlu on HuggingFace."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed.")
        return

    out_dir = Path(cfg.paths.prompts) / "mmlu"
    out_dir.mkdir(parents=True, exist_ok=True)

    for cat in tqdm(cfg.dataset.mmlu_categories, desc="MMLU categories"):
        out_path = out_dir / f"{cat}.jsonl"
        if out_path.exists():
            logger.info(f"  Skipping {cat} (already exists)")
            continue
        try:
            ds = load_dataset("cais/mmlu", cat, split="test")
            n_items = cfg.dataset.mmlu_items_per_cat
            items = list(ds)[:n_items]

            with jsonlines.open(out_path, mode="w") as writer:
                for i, item in enumerate(items):
                    choices = item.get("choices", [])
                    answer_idx = item.get("answer", 0)
                    answer_letter = chr(65 + int(answer_idx))
                    writer.write({
                        "item_id": f"mmlu_{cat}_{i:04d}",
                        "question": item.get("question", ""),
                        "choices": choices,
                        "answer": answer_letter,
                        "task": f"mmlu_{cat}",
                        "task_type": "mcqa",
                    })
            logger.info(f"  ✓ {cat}: {len(items)} items")
        except Exception as e:
            logger.error(f"  ✗ {cat}: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# GSM8K
# ─────────────────────────────────────────────────────────────────────────────

def download_gsm8k(cfg):
    """Download GSM8K from openai/gsm8k."""
    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("datasets not installed.")
        return

    out_path = Path(cfg.paths.prompts) / "gsm8k.jsonl"
    if out_path.exists():
        logger.info("Skipping GSM8K (already exists)")
        return

    try:
        ds = load_dataset("openai/gsm8k", "main", split="test")
        items = list(ds)[:cfg.dataset.gsm8k_items]

        with jsonlines.open(out_path, mode="w") as writer:
            for i, item in enumerate(items):
                # GSM8K answer format: "#### 42"
                answer_raw = item.get("answer", "")
                import re
                m = re.search(r"####\s*(-?[\d,]+)", answer_raw)
                answer_num = m.group(1).replace(",", "") if m else answer_raw
                writer.write({
                    "item_id": f"gsm8k_{i:04d}",
                    "question": item.get("question", ""),
                    "answer": answer_num,
                    "choices": None,
                    "task": "gsm8k",
                    "task_type": "numeric",
                })
        logger.info(f"✓ GSM8K: {len(items)} items → {out_path}")
    except Exception as e:
        logger.error(f"✗ GSM8K: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Turpin hint dataset
# ─────────────────────────────────────────────────────────────────────────────

def download_turpin(cfg):
    """Download / construct Turpin-style hint items.

    Primary source: Turpin et al.'s public dataset if available.
    Fallback: generate synthetic hints from BBH items already downloaded.
    """
    import random
    out_path = Path(cfg.paths.prompts) / "turpin_hints.jsonl"
    if out_path.exists():
        logger.info("Skipping Turpin hints (already exists)")
        return

    # Try loading from HuggingFace; otherwise fall back to synthetic generation
    items = []
    try:
        from datasets import load_dataset
        # Try the publicly released Turpin dataset if available
        ds = load_dataset("mlabonne/Turpin-2023-hints", split="test")
        for i, item in enumerate(list(ds)[:cfg.dataset.turpin_hint_items]):
            items.append({
                "item_id": f"turpin_{i:04d}",
                "question": item.get("question", ""),
                "choices": item.get("choices", []),
                "answer": item.get("answer", ""),
                "hint_choice": item.get("hint_choice", ""),
                "task": item.get("task", "turpin"),
                "task_type": "mcqa",
            })
        logger.info(f"Loaded {len(items)} items from Turpin HF dataset")
    except Exception:
        logger.info("Turpin dataset not on HF — generating synthetic hints from BBH…")

    # Fallback: use BBH items + random wrong-choice hints
    if not items:
        rng = random.Random(42)
        bbh_dir = Path(cfg.paths.prompts) / "bbh"
        for subtask in cfg.dataset.bbh_subtasks:
            fpath = bbh_dir / f"{subtask}.jsonl"
            if not fpath.exists():
                continue
            with jsonlines.open(fpath) as reader:
                for item in reader:
                    choices = item.get("choices", [])
                    correct = item.get("answer", "")
                    if not choices or not correct:
                        continue
                    wrong = [chr(65 + i) for i, c in enumerate(choices)
                             if chr(65 + i) != correct.upper()]
                    if not wrong:
                        continue
                    hint_choice = rng.choice(wrong)
                    items.append({
                        "item_id": f"turpin_{item['item_id']}",
                        "question": item["question"],
                        "choices": choices,
                        "answer": correct,
                        "hint_choice": hint_choice,
                        "task": item["task"],
                        "task_type": "mcqa",
                    })
                    if len(items) >= cfg.dataset.turpin_hint_items:
                        break
            if len(items) >= cfg.dataset.turpin_hint_items:
                break

    with jsonlines.open(out_path, mode="w") as writer:
        for item in items:
            writer.write(item)
    logger.info(f"✓ Turpin hints: {len(items)} items → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--datasets", nargs="*",
        default=["bbh", "mmlu", "gsm8k", "turpin"],
        choices=["bbh", "mmlu", "gsm8k", "turpin"],
    )
    args = parser.parse_args()
    cfg = load_config(args.config)

    if "bbh" in args.datasets:
        logger.info("=== Downloading BBH ===")
        download_bbh(cfg)

    if "mmlu" in args.datasets:
        logger.info("=== Downloading MMLU ===")
        download_mmlu(cfg)

    if "gsm8k" in args.datasets:
        logger.info("=== Downloading GSM8K ===")
        download_gsm8k(cfg)

    if "turpin" in args.datasets:
        logger.info("=== Setting up Turpin hints ===")
        download_turpin(cfg)

    logger.info("\nAll datasets ready.")


if __name__ == "__main__":
    main()
