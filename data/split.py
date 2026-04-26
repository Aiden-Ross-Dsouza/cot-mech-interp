"""
data/split.py
Create a 60/40 stratified (by task) train/test split over all base item IDs.

Reads all prompt JSONL files, collects item_ids, stratifies by task, then
writes:
  data/train_ids.txt
  data/test_ids.txt

The split is seeded (seed=42) and frozen. Re-running this script always
produces the same split from the same data.

IMPORTANT: Run this BEFORE generating any graphs. The train/test split must
be fixed before the experiment begins (pre-registration requirement).

Usage:
    python data/split.py --config config.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import jsonlines

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config


def collect_items(cfg) -> Dict[str, List[str]]:
    """Collect {task: [item_id, ...]} from all prompt JSONL files."""
    prompt_dir = Path(cfg.paths.prompts)
    task_items: Dict[str, List[str]] = defaultdict(list)

    # BBH
    bbh_dir = prompt_dir / "bbh"
    for subtask in cfg.dataset.bbh_subtasks:
        p = bbh_dir / f"{subtask}.jsonl"
        if not p.exists():
            continue
        with jsonlines.open(p) as reader:
            for item in reader:
                task_items[f"bbh_{subtask}"].append(item["item_id"])

    # MMLU
    mmlu_dir = prompt_dir / "mmlu"
    for cat in cfg.dataset.mmlu_categories:
        p = mmlu_dir / f"{cat}.jsonl"
        if not p.exists():
            continue
        with jsonlines.open(p) as reader:
            for item in reader:
                task_items[f"mmlu_{cat}"].append(item["item_id"])

    # GSM8K
    p = prompt_dir / "gsm8k.jsonl"
    if p.exists():
        with jsonlines.open(p) as reader:
            for item in reader:
                task_items["gsm8k"].append(item["item_id"])

    return task_items


def stratified_split(
    task_items: Dict[str, List[str]],
    train_fraction: float,
    seed: int,
) -> tuple[List[str], List[str]]:
    """Stratified split preserving task proportions."""
    import random
    rng = random.Random(seed)

    train_ids = []
    test_ids = []

    for task, ids in sorted(task_items.items()):
        shuffled = list(ids)
        rng.shuffle(shuffled)
        n_train = max(1, int(len(shuffled) * train_fraction))
        train_ids.extend(shuffled[:n_train])
        test_ids.extend(shuffled[n_train:])
        logger.info(f"  {task}: {n_train} train / {len(shuffled) - n_train} test")

    return sorted(train_ids), sorted(test_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing split files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_dir = Path(cfg.paths.data)
    train_path = data_dir / "train_ids.txt"
    test_path = data_dir / "test_ids.txt"

    if train_path.exists() and test_path.exists() and not args.force:
        # Load existing split for display
        train_ids = train_path.read_text().strip().splitlines()
        test_ids = test_path.read_text().strip().splitlines()
        logger.info(
            f"Split already exists: {len(train_ids)} train / {len(test_ids)} test. "
            "Use --force to regenerate."
        )
        return

    logger.info("Collecting items from prompt JSONL files…")
    task_items = collect_items(cfg)
    total = sum(len(v) for v in task_items.values())
    logger.info(f"Total items: {total} across {len(task_items)} tasks")

    train_ids, test_ids = stratified_split(
        task_items,
        train_fraction=cfg.dataset.train_fraction,
        seed=cfg.seed,
    )

    train_path.write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    test_path.write_text("\n".join(test_ids) + "\n", encoding="utf-8")

    logger.info(f"\n✓ Split saved:")
    logger.info(f"  {len(train_ids)} train → {train_path}")
    logger.info(f"  {len(test_ids)} test  → {test_path}")
    logger.info(
        f"  Actual train fraction: {len(train_ids) / (len(train_ids) + len(test_ids)):.3f}"
    )
    logger.info(
        "\nIMPORTANT: Commit this split to git and do not touch "
        "test_ids.txt until Day 11."
    )


if __name__ == "__main__":
    main()
