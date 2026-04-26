"""
scripts/04_generate_graphs.py
Main attribution-graph campaign.

For every row in the Regime A/B/C pairs JSONL files, generates attribution
graphs for BOTH conditions (prompt0 and prompt1) and saves them as:
  artifacts/graphs/{item_id}_{condition}.json

Fully resumable: skips any graph that already exists on disk.
Checkpoints every N graphs (cfg.graph_gen.checkpoint_every).

Usage:
    python scripts/04_generate_graphs.py --config config.yaml [--pilot] [--regime A|B|C]
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List

import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, Config
from src.model_utils import load_main_model
from src.graph_utils import (
    generate_attribution_graph, save_graph, graph_exists, graph_path
)


PAIR_FILES = {
    "A": ["regime_A_pairs.jsonl"],
    "B": ["regime_B_truncate.jsonl", "regime_B_addmistake.jsonl"],
    "C": ["regime_C_hint.jsonl"],
}


def iter_pair_rows(cfg: Config, regimes: List[str], pilot: bool) -> Iterator[dict]:
    """Yield rows from all requested regime pair files."""
    pairs_dir = Path(cfg.paths.pairs)
    count = 0
    limit = cfg.pilot.n_items * 3 if pilot else None  # ×3 for the 3 regimes

    for regime in regimes:
        for fname in PAIR_FILES.get(regime, []):
            fpath = pairs_dir / fname
            if not fpath.exists():
                logger.warning(f"Pair file not found: {fpath} — skipping.")
                continue
            with jsonlines.open(fpath) as reader:
                for row in reader:
                    yield row
                    count += 1
                    if limit and count >= limit:
                        return


def generate_pair(model, tokenizer, row: dict, cfg: Config) -> int:
    """Generate graphs for both conditions in a row. Returns number of new graphs written."""
    graph_dir = Path(cfg.paths.graphs)
    iid = row["item_id"]
    target = row.get("target_token", "")
    n_new = 0

    for cond_key in ("condition0", "condition1"):
        cond = row[cond_key]
        prompt_key = "prompt0" if cond_key == "condition0" else "prompt1"
        prompt = row[prompt_key]

        if graph_exists(iid, cond, graph_dir):
            continue

        try:
            graph = generate_attribution_graph(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                target_token=target if target else "A",
                cfg=cfg,
                item_id=iid,
                condition=cond,
            )
            save_graph(graph, graph_path(iid, cond, graph_dir))
            n_new += 1
        except Exception as e:
            logger.error(f"  [{iid}/{cond}] Graph generation failed: {e}")

    return n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--regime", default="ABC",
        help="Which regimes to process (e.g. 'A', 'BC', 'ABC')"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    regimes = list(args.regime.upper())

    logger.info(f"Graph generation campaign — regimes: {regimes}, pilot={args.pilot}")
    model, tokenizer = load_main_model(cfg)

    rows = list(iter_pair_rows(cfg, regimes, args.pilot))
    logger.info(f"Total pairs to process: {len(rows)}")

    n_new_total = 0
    checkpoint_every = cfg.graph_gen.checkpoint_every
    t_start = time.time()

    for i, row in enumerate(tqdm(rows, desc="Generating graphs")):
        n_new = generate_pair(model, tokenizer, row, cfg)
        n_new_total += n_new

        if (i + 1) % checkpoint_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_h = (len(rows) - i - 1) / rate / 3600 if rate > 0 else float("inf")
            logger.info(
                f"Checkpoint {i+1}/{len(rows)} — "
                f"{n_new_total} new graphs — "
                f"ETA: {eta_h:.1f}h"
            )

    logger.info(f"\nDone. {n_new_total} new graphs written to {cfg.paths.graphs}")


if __name__ == "__main__":
    main()
