"""
scripts/05_extract_activations.py
Extract residual-stream activations at the answer-token position for all pairs.

Used for the activation-cosine baseline (baseline #1).
Activations are saved as .npz files:
  artifacts/activations/{item_id}_{condition}.npz
    keys: resid_{layer_id}  → shape [d_model]

Uses TransformerLens HookedTransformer for clean activation access.

Usage:
    python scripts/05_extract_activations.py --config config.yaml [--pilot]
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

import jsonlines
import numpy as np
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_lens


def extract_activations(
    model,  # HookedTransformer
    prompt: str,
    device: str = "cuda",
) -> dict[str, np.ndarray]:
    """Extract residual-stream at last token for all layers. Returns dict of np arrays."""
    _, cache = model.run_with_cache(
        prompt,
        names_filter=lambda name: name.endswith("hook_resid_post"),
        return_type=None,
    )
    result = {}
    for key, val in cache.items():
        # key: 'blocks.{layer}.hook_resid_post', shape: [1, seq_len, d_model]
        layer_id = int(key.split(".")[1])
        result[f"resid_{layer_id}"] = val[0, -1, :].float().cpu().numpy()
    return result


def activation_path(item_id: str, condition: str, act_dir: Path) -> Path:
    return act_dir / f"{item_id}_{condition}.npz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    act_dir = Path(cfg.paths.activations)
    act_dir.mkdir(parents=True, exist_ok=True)

    # TransformerLens uses a simplified model name (no -it suffix officially)
    # Gemma-2-2B-it maps to 'gemma-2-2b' in TL's model zoo
    lens_model_name = "gemma-2-2b"
    model = load_lens(lens_model_name, device=cfg.models.main.device)

    # Collect all (item_id, condition, prompt) triples from pair files
    pairs_dir = Path(cfg.paths.pairs)
    triples: List[tuple] = []
    seen = set()

    for fname in ["regime_A_pairs.jsonl", "regime_B_truncate.jsonl",
                  "regime_B_addmistake.jsonl", "regime_C_hint.jsonl"]:
        fpath = pairs_dir / fname
        if not fpath.exists():
            continue
        with jsonlines.open(fpath) as reader:
            for row in reader:
                iid = row["item_id"]
                for cond_key, prompt_key in [("condition0", "prompt0"), ("condition1", "prompt1")]:
                    cond = row[cond_key]
                    prompt = row[prompt_key]
                    key = (iid, cond)
                    if key not in seen:
                        seen.add(key)
                        triples.append((iid, cond, prompt))

    if args.pilot:
        triples = triples[:cfg.pilot.n_items * 2]

    logger.info(f"Extracting activations for {len(triples)} (item, condition) pairs…")

    n_new = 0
    for iid, cond, prompt in tqdm(triples, desc="Extracting"):
        out_path = activation_path(iid, cond, act_dir)
        if out_path.exists():
            continue
        try:
            with torch.inference_mode():
                acts = extract_activations(model, prompt, device=cfg.models.main.device)
            np.savez_compressed(out_path, **acts)
            n_new += 1
        except Exception as e:
            logger.error(f"[{iid}/{cond}] Activation extraction failed: {e}")

    logger.info(f"Done. {n_new} new activation files written to {act_dir}")


if __name__ == "__main__":
    main()
