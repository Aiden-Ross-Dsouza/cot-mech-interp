"""
scripts/07_compute_baselines.py
Compute all 5 baseline measures for all pairs → artifacts/baselines.parquet

Baselines:
  1. activation_cosine   (requires TransformerLens + activation .npz files)
  2. kl_next_token       (HF model)
  3. cot_perplexity      (HF model)
  4. sc_variance         (HF model, N=8 samples — slow)
  5. random_jaccard      (graphs + random sampling)

Output columns:
  item_id, regime_label, activation_cosine, kl_next_token,
  cot_perplexity, sc_variance, random_jaccard

Usage:
    python scripts/07_compute_baselines.py --config config.yaml [--pilot]
    python scripts/07_compute_baselines.py --config config.yaml --skip sc_variance
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import jsonlines
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.model_utils import load_main_model, load_lens
from src.baselines import (
    activation_cosine as act_cos_fn,
    kl_next_token,
    cot_perplexity,
    self_consistency_variance,
    random_feature_jaccard,
)
from src.graph_utils import load_graph, graph_path


PAIR_FILES = [
    "regime_A_pairs.jsonl",
    "regime_B_truncate.jsonl",
    "regime_B_addmistake.jsonl",
    "regime_C_hint.jsonl",
]


def load_act_npz(item_id: str, condition: str, act_dir: Path) -> Optional[np.ndarray]:
    """Load activation NPZ and return a stacked [n_layers, d_model] array."""
    p = act_dir / f"{item_id}_{condition}.npz"
    if not p.exists():
        return None
    data = np.load(p)
    keys = sorted(data.files, key=lambda k: int(k.split("_")[1]))
    return np.stack([data[k] for k in keys], axis=0)  # [n_layers, d_model]


def cosine_from_npz(acts0: np.ndarray, acts1: np.ndarray) -> float:
    """Layer-averaged cosine from stacked activation arrays."""
    cosines = []
    for a0, a1 in zip(acts0, acts1):
        n0, n1 = np.linalg.norm(a0), np.linalg.norm(a1)
        if n0 == 0 or n1 == 0:
            continue
        c = float(np.dot(a0, a1) / (n0 * n1))
        cosines.append((1.0 + c) / 2.0)
    return float(np.mean(cosines)) if cosines else float("nan")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--skip", nargs="*", default=[],
                        help="Baselines to skip (e.g. --skip sc_variance kl_next_token)")
    args = parser.parse_args()
    skip = set(args.skip)

    cfg = load_config(args.config)
    act_dir = Path(cfg.paths.activations)
    graph_dir = Path(cfg.paths.graphs)
    pairs_dir = Path(cfg.paths.pairs)

    # Load HF model (for KL, PPL, SC)
    model_hf, tokenizer = load_main_model(cfg)

    records = []
    seen = set()

    for fname in PAIR_FILES:
        fpath = pairs_dir / fname
        if not fpath.exists():
            continue
        with jsonlines.open(fpath) as reader:
            rows = list(reader)

        if args.pilot:
            rows = rows[:cfg.pilot.n_items]

        for row in tqdm(rows, desc=f"Baselines [{fname}]"):
            iid = row["item_id"]
            if iid in seen:
                continue
            seen.add(iid)

            c0, c1 = row["condition0"], row["condition1"]
            p0, p1 = row["prompt0"], row["prompt1"]
            cot0 = row.get("cot", "")

            result = {"item_id": iid, "regime_label": fname.split(".")[0].replace("regime_", "")}

            # 1. Activation-cosine (from pre-extracted .npz files)
            if "activation_cosine" not in skip:
                acts0 = load_act_npz(iid, c0, act_dir)
                acts1 = load_act_npz(iid, c1, act_dir)
                if acts0 is not None and acts1 is not None:
                    result["activation_cosine"] = cosine_from_npz(acts0, acts1)
                else:
                    result["activation_cosine"] = float("nan")

            # 2. KL next-token
            if "kl_next_token" not in skip:
                try:
                    result["kl_next_token"] = kl_next_token(model_hf, tokenizer, p0, p1)
                except Exception as e:
                    logger.warning(f"[{iid}] kl_next_token: {e}")
                    result["kl_next_token"] = float("nan")

            # 3. CoT perplexity
            if "cot_perplexity" not in skip:
                try:
                    result["cot_perplexity"] = cot_perplexity(model_hf, tokenizer, p0, cot0)
                except Exception as e:
                    logger.warning(f"[{iid}] cot_perplexity: {e}")
                    result["cot_perplexity"] = float("nan")

            # 4. Self-consistency variance (expensive)
            if "sc_variance" not in skip:
                try:
                    result["sc_variance"] = self_consistency_variance(
                        model_hf, tokenizer, p0,
                        n=cfg.behavioral.n_self_consistency,
                        seed=cfg.seed,
                    )
                except Exception as e:
                    logger.warning(f"[{iid}] sc_variance: {e}")
                    result["sc_variance"] = float("nan")

            # 5. Random-feature Jaccard
            if "random_jaccard" not in skip:
                g0_path = graph_path(iid, c0, graph_dir)
                g1_path = graph_path(iid, c1, graph_dir)
                if g0_path.exists() and g1_path.exists():
                    try:
                        g0 = load_graph(g0_path)
                        g1 = load_graph(g1_path)
                        result["random_jaccard"] = random_feature_jaccard(
                            g0, g1, k=cfg.agd.k, seed=cfg.seed
                        )
                    except Exception as e:
                        logger.warning(f"[{iid}] random_jaccard: {e}")
                        result["random_jaccard"] = float("nan")
                else:
                    result["random_jaccard"] = float("nan")

            records.append(result)

    df = pd.DataFrame(records)
    out_path = Path(cfg.paths.agd).parent / "baselines.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"\nDone. {len(df)} baseline rows → {out_path}")
    logger.info(f"\n{df.describe().to_string()}")


if __name__ == "__main__":
    main()
