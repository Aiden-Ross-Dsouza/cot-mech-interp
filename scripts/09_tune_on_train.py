"""
scripts/09_tune_on_train.py
Tune (alpha, k) on the TRAINING half only.

Reads:
  - artifacts/agd/pairs.parquet       (AGD for all pairs, must be rerun per alpha/k)
  - artifacts/behavioral/aoc_lanham.parquet
  - artifacts/behavioral/turpin_flips.parquet
  - data/train_ids.txt                (60% train split from data/split.py)

Procedure:
  For each (alpha, k) in the grid:
    1. Re-compute AGD restricted to training items.
    2. Evaluate H1 (Spearman rho) and H2 (AUROC) on training half.
    3. Pick (alpha, k) maximizing H2 AUROC (primary metric for hint-flip prediction).

Saves analysis/best_hyperparams.json with:
  {alpha, k, train_h1_rho, train_h2_auroc, grid_results}

IMPORTANT: This script MUST NOT be run until prereg.md is committed.

Usage:
    python scripts/09_tune_on_train.py --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.agd import batch_agd
from src.stats import spearman_with_ci, auroc_with_ci


def check_prereg_committed(cfg) -> bool:
    """Verify prereg.md is committed to git before proceeding."""
    import subprocess
    prereg_path = Path(cfg.paths.analysis) / "prereg.md"
    if not prereg_path.exists():
        logger.error(f"prereg.md not found at {prereg_path}. Write and commit it first!")
        return False
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain", str(prereg_path)],
            capture_output=True, text=True, cwd=str(Path.cwd())
        )
        if result.stdout.strip():
            logger.error(
                "prereg.md has uncommitted changes. Commit it first!\n"
                f"  git add {prereg_path} && git commit -m 'Pre-registration'"
            )
            return False
        logger.info("✓ prereg.md is committed.")
        return True
    except Exception:
        logger.warning("Could not verify git status of prereg.md. Proceeding with caution.")
        return True


def load_pairs_with_behavioral(cfg) -> pd.DataFrame:
    """Load the raw pair metadata (without pre-computed AGD — will recompute per (alpha,k))."""
    import jsonlines
    pairs_dir = Path(cfg.paths.pairs)
    all_rows = []
    for fname in ["regime_A_pairs.jsonl", "regime_B_truncate.jsonl",
                  "regime_B_addmistake.jsonl", "regime_C_hint.jsonl"]:
        fpath = pairs_dir / fname
        if not fpath.exists():
            continue
        with jsonlines.open(fpath) as r:
            for row in r:
                all_rows.append(row)
    return pd.DataFrame(all_rows)


def load_train_ids(cfg) -> set:
    train_path = Path(cfg.paths.data) / "train_ids.txt"
    if not train_path.exists():
        logger.error("train_ids.txt not found. Run data/split.py first.")
        sys.exit(1)
    with open(train_path) as f:
        return set(line.strip() for line in f if line.strip())


def evaluate_on_split(pairs_df: pd.DataFrame, split_ids: set, cfg, alpha: float, k: int):
    """Compute H1 rho and H2 AUROC for given (alpha, k) on a subset of items."""
    behavior_dir = Path(cfg.paths.behavioral)
    graph_dir = Path(cfg.paths.graphs)

    # Filter to split items
    # Use base_item_id if available, else item_id
    def get_base(row):
        return row.get("base_item_id", row["item_id"])

    mask = pairs_df.apply(lambda row: get_base(row) in split_ids, axis=1)
    sub_df = pairs_df[mask].copy()

    if sub_df.empty:
        logger.warning("No pairs found for this split. Check split IDs.")
        return None, None

    # Compute AGD for this (alpha, k)
    agd_df = batch_agd(sub_df, graph_dir=graph_dir, alpha=alpha, k=k,
                       top_edges=cfg.agd.top_edges)

    # ── H1: Spearman(AGD, AOC) on Regime B pairs ──
    h1_rho = None
    aoc_path = behavior_dir / "aoc_lanham.parquet"
    if aoc_path.exists():
        aoc_df = pd.read_parquet(aoc_path)
        b_agd = agd_df[agd_df["regime"].isin(["B", "B_trunc", "B_mistake"])].copy()
        # R3 fix: Regime B has 4 pair variants per base item (3 truncations + 1 mistake).
        # Each shares the same AOC value, so averaging AGD per base_item_id first
        # gives one independent observation per item (correct unit for the correlation).
        b_agd_per_item = (
            b_agd.groupby("base_item_id")["agd"].mean().reset_index()
        )
        # A2 fix: merge on base_item_id → aoc item_id
        merged = b_agd_per_item.merge(aoc_df, left_on="base_item_id", right_on="item_id", how="inner")
        if len(merged) >= 30:
            res = spearman_with_ci(
                merged["agd"].values,
                1 - merged["aoc_composite"].values,  # A3 fix: ρ(AGD, 1-AOC) ≥ 0.30
                n_boot=500,  # smaller for speed during grid search
            )
            h1_rho = res["rho"]

    # ── H2: AUROC(AGD, hint_flip) on Regime C pairs ──
    h2_auc = None
    flip_path = behavior_dir / "turpin_flips.parquet"
    if flip_path.exists():
        flip_df = pd.read_parquet(flip_path)
        c_agd = agd_df[agd_df["regime"] == "C"].copy() if "regime" in agd_df else \
                agd_df[agd_df.get("regime_label", "") == "C"].copy()
        
        # BCa fix: If unfaithful_flip already exists in AGD df, drop it before merging 
        # to avoid the _x/_y suffix issue.
        if "unfaithful_flip" in c_agd.columns:
            c_agd = c_agd.drop(columns=["unfaithful_flip"])

        merged_c = c_agd.merge(
            flip_df[["item_id", "unfaithful_flip"]],
            on="item_id", how="inner"
        )
        if len(merged_c) >= 30 and merged_c["unfaithful_flip"].sum() >= 10:
            res = auroc_with_ci(
                merged_c["agd"].values,
                merged_c["unfaithful_flip"].astype(int).values,
                n_boot=500,
            )
            h2_auc = res["auc"]

    return h1_rho, h2_auc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true",
                        help="Skip prereg check (for debugging only)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if not args.force and not check_prereg_committed(cfg):
        sys.exit(1)

    train_ids = load_train_ids(cfg)
    logger.info(f"Training set: {len(train_ids)} items")

    pairs_df = load_pairs_with_behavioral(cfg)
    logger.info(f"Total pairs: {len(pairs_df)}")

    alpha_grid = list(cfg.ablations.alpha_grid)
    k_grid = list(cfg.ablations.k_grid)
    total = len(alpha_grid) * len(k_grid)

    grid_results = []
    best_auc = -1.0
    best_alpha, best_k = cfg.agd.alpha, cfg.agd.k

    logger.info(f"Grid search over {total} (alpha, k) combinations…")

    with tqdm(total=total, desc="Grid search") as pbar:
        for alpha in alpha_grid:
            for k in k_grid:
                h1_rho, h2_auc = evaluate_on_split(pairs_df, train_ids, cfg, alpha, k)
                entry = {
                    "alpha": alpha, "k": k,
                    "train_h1_rho": h1_rho,
                    "train_h2_auroc": h2_auc,
                }
                grid_results.append(entry)

                if h2_auc is not None and h2_auc > best_auc:
                    best_auc = h2_auc
                    best_alpha, best_k = alpha, k

                pbar.set_postfix({"alpha": alpha, "k": k, "H2": f"{h2_auc:.3f}" if h2_auc else "—"})
                pbar.update(1)

    result = {
        "alpha": best_alpha,
        "k": best_k,
        "train_h1_rho": next(
            (r["train_h1_rho"] for r in grid_results
             if r["alpha"] == best_alpha and r["k"] == best_k), None
        ),
        "train_h2_auroc": best_auc,
        "grid_results": grid_results,
    }

    out_path = Path(cfg.paths.analysis) / "best_hyperparams.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"Best hyperparams: alpha={best_alpha}, k={best_k}")
    logger.info(f"  Train H1 Spearman ρ: {result['train_h1_rho']}")
    logger.info(f"  Train H2 AUROC:      {best_auc:.4f}")
    logger.info(f"Saved → {out_path}")
    logger.info(f"\n>>> NOW commit best_hyperparams.json before running script 10 <<<")


if __name__ == "__main__":
    main()
