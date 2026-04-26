"""
analysis/fit_alpha_k_on_train.py
Importable library for fitting (alpha, k) on the training half.
Can be called from script 09 or used interactively.

Also provides:
  - evaluate_single_config(pairs_df, train_ids, cfg, alpha, k) → {h1, h2}
  - grid_search(pairs_df, train_ids, cfg) → best (alpha, k), full grid
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def evaluate_single_config(
    pairs_df: pd.DataFrame,
    train_ids: set,
    cfg,
    alpha: float,
    k: int,
    n_boot: int = 500,
) -> Dict[str, Optional[float]]:
    """Evaluate a single (alpha, k) config on the training split.

    Returns dict with keys: h1_rho, h2_auroc (None if insufficient data).
    """
    from src.agd import batch_agd
    from src.stats import spearman_with_ci, auroc_with_ci

    behavior_dir = Path(cfg.paths.behavioral)
    graph_dir = cfg.paths.graphs

    # Filter to train items
    def get_base(row):
        return row.get("base_item_id", row["item_id"])

    mask = pairs_df.apply(lambda row: get_base(row) in train_ids, axis=1)
    sub_df = pairs_df[mask].copy()

    if sub_df.empty:
        return {"h1_rho": None, "h2_auroc": None}

    agd_df = batch_agd(sub_df, graph_dir=graph_dir, alpha=alpha, k=k,
                       top_edges=cfg.agd.top_edges)

    # H1
    h1_rho = None
    aoc_path = behavior_dir / "aoc_lanham.parquet"
    if aoc_path.exists():
        aoc_df = pd.read_parquet(aoc_path)
        b_df = agd_df[agd_df.get("fname", agd_df.get("regime_label", pd.Series([""]
                                  * len(agd_df)))).str.contains("B", na=False)]
        merged = b_df.merge(aoc_df, on="item_id", how="inner").dropna(subset=["agd", "aoc_composite"])
        if len(merged) >= 20:
            h1_rho = spearman_with_ci(
                merged["agd"].values, merged["aoc_composite"].values, n_boot=n_boot
            )["rho"]

    # H2
    h2_auroc = None
    flip_path = behavior_dir / "turpin_flips.parquet"
    if flip_path.exists():
        flip_df = pd.read_parquet(flip_path)
        c_df = agd_df[agd_df.get("fname", pd.Series([""] * len(agd_df))).str.contains("C", na=False)]
        merged_c = c_df.merge(flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner")
        merged_c = merged_c.dropna(subset=["agd"])
        if len(merged_c) >= 20 and merged_c["unfaithful_flip"].sum() >= 5:
            h2_auroc = auroc_with_ci(
                merged_c["agd"].values,
                merged_c["unfaithful_flip"].astype(int).values,
                n_boot=n_boot,
            )["auc"]

    return {"h1_rho": h1_rho, "h2_auroc": h2_auroc}


def grid_search(
    pairs_df: pd.DataFrame,
    train_ids: set,
    cfg,
    alpha_grid: Optional[List[float]] = None,
    k_grid: Optional[List[int]] = None,
    primary_metric: str = "h2_auroc",
) -> Tuple[float, int, List[Dict]]:
    """Grid-search (alpha, k) on training half.

    Parameters
    ----------
    primary_metric:
        'h2_auroc' (default) or 'h1_rho' — metric used to select best config.

    Returns
    -------
    (best_alpha, best_k, grid_results)
    """
    if alpha_grid is None:
        alpha_grid = list(cfg.ablations.alpha_grid)
    if k_grid is None:
        k_grid = list(cfg.ablations.k_grid)

    results = []
    best_val = -np.inf
    best_alpha, best_k = cfg.agd.alpha, cfg.agd.k

    for alpha in alpha_grid:
        for k in k_grid:
            scores = evaluate_single_config(pairs_df, train_ids, cfg, alpha, k)
            entry = {"alpha": alpha, "k": k, **scores}
            results.append(entry)

            val = scores.get(primary_metric)
            if val is not None and val > best_val:
                best_val = val
                best_alpha, best_k = alpha, k

            logger.debug(
                f"alpha={alpha}, k={k}: "
                f"H1={scores['h1_rho']:.3f if scores['h1_rho'] else 'N/A'}, "
                f"H2={scores['h2_auroc']:.3f if scores['h2_auroc'] else 'N/A'}"
            )

    logger.info(
        f"Best config: alpha={best_alpha}, k={best_k}, "
        f"{primary_metric}={best_val:.4f}"
    )
    return best_alpha, best_k, results
