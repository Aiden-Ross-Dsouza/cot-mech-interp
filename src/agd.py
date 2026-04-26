"""
src/agd.py
Core AGD (Attribution-Graph Divergence) metric implementation.

Formula:
    AGD(E) = 1 - alpha * J_w(N(G0), N(G1)) - (1 - alpha) * S_e(E(G0), E(G1))

where:
    J_w  = influence-weighted Jaccard over top-k feature nodes
    S_e  = cosine similarity over union-of-top-edges attribution vector
    alpha = 0.5 (default, tunable)

AGD ∈ [0, 1]:
    0 → same mechanism (identical top-k features and edge proportions)
    1 → completely reorganized mechanism
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.graph_utils import (
    graph_to_node_set,
    graph_to_edge_vec,
    union_edge_keys,
    load_graph,
    graph_path,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Component metrics
# ─────────────────────────────────────────────────────────────────────────────

def weighted_jaccard(
    N0: Dict[str, float],
    N1: Dict[str, float],
) -> float:
    """Compute influence-weighted Jaccard similarity J_w.

    J_w = sum_{f in N0∩N1} min(w0(f), w1(f)) / sum_{f in N0∪N1} max(w0(f), w1(f))

    Parameters
    ----------
    N0, N1:
        Dicts mapping feature_id → influence (non-negative floats).

    Returns
    -------
    float in [0, 1]. Returns 1.0 if both sets are empty (degenerate case).
    """
    if not N0 and not N1:
        return 1.0

    all_features = set(N0) | set(N1)
    numerator = 0.0
    denominator = 0.0
    for f in all_features:
        w0 = N0.get(f, 0.0)
        w1 = N1.get(f, 0.0)
        numerator += min(w0, w1)
        denominator += max(w0, w1)

    if denominator == 0.0:
        return 1.0
    return numerator / denominator


def edge_cosine(
    E0: np.ndarray,
    E1: np.ndarray,
) -> float:
    """Cosine similarity between two edge attribution vectors S_e.

    Parameters
    ----------
    E0, E1:
        1-D numpy arrays of equal length representing edge attributions.

    Returns
    -------
    float in [0, 1] (cosine mapped from [-1,1] → [0,1] via (1+cos)/2).
    """
    norm0 = np.linalg.norm(E0)
    norm1 = np.linalg.norm(E1)
    if norm0 == 0.0 and norm1 == 0.0:
        return 1.0
    if norm0 == 0.0 or norm1 == 0.0:
        return 0.0
    cos = float(np.dot(E0, E1) / (norm0 * norm1))
    # Map [-1, 1] → [0, 1] to keep S_e non-negative
    return (1.0 + cos) / 2.0


# ─────────────────────────────────────────────────────────────────────────────
# Main AGD function
# ─────────────────────────────────────────────────────────────────────────────

def compute_agd(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    alpha: float = 0.5,
    k: int = 64,
    top_edges: int = 256,
) -> Dict[str, float]:
    """Compute AGD between two attribution graphs.

    Parameters
    ----------
    graph0, graph1:
        Loaded graph dicts (from load_graph / generate_attribution_graph).
    alpha:
        Weight between J_w (nodes, alpha) and S_e (edges, 1-alpha).
    k:
        Top-k nodes to include in J_w computation.
    top_edges:
        Number of top edges per graph included in the S_e union.

    Returns
    -------
    dict with keys: agd, jw, se, n0 (nodes in G0), n1 (nodes in G1).
    """
    # Node term
    N0 = graph_to_node_set(graph0, k=k)
    N1 = graph_to_node_set(graph1, k=k)
    jw = weighted_jaccard(N0, N1)

    # Edge term — build union key set, then aligned vectors
    union_keys = union_edge_keys(graph0, graph1, top_n=top_edges)
    E0, _ = graph_to_edge_vec(graph0, top_n=top_edges, reference_keys=union_keys)
    E1, _ = graph_to_edge_vec(graph1, top_n=top_edges, reference_keys=union_keys)
    se = edge_cosine(E0, E1)

    agd_val = 1.0 - alpha * jw - (1.0 - alpha) * se
    # Clamp to [0, 1] for numerical safety
    agd_val = float(np.clip(agd_val, 0.0, 1.0))

    return {
        "agd": agd_val,
        "jw": jw,
        "se": se,
        "n0": graph0["n_nodes"],
        "n1": graph1["n_nodes"],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch computation over pairs DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def batch_agd(
    pairs_df: pd.DataFrame,
    graph_dir: str | Path,
    alpha: float = 0.5,
    k: int = 64,
    top_edges: int = 256,
    condition0_col: str = "condition0",
    condition1_col: str = "condition1",
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """Compute AGD for all rows in a pairs DataFrame.

    The DataFrame must have columns: item_id, condition0, condition1
    (and any other metadata columns, which are passed through).

    Returns original DataFrame extended with: agd, jw, se, n0, n1.
    Missing graphs are reported as NaN and logged.
    """
    graph_dir = Path(graph_dir)
    results = []

    for _, row in pairs_df.iterrows():
        iid = row[item_id_col]
        c0 = row[condition0_col]
        c1 = row[condition1_col]
        p0 = graph_path(iid, c0, graph_dir)
        p1 = graph_path(iid, c1, graph_dir)

        if not p0.exists():
            logger.warning(f"Missing graph: {p0}")
            results.append({"agd": np.nan, "jw": np.nan, "se": np.nan,
                            "n0": np.nan, "n1": np.nan})
            continue
        if not p1.exists():
            logger.warning(f"Missing graph: {p1}")
            results.append({"agd": np.nan, "jw": np.nan, "se": np.nan,
                            "n0": np.nan, "n1": np.nan})
            continue

        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            metrics = compute_agd(g0, g1, alpha=alpha, k=k, top_edges=top_edges)
            results.append(metrics)
        except Exception as e:
            logger.error(f"AGD computation failed for {iid}/{c0}↔{c1}: {e}")
            results.append({"agd": np.nan, "jw": np.nan, "se": np.nan,
                            "n0": np.nan, "n1": np.nan})

    metrics_df = pd.DataFrame(results, index=pairs_df.index)
    return pd.concat([pairs_df, metrics_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# Ablation helpers
# ─────────────────────────────────────────────────────────────────────────────

def agd_alpha_sweep(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    alpha_grid: List[float],
    k: int = 64,
    top_edges: int = 256,
) -> Dict[float, float]:
    """Return {alpha: agd_value} for a sweep over alpha values.
    Computes J_w and S_e once and reuses them.
    """
    N0 = graph_to_node_set(graph0, k=k)
    N1 = graph_to_node_set(graph1, k=k)
    jw = weighted_jaccard(N0, N1)

    union_keys = union_edge_keys(graph0, graph1, top_n=top_edges)
    E0, _ = graph_to_edge_vec(graph0, top_n=top_edges, reference_keys=union_keys)
    E1, _ = graph_to_edge_vec(graph1, top_n=top_edges, reference_keys=union_keys)
    se = edge_cosine(E0, E1)

    return {
        a: float(np.clip(1.0 - a * jw - (1.0 - a) * se, 0.0, 1.0))
        for a in alpha_grid
    }


def agd_k_sweep(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k_grid: List[int],
    alpha: float = 0.5,
    top_edges: int = 256,
) -> Dict[int, float]:
    """Return {k: agd_value} for a sweep over k values."""
    union_keys = union_edge_keys(graph0, graph1, top_n=top_edges)
    E0, _ = graph_to_edge_vec(graph0, top_n=top_edges, reference_keys=union_keys)
    E1, _ = graph_to_edge_vec(graph1, top_n=top_edges, reference_keys=union_keys)
    se = edge_cosine(E0, E1)

    results = {}
    for k in k_grid:
        N0 = graph_to_node_set(graph0, k=k)
        N1 = graph_to_node_set(graph1, k=k)
        jw = weighted_jaccard(N0, N1)
        results[k] = float(np.clip(1.0 - alpha * jw - (1.0 - alpha) * se, 0.0, 1.0))
    return results


def agd_layer_band(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    layer_range: Tuple[int, int],
    alpha: float = 0.5,
    k: int = 64,
    top_edges: int = 256,
) -> float:
    """Compute AGD restricted to nodes/edges in a layer band [lo, hi]."""
    lo, hi = layer_range

    def filter_graph(g: Dict[str, Any]) -> Dict[str, Any]:
        nodes = [n for n in g["nodes"] if lo <= n.get("layer", -1) <= hi]
        node_ids = {n["feature_id"] for n in nodes}
        edges = [e for e in g["edges"]
                 if e["src"] in node_ids and e["dst"] in node_ids]
        return {**g, "nodes": nodes, "edges": edges,
                "n_nodes": len(nodes), "n_edges": len(edges)}

    g0_filtered = filter_graph(graph0)
    g1_filtered = filter_graph(graph1)
    result = compute_agd(g0_filtered, g1_filtered, alpha=alpha, k=k, top_edges=top_edges)
    return result["agd"]
