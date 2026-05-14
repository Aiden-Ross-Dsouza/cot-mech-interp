"""
scripts/20_graph_census.py
Graph census: per-graph statistics over all ~1978 JSON files.

Produces a single parquet with one row per graph containing structural,
influence, and depth-band statistics. Used as the starting point for
§4.1–§4.3 analyses (graph understanding before metric finalisation).

Reads:  artifacts/graphs/*.json
Writes: artifacts/agd/graph_census.parquet

Usage:
    python scripts/20_graph_census.py --config config.yaml
    python scripts/20_graph_census.py --config config.yaml --graph-dir artifacts/graphs
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph_utils import load_graph
from src.pano import (
    strip_position,
    graph_to_pano_node_set,
    get_n_layers,
    depth_band,
    normalize_graph_influence,
)


def _gini(values: np.ndarray) -> float:
    """Gini coefficient of an array of non-negative values."""
    if len(values) == 0 or values.sum() == 0:
        return 0.0
    values = np.sort(values)
    n = len(values)
    cumvals = np.cumsum(values)
    return float((2 * np.dot(np.arange(1, n + 1), values) - (n + 1) * cumvals[-1]) / (n * cumvals[-1]))


def _entropy(values: np.ndarray) -> float:
    """Shannon entropy of a normalized distribution."""
    total = values.sum()
    if total == 0:
        return 0.0
    p = values / total
    p = p[p > 0]
    return float(-np.dot(p, np.log(p)))


def census_one_graph(graph: Dict[str, Any], k: int = 64) -> Dict[str, Any]:
    """Compute census statistics for a single graph dict."""
    nodes = graph["nodes"]
    n_layers = get_n_layers(graph)

    # Node type counts
    n_feature = sum(1 for n in nodes if n["feature_id"].startswith("L") and "_F" in n["feature_id"])
    n_error   = sum(1 for n in nodes if "_ERR" in n["feature_id"])
    n_token   = sum(1 for n in nodes if n["feature_id"].startswith("P") and "_TOK_" in n["feature_id"])
    n_logit   = sum(1 for n in nodes if n["feature_id"].startswith("LOGIT_"))
    n_total   = len(nodes)

    # Influence stats
    influences = np.array([abs(n["influence"]) for n in nodes], dtype=float)
    total_inf = influences.sum()
    top1_inf  = influences.max() if len(influences) else 0.0
    top10_inf = np.sort(influences)[-10:].sum() if len(influences) >= 10 else total_inf

    # Unique concepts (post-PANO strip)
    concepts = {strip_position(n["feature_id"]) for n in nodes}
    n_unique_concepts = len(concepts)

    # Per-band statistics (R1: depth-as-fraction)
    band_stats: Dict[str, Any] = {}
    for band in ("early", "mid", "late"):
        band_nodes = [n for n in nodes if n["layer"] >= 0 and depth_band(n["layer"], n_layers) == band]
        band_infs = np.array([abs(n["influence"]) for n in band_nodes], dtype=float)
        band_concepts = {strip_position(n["feature_id"]) for n in band_nodes}
        band_stats[f"n_features_{band}"] = len(band_nodes)
        band_stats[f"n_concepts_{band}"] = len(band_concepts)
        band_stats[f"total_inf_{band}"]  = float(band_infs.sum()) if len(band_infs) else 0.0
        band_stats[f"frac_inf_{band}"]   = float(band_infs.sum() / total_inf) if total_inf > 0 else float("nan")

    row: Dict[str, Any] = {
        "item_id":    graph["item_id"],
        "condition":  graph["condition"],
        "model":      graph.get("metadata", {}).get("model", "unknown"),
        "n_nodes_total":   n_total,
        "n_feature_nodes": n_feature,
        "n_error_nodes":   n_error,
        "n_token_nodes":   n_token,
        "n_logit_nodes":   n_logit,
        "n_edges":         graph["n_edges"],
        "n_layers":        n_layers,
        "n_unique_concepts": n_unique_concepts,
        "total_influence": float(total_inf),
        "top1_influence":  float(top1_inf),
        "top10_influence": float(top10_inf),
        "entropy_of_influence": _entropy(influences),
        "gini_of_influence":    _gini(influences),
    }
    row.update(band_stats)

    # Top concept by influence per band
    for band in ("early", "mid", "late"):
        band_nodes = [n for n in nodes if n["layer"] >= 0 and depth_band(n["layer"], n_layers) == band]
        if band_nodes:
            top_node = max(band_nodes, key=lambda n: abs(n["influence"]))
            row[f"top_concept_{band}"] = strip_position(top_node["feature_id"])
        else:
            row[f"top_concept_{band}"] = None

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graph-dir", default=None, help="Override graph directory")
    parser.add_argument("--k", type=int, default=64)
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir = Path(args.graph_dir or cfg.paths.graphs)
    out_path  = Path(cfg.paths.agd) / "graph_census.parquet"

    graph_files = sorted(graph_dir.glob("*.json"))
    logger.info(f"Found {len(graph_files)} graph files in {graph_dir}")

    rows: List[Dict[str, Any]] = []
    for gf in tqdm(graph_files, desc="Graph census"):
        try:
            g = load_graph(gf)
            rows.append(census_one_graph(g, k=args.k))
        except Exception as e:
            logger.warning(f"Failed {gf.name}: {e}")
            rows.append({"item_id": gf.stem, "condition": "ERROR", "error": str(e)})

    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    logger.info(f"Census saved → {out_path}  ({len(df)} rows × {len(df.columns)} cols)")

    # Quick summary
    logger.info(f"\n--- Census summary ---")
    logger.info(f"Models: {df['model'].value_counts().to_dict()}")
    logger.info(f"Conditions: {df['condition'].value_counts().head(10).to_dict()}")
    logger.info(f"Nodes (mean ± std): {df['n_nodes_total'].mean():.0f} ± {df['n_nodes_total'].std():.0f}")
    logger.info(f"Unique concepts (mean): {df['n_unique_concepts'].mean():.0f}")
    logger.info(f"Influence entropy (mean): {df['entropy_of_influence'].mean():.3f}")
    for band in ("early", "mid", "late"):
        logger.info(f"  Influence fraction {band} (mean): {df[f'frac_inf_{band}'].mean():.3f}")


if __name__ == "__main__":
    main()
