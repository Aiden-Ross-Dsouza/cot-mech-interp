"""
src/pano.py
Position-Agnostic Node Overlap (PANO) and Differential AGD (delta-AGD).

Why PANO:
    Attribution graph feature IDs encode token position:
        "L3_P12_F1047"  = Layer 3, Position 12, Feature 1047
    Two CoT variants of different lengths place the same conceptual feature
    at different token positions, so the standard Jaccard treats them as
    entirely distinct nodes even when the mechanism is identical.
    PANO strips the position component before comparison, collapsing
    "L3_P12_F1047" and "L3_P24_F1047" to the same concept "L3_F1047".

PANO formula:
    concept(feature_id) = strip_position(feature_id)
    influence(concept) = max over positions of influence(L_P_F)
    PANO_sim(G0, G1) = weighted_jaccard(top-k concepts from G0, top-k from G1)
    PANO_div(G0, G1) = 1 - PANO_sim(G0, G1)

delta-AGD formula:
    delta_AGD(item) = PANO_div(G_perturb, G_clean) - PANO_div(G_paraphrase, G_clean)

    Positive delta-AGD: the perturbation (mistake/hint) caused MORE mechanism
    shift beyond what a neutral paraphrase causes. This is evidence that the
    CoT perturbation was causally engaging the model's computation (faithful).
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.graph_utils import load_graph, graph_path
from src.agd import weighted_jaccard

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Position stripping
# ─────────────────────────────────────────────────────────────────────────────

_FEATURE_RE = re.compile(r"^(L\d+)_P\d+_(F\d+)$")
_ERROR_RE   = re.compile(r"^(L\d+)_P\d+_(ERR)$")
_TOKEN_RE   = re.compile(r"^P\d+_(TOK_.+)$")


def strip_position(feature_id: str) -> str:
    """Convert position-specific feature ID to position-agnostic concept ID.

    Mapping:
        "L3_P12_F1047"  → "L3_F1047"   (transcoder feature)
        "L3_P12_ERR"    → "L3_ERR"      (error node)
        "P12_TOK_1023"  → "TOK_1023"    (input token by vocab ID)
        "LOGIT_42"      → "LOGIT_42"    (logit node — position-free already)
    """
    m = _FEATURE_RE.match(feature_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = _ERROR_RE.match(feature_id)
    if m:
        return f"{m.group(1)}_{m.group(2)}"
    m = _TOKEN_RE.match(feature_id)
    if m:
        return m.group(1)
    # LOGIT_* and any unknown format: return as-is
    return feature_id


# ─────────────────────────────────────────────────────────────────────────────
# PANO node set construction
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_pano_node_set(graph: Dict[str, Any], k: int) -> Dict[str, float]:
    """Return top-k position-agnostic concepts by max influence.

    For each unique concept (layer, feature_index), take the maximum
    influence across all token positions where that concept fires.
    Then return the top-k concepts by this max-influence score.

    Parameters
    ----------
    graph:
        Loaded graph dict (from load_graph).
    k:
        Number of top concepts to keep.

    Returns
    -------
    Dict mapping concept_id → max_influence (non-negative floats).
    """
    concept_influence: Dict[str, float] = {}
    for node in graph["nodes"]:
        concept = strip_position(node["feature_id"])
        inf = abs(node["influence"])
        if concept not in concept_influence or inf > concept_influence[concept]:
            concept_influence[concept] = inf

    # Return top-k by influence
    sorted_concepts = sorted(concept_influence.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_concepts[:k])


# ─────────────────────────────────────────────────────────────────────────────
# PANO similarity and divergence
# ─────────────────────────────────────────────────────────────────────────────

def pano_similarity(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k: int = 64,
) -> float:
    """PANO similarity in [0, 1]. Higher = more mechanism overlap."""
    N0 = graph_to_pano_node_set(graph0, k=k)
    N1 = graph_to_pano_node_set(graph1, k=k)
    return weighted_jaccard(N0, N1)


def compute_pano(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k: int = 64,
) -> Dict[str, float]:
    """Compute PANO divergence and its components for one graph pair.

    Returns
    -------
    dict with keys: pano_div, pano_sim, n_concepts_0, n_concepts_1, n_shared
    """
    N0 = graph_to_pano_node_set(graph0, k=k)
    N1 = graph_to_pano_node_set(graph1, k=k)
    sim = weighted_jaccard(N0, N1)
    shared = len(set(N0) & set(N1))
    return {
        "pano_sim": float(sim),
        "pano_div": float(1.0 - sim),
        "n_concepts_0": len(N0),
        "n_concepts_1": len(N1),
        "n_shared_concepts": shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Batch PANO over a pairs DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def batch_pano(
    pairs_df: pd.DataFrame,
    graph_dir: str | Path,
    k: int = 64,
    item_id_col: str = "item_id",
    condition0_col: str = "condition0",
    condition1_col: str = "condition1",
) -> pd.DataFrame:
    """Compute PANO for all rows in a pairs DataFrame.

    Returns original DataFrame extended with:
        pano_sim, pano_div, n_concepts_0, n_concepts_1, n_shared_concepts
    """
    graph_dir = Path(graph_dir)
    results = []

    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="PANO"):
        iid = row[item_id_col]
        c0 = row[condition0_col]
        c1 = row[condition1_col]
        p0 = graph_dir / f"{iid}_{c0}.json"
        p1 = graph_dir / f"{iid}_{c1}.json"

        if not p0.exists() or not p1.exists():
            missing = p0 if not p0.exists() else p1
            logger.warning(f"Missing graph: {missing.name}")
            results.append({
                "pano_sim": np.nan, "pano_div": np.nan,
                "n_concepts_0": np.nan, "n_concepts_1": np.nan,
                "n_shared_concepts": np.nan,
            })
            continue

        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            metrics = compute_pano(g0, g1, k=k)
            results.append(metrics)
        except Exception as e:
            logger.error(f"PANO failed for {iid}/{c0}<->{c1}: {e}")
            results.append({
                "pano_sim": np.nan, "pano_div": np.nan,
                "n_concepts_0": np.nan, "n_concepts_1": np.nan,
                "n_shared_concepts": np.nan,
            })

    metrics_df = pd.DataFrame(results, index=pairs_df.index)
    return pd.concat([pairs_df, metrics_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# delta-AGD computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_delta_agd(
    pano_df: pd.DataFrame,
    regime_col: str = "regime_label",
    base_id_col: str = "base_item_id",
    item_id_col: str = "item_id",
) -> pd.DataFrame:
    """Compute delta-AGD for Regime B and C rows, using Regime A as baseline.

    delta_AGD(item, regime) = PANO_div(regime) - PANO_div(A)

    Positive delta_AGD: the perturbation causes MORE mechanism shift above
    the neutral paraphrase baseline. Interpreted as higher CoT engagement
    (faithfulness).

    Parameters
    ----------
    pano_df:
        DataFrame with pano_div column and regime_label, item_id, base_item_id.

    Returns
    -------
    DataFrame with added column 'delta_agd'. Rows without a matching
    Regime A baseline get NaN.
    """
    pano_df = pano_df.copy()

    # Build baseline: Regime A pano_div indexed by item_id (which IS the base item)
    regime_a = pano_df[pano_df[regime_col] == "A"][[item_id_col, "pano_div"]].copy()
    regime_a = regime_a.dropna(subset=["pano_div"])
    baseline_map = regime_a.set_index(item_id_col)["pano_div"].to_dict()

    def get_baseline(row):
        if row[regime_col] == "A":
            return np.nan  # delta-AGD undefined for the baseline itself
        # For B/C rows, base_item_id links back to the Regime A item
        base_id = row.get(base_id_col)
        if base_id is None or pd.isna(base_id):
            return np.nan
        return baseline_map.get(base_id, np.nan)

    pano_df["pano_div_A_baseline"] = pano_df.apply(get_baseline, axis=1)
    pano_df["delta_agd"] = pano_df["pano_div"] - pano_df["pano_div_A_baseline"]
    return pano_df
