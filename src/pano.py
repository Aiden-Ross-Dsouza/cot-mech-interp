"""
src/pano.py
Position-Agnostic Node Overlap (PANO), Differential AGD (delta-AGD),
and the v2.1 metric family: GRACE-T band decomposition, ED, HTIR.

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

v2.1 model-agnostic metrics (R1–R6 from research_plan_v2.md):
    R1: depth as fraction — layer_index / n_layers → early/mid/late bands
    R2: position windows as token-fraction of CoT length
    R3: top-k as min(64, ceil(f * n_unique_concepts))
    R4: backbone/item-specific re-derived per model corpus
    R5: influence normalized within graph (fraction of total)
    R6: every metric returns scalar in [0,1]

    compute_pano_bands(G0, G1, k=64)
        → GRACE-T per depth band {early, mid, late}

    compute_ed(G0, G1, mistake_token_pos, cot_len_tokens, f=0.10)
        → ED ∈ [0,1]: fraction of attribution shift local to mistake position

    compute_htir(G_biased, hint_token_positions)
        → HTIR ∈ [0,1]: fraction of influence through hint-token positions
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


# ─────────────────────────────────────────────────────────────────────────────
# v2.1 model-agnostic helpers (R1–R6)
# ─────────────────────────────────────────────────────────────────────────────

_POS_RE = re.compile(r"_P(\d+)_|^P(\d+)_")


def get_node_position(feature_id: str) -> Optional[int]:
    """Extract token position from a feature_id string, or None for LOGIT nodes."""
    m = _POS_RE.search(feature_id)
    if m:
        return int(m.group(1) or m.group(2))
    return None


def get_n_layers(graph: Dict[str, Any]) -> int:
    """Infer number of model layers from graph nodes (model-agnostic, R1)."""
    layers = [n["layer"] for n in graph["nodes"] if n["layer"] >= 0]
    return max(layers) + 1 if layers else 26


def depth_band(layer: int, n_layers: int) -> str:
    """Map layer index to 'early'/'mid'/'late' via depth fraction (R1)."""
    d = layer / n_layers
    if d < 1 / 3:
        return "early"
    elif d < 2 / 3:
        return "mid"
    else:
        return "late"


def normalize_graph_influence(graph: Dict[str, Any]) -> Dict[str, float]:
    """Normalized influence per node as fraction of total (R5)."""
    total = sum(abs(n["influence"]) for n in graph["nodes"])
    if total == 0:
        return {n["feature_id"]: 0.0 for n in graph["nodes"]}
    return {n["feature_id"]: abs(n["influence"]) / total for n in graph["nodes"]}


def get_position_influence(
    graph: Dict[str, Any],
    normalized: bool = True,
) -> Dict[int, float]:
    """Sum (normalized) influence over all nodes at each token position.

    Returns Dict[position → total_influence].
    Nodes without a position (LOGIT_*) are excluded.
    """
    inf_map = normalize_graph_influence(graph) if normalized else {
        n["feature_id"]: abs(n["influence"]) for n in graph["nodes"]
    }
    pos_inf: Dict[int, float] = {}
    for node in graph["nodes"]:
        pos = get_node_position(node["feature_id"])
        if pos is None:
            continue
        pos_inf[pos] = pos_inf.get(pos, 0.0) + inf_map[node["feature_id"]]
    return pos_inf


def graph_to_pano_node_set_band(
    graph: Dict[str, Any],
    k: int,
    band: str,
    n_layers: Optional[int] = None,
) -> Dict[str, float]:
    """Top-k position-agnostic concepts restricted to one depth band (R1).

    Parameters
    ----------
    graph:
        Loaded graph dict.
    k:
        Max concepts to keep (further limited by available concepts in band).
    band:
        One of 'early', 'mid', 'late'.
    n_layers:
        Total layer count for depth fraction; inferred from graph if None.
    """
    if n_layers is None:
        n_layers = get_n_layers(graph)
    concept_influence: Dict[str, float] = {}
    for node in graph["nodes"]:
        layer = node["layer"]
        if layer < 0:  # token/logit nodes have layer=-1
            continue
        if depth_band(layer, n_layers) != band:
            continue
        concept = strip_position(node["feature_id"])
        inf = abs(node["influence"])
        if concept not in concept_influence or inf > concept_influence[concept]:
            concept_influence[concept] = inf
    sorted_concepts = sorted(concept_influence.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_concepts[:k])


def compute_pano_bands(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k: int = 64,
) -> Dict[str, float]:
    """GRACE-T per depth band — early/mid/late (R1-compliant).

    Returns dict with keys: pano_div_early, pano_div_mid, pano_div_late,
    plus n_concepts_{band}_0 and n_concepts_{band}_1 for diagnostics.
    """
    n_layers = max(get_n_layers(graph0), get_n_layers(graph1))
    result: Dict[str, float] = {}
    for band in ("early", "mid", "late"):
        N0 = graph_to_pano_node_set_band(graph0, k, band, n_layers)
        N1 = graph_to_pano_node_set_band(graph1, k, band, n_layers)
        result[f"n_concepts_{band}_0"] = len(N0)
        result[f"n_concepts_{band}_1"] = len(N1)
        if not N0 and not N1:
            result[f"pano_div_{band}"] = float("nan")
        else:
            result[f"pano_div_{band}"] = float(1.0 - weighted_jaccard(N0, N1))
    return result


# ─────────────────────────────────────────────────────────────────────────────
# ED — Error Detection Localisation (v2.1 §2.2)
# ─────────────────────────────────────────────────────────────────────────────

def compute_ed(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    mistake_token_pos: int,
    cot_len_tokens: int,
    f: float = 0.10,
) -> float:
    """Error Detection Localisation metric (ED). R2-compliant.

    Measures what fraction of the total attribution shift between graph0 (clean)
    and graph1 (mistake-injected) is concentrated in a window around the
    mistake position.

    Parameters
    ----------
    graph0:
        Clean graph (condition0).
    graph1:
        Mistake-injected graph (condition1).
    mistake_token_pos:
        Token position where the mistake was inserted in graph1's prompt.
    cot_len_tokens:
        Length of the CoT portion in tokens (for R2: window = f * cot_len).
    f:
        Window fraction of cot_len (default 0.10 per R2).

    Returns
    -------
    ED ∈ [0, 1]: fraction of |attribution shift| local to mistake.
    Returns NaN if total shift is zero.
    """
    half_w = max(1, int(np.ceil(f * cot_len_tokens / 2)))
    w_start = max(0, mistake_token_pos - half_w)
    w_end = mistake_token_pos + half_w  # inclusive upper bound

    inf0 = get_position_influence(graph0, normalized=True)
    inf1 = get_position_influence(graph1, normalized=True)

    all_positions = set(inf0) | set(inf1)
    total_shift = 0.0
    local_shift = 0.0
    for pos in all_positions:
        delta = abs(inf1.get(pos, 0.0) - inf0.get(pos, 0.0))
        total_shift += delta
        if w_start <= pos <= w_end:
            local_shift += delta

    if total_shift == 0.0:
        return float("nan")
    return float(local_shift / total_shift)


# ─────────────────────────────────────────────────────────────────────────────
# HTIR — Hint-Token Influence Ratio (v2.1 §2.3)
# ─────────────────────────────────────────────────────────────────────────────

def compute_htir(
    graph_biased: Dict[str, Any],
    hint_token_positions: List[int],
) -> float:
    """Hint-Token Influence Ratio (HTIR). R5-compliant.

    Measures the fraction of total normalized influence-on-target that passes
    through nodes at the hint-token positions.

    Parameters
    ----------
    graph_biased:
        Attribution graph for the hint-biased prompt (condition1 in Regime C).
    hint_token_positions:
        List of token positions corresponding to the hint phrase. Computed
        programmatically from the tokenizer (do not hard-code).

    Returns
    -------
    HTIR ∈ [0, 1]. Returns NaN if total influence is zero.
    """
    if not hint_token_positions:
        return float("nan")

    hint_set = set(hint_token_positions)
    norm_inf = normalize_graph_influence(graph_biased)

    total_inf = sum(norm_inf.values())
    hint_inf = 0.0
    for node in graph_biased["nodes"]:
        pos = get_node_position(node["feature_id"])
        if pos is not None and pos in hint_set:
            hint_inf += norm_inf[node["feature_id"]]

    if total_inf == 0.0:
        return float("nan")
    return float(hint_inf / total_inf)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone / item-specific concept split helpers (v2.1 §4.2)
# ─────────────────────────────────────────────────────────────────────────────

def build_concept_frequency_map(
    graph_dicts: List[Dict[str, Any]],
    k: int = 64,
) -> Dict[str, float]:
    """Compute fraction of graphs in which each concept appears in top-k.

    Returns Dict[concept_id → frequency (0–1)].
    """
    n = len(graph_dicts)
    if n == 0:
        return {}
    counts: Dict[str, int] = {}
    for g in graph_dicts:
        top_k = graph_to_pano_node_set(g, k=k)
        for concept in top_k:
            counts[concept] = counts.get(concept, 0) + 1
    return {c: cnt / n for c, cnt in counts.items()}


def split_concepts_by_frequency(
    freq_map: Dict[str, float],
    backbone_threshold: float = 0.50,
    specific_threshold: float = 0.10,
) -> Tuple[set, set, set]:
    """Partition concepts into backbone / item-specific / middle.

    Returns
    -------
    backbone: concepts appearing in > backbone_threshold fraction of graphs
    item_specific: concepts appearing in < specific_threshold fraction of graphs
    middle: everything else
    """
    backbone = {c for c, f in freq_map.items() if f > backbone_threshold}
    item_specific = {c for c, f in freq_map.items() if f < specific_threshold}
    middle = set(freq_map) - backbone - item_specific
    return backbone, item_specific, middle


def graph_to_pano_node_set_filtered(
    graph: Dict[str, Any],
    k: int,
    allowed_concepts: set,
) -> Dict[str, float]:
    """Top-k concepts restricted to an allowed set (backbone or item-specific)."""
    concept_influence: Dict[str, float] = {}
    for node in graph["nodes"]:
        concept = strip_position(node["feature_id"])
        if concept not in allowed_concepts:
            continue
        inf = abs(node["influence"])
        if concept not in concept_influence or inf > concept_influence[concept]:
            concept_influence[concept] = inf
    sorted_concepts = sorted(concept_influence.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_concepts[:k])


def compute_pano_filtered(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    k: int,
    allowed_concepts: set,
) -> float:
    """GRACE-T restricted to allowed_concepts (backbone or item-specific)."""
    N0 = graph_to_pano_node_set_filtered(graph0, k, allowed_concepts)
    N1 = graph_to_pano_node_set_filtered(graph1, k, allowed_concepts)
    if not N0 and not N1:
        return float("nan")
    return float(1.0 - weighted_jaccard(N0, N1))


# ─────────────────────────────────────────────────────────────────────────────
# GRM — Global Reorganization Magnitude (v2.2 §2.5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_grm(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
) -> float:
    """Global Reorganization Magnitude (GRM). R5-compliant, [0, 1].

    The L1 norm of the position-resolved normalized-influence difference,
    divided by 2 to bound to [0, 1].

    GRM(G0, G1) = (1/2) * Σ_p |inf_norm(p, G1) - inf_norm(p, G0)|

    This is the denominator of ED, surfaced as a standalone metric.
    GRM near 1 = mechanisms at entirely different positions; GRM near 0 = stable.

    Hypothesis (§2.5): if ρ(GRM, aoc_mistake) ≥ 0.20 on Gemma B_mistake
    pairs, GRM becomes the paper's 4th metric; otherwise it is dropped and
    only the ED-locality failure result is reported.
    """
    inf0 = get_position_influence(graph0, normalized=True)
    inf1 = get_position_influence(graph1, normalized=True)
    all_positions = set(inf0) | set(inf1)
    l1 = sum(abs(inf1.get(p, 0.0) - inf0.get(p, 0.0)) for p in all_positions)
    # Maximum possible L1 distance for two normalized distributions = 2.0
    return float(l1 / 2.0)


def get_layer_shift_profile(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
) -> Dict[str, float]:
    """Compute fraction of total attribution shift living in each depth band (R1).

    Returns dict with keys 'early', 'mid', 'late', 'other', each a fraction
    of the total L1 shift across all nodes (not position-resolved but
    layer-resolved — useful for characterizing where diffuse mistake shifts live).
    """
    n_layers = max(get_n_layers(graph0), get_n_layers(graph1))
    norm0 = normalize_graph_influence(graph0)
    norm1 = normalize_graph_influence(graph1)

    # Aggregate normalized influence by (layer, concept) to avoid position drift
    def layer_concept_inf(graph: Dict[str, Any], norm: Dict[str, float]) -> Dict[Tuple[int, str], float]:
        lc: Dict[Tuple[int, str], float] = {}
        for node in graph["nodes"]:
            layer = node["layer"]
            concept = strip_position(node["feature_id"])
            key = (layer, concept)
            lc[key] = lc.get(key, 0.0) + norm[node["feature_id"]]
        return lc

    lc0 = layer_concept_inf(graph0, norm0)
    lc1 = layer_concept_inf(graph1, norm1)
    all_keys = set(lc0) | set(lc1)

    band_shift = {"early": 0.0, "mid": 0.0, "late": 0.0, "other": 0.0}
    total_shift = 0.0
    for (layer, concept) in all_keys:
        delta = abs(lc1.get((layer, concept), 0.0) - lc0.get((layer, concept), 0.0))
        total_shift += delta
        if layer < 0:
            band_shift["other"] += delta
        else:
            band_shift[depth_band(layer, n_layers)] += delta

    if total_shift == 0.0:
        return {k: float("nan") for k in band_shift}
    return {k: float(v / total_shift) for k, v in band_shift.items()}
