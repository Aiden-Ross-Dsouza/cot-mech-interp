"""
src/graph_utils.py
Attribution graph generation, I/O, and feature-extraction utilities.

The core function `generate_attribution_graph` wraps circuit-tracer's
AttributionGraph API. Everything downstream (AGD computation, ablations)
operates on plain Python dicts + numpy arrays loaded from JSON.

Graph JSON schema
─────────────────
{
  "item_id":   str,
  "condition": str,            # "clean" | "paraphrase" | "truncate_25" | ...
  "prompt":    str,
  "target_token": str,
  "target_token_id": int,
  "pruning_threshold": float,
  "n_nodes": int,
  "n_edges": int,
  "nodes": [
    {"feature_id": str, "layer": int, "influence": float, "label": str},
    ...
  ],
  "edges": [
    {"src": str, "dst": str, "weight": float},
    ...
  ],
  "metadata": {
    "model": str,
    "transcoder_type": str,
    "k": int,
    "timestamp": str
  }
}
"""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.config import Config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Graph generation (circuit-tracer wrapper)
# ─────────────────────────────────────────────────────────────────────────────

def generate_attribution_graph(
    model,                     # loaded HF model (Gemma-2-2B-it)
    tokenizer,
    prompt: str,
    target_token: str,
    cfg: Config,
    item_id: str = "unknown",
    condition: str = "clean",
) -> Dict[str, Any]:
    """Generate a pruned attribution graph for (prompt, target_token).

    Uses circuit-tracer's AttributionGraph API with Gemma Scope PLT transcoders.

    Parameters
    ----------
    model, tokenizer:
        Loaded Gemma-2-2B-it model and tokenizer.
    prompt:
        Full input prompt (system + CoT), NOT including the target token.
    target_token:
        The answer token whose logit we attribute (e.g. "A", "B", "5").
    cfg:
        Config object.
    item_id, condition:
        Metadata stored in the output JSON.

    Returns
    -------
    graph_dict:
        Dict matching the JSON schema described in the module docstring.
    """
    try:
        from circuit_tracer.attribution.attribute import attribute
        from circuit_tracer.graph import prune_graph
        from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
        from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    except ImportError:
        raise ImportError(
            "circuit-tracer (decoderesearch version) not installed correctly. "
            "Run: pip install git+https://github.com/decoderesearch/circuit-tracer.git"
        )

    # Tokenize target to get its ID
    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) != 1:
        logger.warning(
            f"Target token '{target_token}' encodes to {len(target_ids)} tokens; "
            f"using first: {target_ids[0]}"
        )
    target_token_id = target_ids[0]

    logger.debug(f"[{item_id}/{condition}] Building attribution graph for '{target_token}'…")

    # 1. Load transcoders (PLT/CLT)
    # The new API uses load_transcoder_from_hub which returns (transcoder_set, metadata)
    transcoder_set, _ = load_transcoder_from_hub(
        cfg.transcoders.hf_repo,
        device=torch.device(cfg.models.main.device),
        dtype=torch.float16, # Gemma Scope usually in fp16/bf16
    )

    # 2. Wrap model in TransformerLensReplacementModel
    # Note: load_main_model returns (AutoModelForCausalLM, AutoTokenizer)
    # Circuit-tracer attribute() prefers HookedTransformer for TL backend.
    # We use from_pretrained_and_transcoders if we want to load via TL,
    # but since we already have the model loaded, we'll try to wrap or reload.
    # In practice, circuit-tracer works best if it manages the model loading.
    # For now, we'll initialize the replacement model using the name.
    tl_model = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        cfg.models.main.name,
        transcoder_set,
        device=cfg.models.main.device,
        dtype=torch.float16,
    )

    # 3. Compute dense attribution graph
    # attribution_targets=None auto-selects top logits
    ag = attribute(
        prompt=prompt,
        model=tl_model,
        attribution_targets=[target_token],
        max_feature_nodes=cfg.agd.k * 2, # Buffer for pruning
        verbose=False,
    )

    # 4. Prune graph
    prune_result = prune_graph(
        ag,
        node_threshold=cfg.agd.pruning_threshold,
        edge_threshold=0.98,
    )

    # 5. Extract nodes and edges from pruned result
    n_features = len(ag.selected_features)
    n_pos = ag.n_pos
    n_layers = ag.cfg.n_layers
    n_logits = len(ag.logit_targets)

    # Indices in adjacency matrix: [features, errors, tokens, logits]
    error_start = n_features
    token_start = error_start + n_layers * n_pos
    logit_start = token_start + n_pos

    node_mask = prune_result.node_mask
    edge_mask = prune_result.edge_mask

    nodes = []
    # Map feature nodes
    for i in range(n_features):
        if node_mask[i]:
            active_idx = ag.selected_features[i].item()
            layer, pos, feat_idx = ag.active_features[active_idx].tolist()
            nodes.append({
                "feature_id": f"L{layer}_P{pos}_F{feat_idx}",
                "layer": int(layer),
                "influence": float(ag.activation_values[i]), # Using activation as fallback for weight
                "label": f"Feature {feat_idx} at L{layer} P{pos}",
            })

    # Map error nodes
    for i in range(error_start, token_start):
        if node_mask[i]:
            layer = (i - error_start) // n_pos
            pos = (i - error_start) % n_pos
            nodes.append({
                "feature_id": f"L{layer}_P{pos}_ERR",
                "layer": int(layer),
                "influence": 0.0,
                "label": f"Error at L{layer} P{pos}",
            })

    # Map token nodes
    for i in range(token_start, logit_start):
        if node_mask[i]:
            pos = i - token_start
            token_id = ag.input_tokens[pos].item()
            token_str = tl_model.tokenizer.decode(token_id)
            nodes.append({
                "feature_id": f"P{pos}_TOK_{token_id}",
                "layer": -1,
                "influence": 0.0,
                "label": f"Token: {token_str}",
            })

    # Map logit nodes
    for i in range(logit_start, logit_start + n_logits):
        if node_mask[i]:
            target = ag.logit_targets[i - logit_start]
            nodes.append({
                "feature_id": f"LOGIT_{target.vocab_idx}",
                "layer": n_layers,
                "influence": float(ag.logit_probabilities[i - logit_start]),
                "label": f"Target: {target.token_str}",
            })

    # Map edges
    edges = []
    rows, cols = edge_mask.nonzero(as_tuple=True)
    for r, c in zip(rows.tolist(), cols.tolist()):
        # Adjacency matrix: rows are targets, cols are sources
        # We need to map row/col index to feature_id
        def get_id(idx):
            if idx < n_features:
                active_idx = ag.selected_features[idx].item()
                layer, pos, f_idx = ag.active_features[active_idx].tolist()
                return f"L{layer}_P{pos}_F{f_idx}"
            elif idx < token_start:
                l = (idx - error_start) // n_pos
                p = (idx - error_start) % n_pos
                return f"L{l}_P{p}_ERR"
            elif idx < logit_start:
                p = idx - token_start
                tid = ag.input_tokens[p].item()
                return f"P{p}_TOK_{tid}"
            else:
                target_idx = idx - logit_start
                if target_idx >= 0 and target_idx < len(ag.logit_targets):
                    tid = ag.logit_targets[target_idx].vocab_idx
                else:
                    tid = "UNK"
                return f"LOGIT_{tid}"

        edges.append({
            "src": get_id(c),
            "dst": get_id(r),
            "weight": float(ag.adjacency_matrix[r, c]),
        })

    graph_dict = {
        "item_id": item_id,
        "condition": condition,
        "prompt": prompt,
        "target_token": target_token,
        "target_token_id": target_token_id,
        "pruning_threshold": cfg.agd.pruning_threshold,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "model": cfg.models.main.name,
            "transcoder_type": cfg.transcoders.type,
            "k": cfg.agd.k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    return graph_dict


# ─────────────────────────────────────────────────────────────────────────────
# I/O
# ─────────────────────────────────────────────────────────────────────────────

def graph_path(item_id: str, condition: str, graph_dir: str | Path) -> Path:
    """Canonical path for a graph JSON file."""
    return Path(graph_dir) / f"{item_id}_{condition}.json"


def save_graph(graph_dict: Dict[str, Any], path: str | Path) -> None:
    """Serialize graph to JSON (atomic write via tmp file)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp.json")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(graph_dict, f)
    os.replace(tmp_path, path)  # atomic on most filesystems


def load_graph(path: str | Path) -> Dict[str, Any]:
    """Load a graph JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def graph_exists(item_id: str, condition: str, graph_dir: str | Path) -> bool:
    return graph_path(item_id, condition, graph_dir).exists()


# ─────────────────────────────────────────────────────────────────────────────
# Feature extraction for AGD
# ─────────────────────────────────────────────────────────────────────────────

def graph_to_node_set(
    graph: Dict[str, Any],
    k: int,
) -> Dict[str, float]:
    """Return top-k nodes by influence as {feature_id: influence}.

    If the graph has fewer than k nodes, all nodes are returned (no padding
    done here — padding is handled in weighted_jaccard in agd.py).
    """
    nodes = graph["nodes"]
    # Sort descending by influence
    sorted_nodes = sorted(nodes, key=lambda n: abs(n["influence"]), reverse=True)
    top_k = sorted_nodes[:k]
    return {n["feature_id"]: abs(n["influence"]) for n in top_k}


def graph_to_edge_vec(
    graph: Dict[str, Any],
    top_n: int = 256,
    reference_keys: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[np.ndarray, List[Tuple[str, str]]]:
    """Return edge attribution vector for cosine similarity (S_e).

    Parameters
    ----------
    graph:
        Loaded graph dict.
    top_n:
        Number of top edges (by |weight|) to include from this graph when
        `reference_keys` is None.
    reference_keys:
        If provided, build vector using exactly these (src, dst) keys
        (set to 0 if edge is absent). Used to align two graphs to the same
        vector space before computing cosine.

    Returns
    -------
    vec:
        numpy array of edge weights.
    keys:
        List of (src, dst) tuples corresponding to each vector element.
    """
    edge_dict = {(e["src"], e["dst"]): e["weight"] for e in graph["edges"]}

    if reference_keys is None:
        # Pick top_n edges by |weight|
        sorted_edges = sorted(edge_dict.items(), key=lambda x: abs(x[1]), reverse=True)
        keys = [k for k, _ in sorted_edges[:top_n]]
    else:
        keys = reference_keys

    vec = np.array([edge_dict.get(k, 0.0) for k in keys], dtype=np.float32)
    return vec, keys


def union_edge_keys(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
    top_n: int = 256,
) -> List[Tuple[str, str]]:
    """Return union of top-n edges from both graphs, for aligned S_e computation."""
    edge_dict0 = {(e["src"], e["dst"]): abs(e["weight"]) for e in graph0["edges"]}
    edge_dict1 = {(e["src"], e["dst"]): abs(e["weight"]) for e in graph1["edges"]}

    top0 = sorted(edge_dict0, key=edge_dict0.get, reverse=True)[:top_n]
    top1 = sorted(edge_dict1, key=edge_dict1.get, reverse=True)[:top_n]

    # Union, preserving order (top0 first, then any new keys from top1)
    seen = set(top0)
    union = list(top0)
    for k in top1:
        if k not in seen:
            union.append(k)
            seen.add(k)
    return union
