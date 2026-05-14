"""
scripts/04b_generate_graphs_llama.py
Attribution graph generation for Llama-3.2-1B-Instruct (cross-model campaign).

Mirrors 04_generate_graphs.py but loads:
  - Model:        meta-llama/Llama-3.2-1B-Instruct  (cfg.models.robustness)
  - Transcoders:  mntss/transcoder-Llama-3.2-1B  (circuit-tracer "llama" shorthand)
  - Graph output: artifacts/graphs_llama/

Prerequisites (run in order):
  1. scripts/01b_generate_cots_llama.py   → data/pairs/llama/cots_llama.jsonl
  2. scripts/03_construct_pairs.py        → data/pairs/llama/regime_B_truncate.jsonl etc.
     (point --cot-file to cots_llama.jsonl and --out-dir to data/pairs/llama/)

Usage:
    python scripts/04b_generate_graphs_llama.py --config config.yaml [--pilot] [--regime B]

    --pilot    Generate only cfg.pilot.n_items × 3 graph-pairs (~30 graphs total)
    --regime   A, B, C, or any combination (default: B for pilot)

Pair file lookup: data/pairs/llama/{regime_B_truncate,regime_B_addmistake,...}.jsonl
Graph output:     artifacts/graphs_llama/{item_id}_{condition}.json

The graph JSON schema is identical to Gemma graphs (graph_utils.py docstring).
Downstream analysis scripts (20–23) are model-agnostic and work on these files unchanged.
"""
from __future__ import annotations

# ── SSL patch — must be the very first thing, before any circuit-tracer/HF imports
import ssl
import os
os.environ["HF_HUB_DISABLE_SSL_VERIFICATION"] = "1"
os.environ["REQUESTS_CA_BUNDLE"] = ""
os.environ["CURL_CA_BUNDLE"] = ""
ssl._create_default_https_context = ssl._create_unverified_context

# Patch requests.adapters.HTTPAdapter.send (base class for all HF requests)
import requests.adapters as _ra
_orig_adapter_send = _ra.HTTPAdapter.send
def _adapter_send_no_verify(self, request, **kwargs):
    kwargs["verify"] = False
    return _orig_adapter_send(self, request, **kwargs)
_ra.HTTPAdapter.send = _adapter_send_no_verify

import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# ─────────────────────────────────────────────────────────────────────────────

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional, Any

import jsonlines
import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, Config
from src.graph_utils import save_graph, graph_path


PAIR_FILES = {
    "A": ["regime_A_pairs.jsonl"],
    "B": ["regime_B_truncate.jsonl", "regime_B_addmistake.jsonl"],
    "C": ["regime_C_hint.jsonl"],
}


def build_llama_tl_model(cfg: Config, device: str = "cuda"):
    """Load Llama-3.2-1B-Instruct + Llama-Scope transcoders into a
    TransformerLensReplacementModel.  Cached in module scope; safe to call
    multiple times.
    """
    from circuit_tracer.replacement_model.replacement_model_transformerlens import (
        TransformerLensReplacementModel,
    )
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub

    model_name = cfg.models.robustness.name  # meta-llama/Llama-3.2-1B-Instruct
    logger.info(f"Loading Llama-Scope transcoders (mntss/transcoder-Llama-3.2-1B)…")
    transcoder_set, _ = load_transcoder_from_hub(
        "llama",                 # resolved → mntss/transcoder-Llama-3.2-1B
        device=torch.device(device),
        dtype=torch.float16,
    )
    logger.info(f"Building TransformerLensReplacementModel for {model_name}…")
    tl_model = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        model_name,
        transcoder_set,
        device=device,
        dtype=torch.float16,
    )
    logger.info("  ✓ Llama TL model ready.")
    return tl_model


def iter_pair_rows(pairs_dir: Path, regimes: List[str], pilot: bool,
                   pilot_limit: int) -> Iterator[dict]:
    """Yield rows from Llama regime pair files."""
    count = 0
    limit = pilot_limit * 3 if pilot else None  # ×3 for three regimes

    for regime in regimes:
        for fname in PAIR_FILES.get(regime, []):
            fpath = pairs_dir / fname
            if not fpath.exists():
                logger.warning(f"Pair file not found: {fpath} — skipping.")
                continue
            with jsonlines.open(fpath) as reader:
                for row in reader:
                    row["_source_regime"] = regime
                    yield row
                    count += 1
                    if limit and count >= limit:
                        return


def generate_llama_graph(
    tl_model,
    tokenizer,
    prompt: str,
    target_token: str,
    cfg: Config,
    item_id: str,
    condition: str,
    graph_dir: Path,
) -> Optional[Any]:
    """Generate and save one attribution graph with the Llama TL model.

    Returns the graph dict on success, None on skip/error.
    """
    from circuit_tracer.attribution.attribute import attribute
    from circuit_tracer.graph import prune_graph, compute_node_influence

    # Check output exists (resumable)
    out_path = graph_dir / f"{item_id}_{condition}.json"
    if out_path.exists():
        return None

    target_ids = tokenizer.encode(target_token, add_special_tokens=False)
    if len(target_ids) != 1:
        logger.warning(
            f"  [{item_id}/{condition}] Target '{target_token}' encodes to "
            f"{len(target_ids)} tokens — skipping."
        )
        return None

    n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
    max_prompt_tokens = int(os.environ.get("AGD_MAX_GRAPH_TOKENS", "768"))
    if n_tokens > max_prompt_tokens:
        logger.warning(
            f"  [{item_id}/{condition}] Prompt {n_tokens} tokens > {max_prompt_tokens} — skipping."
        )
        return None

    logger.info(f"  [{item_id}/{condition}] Attributing '{target_token}' ({n_tokens} tok)…")
    torch.cuda.empty_cache()

    ag = attribute(
        prompt=prompt,
        model=tl_model,
        attribution_targets=[target_token],
        max_feature_nodes=512,
        batch_size=64,
        verbose=False,
    )

    prune_result = prune_graph(
        ag,
        node_threshold=cfg.agd.pruning_threshold,
        edge_threshold=0.98,
    )

    n_features = len(ag.selected_features)
    n_pos = ag.n_pos
    n_layers = ag.cfg.n_layers
    n_logits = len(ag.logit_targets)

    error_start = n_features
    token_start = error_start + n_layers * n_pos
    logit_start = token_start + n_pos

    node_mask = prune_result.node_mask
    edge_mask = prune_result.edge_mask

    logit_weights = torch.zeros(ag.adjacency_matrix.shape[0], device=ag.adjacency_matrix.device)
    logit_weights[-n_logits:] = ag.logit_probabilities
    try:
        node_influence = compute_node_influence(ag.adjacency_matrix, logit_weights)
    except Exception:
        node_influence = None

    nodes = []
    for i in range(n_features):
        if node_mask[i]:
            active_idx = ag.selected_features[i].item()
            layer, pos, feat_idx = ag.active_features[active_idx].tolist()
            inf_val = float(node_influence[i]) if node_influence is not None else float(ag.activation_values[i])
            nodes.append({
                "feature_id": f"L{layer}_P{pos}_F{feat_idx}",
                "layer": int(layer),
                "influence": inf_val,
                "label": f"Feature {feat_idx} at L{layer} P{pos}",
            })
    for i in range(error_start, token_start):
        if node_mask[i]:
            layer = (i - error_start) // n_pos
            pos = (i - error_start) % n_pos
            inf_val = float(node_influence[i]) if node_influence is not None else 0.0
            nodes.append({
                "feature_id": f"L{layer}_P{pos}_ERR",
                "layer": int(layer),
                "influence": inf_val,
                "label": f"Error node L{layer} P{pos}",
            })
    for i in range(token_start, logit_start):
        if node_mask[i]:
            pos = i - token_start
            inf_val = float(node_influence[i]) if node_influence is not None else 0.0
            nodes.append({
                "feature_id": f"P{pos}_TOK",
                "layer": -1,
                "influence": inf_val,
                "label": f"Token at P{pos}",
            })
    for i in range(logit_start, logit_start + n_logits):
        if node_mask[i]:
            inf_val = float(node_influence[i]) if node_influence is not None else 0.0
            nodes.append({
                "feature_id": f"LOGIT_{i - logit_start}",
                "layer": -1,
                "influence": inf_val,
                "label": f"Logit {i - logit_start}",
            })

    edges = []
    adj = ag.adjacency_matrix
    src_indices, dst_indices = edge_mask.nonzero(as_tuple=True)
    for s, d in zip(src_indices.tolist(), dst_indices.tolist()):
        edges.append({"src": str(s), "dst": str(d), "weight": float(adj[s, d])})

    from datetime import datetime, timezone
    graph_dict = {
        "item_id": item_id,
        "condition": condition,
        "prompt": prompt,
        "target_token": target_token,
        "target_token_id": int(target_ids[0]),
        "pruning_threshold": cfg.agd.pruning_threshold,
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "nodes": nodes,
        "edges": edges,
        "metadata": {
            "model": cfg.models.robustness.name,
            "transcoder_type": "llama-scope-plt",
            "k": cfg.agd.k,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    }
    save_graph(graph_dict, out_path)
    return graph_dict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true",
                        help="Generate only cfg.pilot.n_items × 3 pairs (~30 graphs)")
    parser.add_argument("--regime", default="B",
                        help="Regimes to process: A, B, C, or combination e.g. 'BC' (default: B)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    regimes = list(args.regime.upper())

    llama_pairs_dir = Path(cfg.paths.pairs) / "llama"
    graph_dir = Path(cfg.paths.graphs).parent / "graphs_llama"
    graph_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Llama pilot graph generation: regimes={regimes}, pilot={args.pilot}")
    logger.info(f"Pair files: {llama_pairs_dir}")
    logger.info(f"Graph output: {graph_dir}")

    # Load Llama TL model once
    device = cfg.models.robustness.device if hasattr(cfg.models.robustness, "device") else "cuda"
    tl_model = build_llama_tl_model(cfg, device=device)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.models.robustness.name)

    rows = list(iter_pair_rows(
        llama_pairs_dir, regimes, args.pilot, cfg.pilot.n_items
    ))
    logger.info(f"Found {len(rows)} pair rows to process.")

    n_new = 0
    n_skipped = 0
    for row in tqdm(rows, desc="Llama graphs"):
        iid = row["item_id"]
        target = row.get("target_token", "")

        for cond_key in ("condition0", "condition1"):
            cond = row.get(cond_key)
            prompt_key = "prompt0" if cond_key == "condition0" else "prompt1"
            prompt = row.get(prompt_key, "")
            if not prompt or not cond:
                continue

            # Strip trailing whitespace, ensure "Answer:" trigger
            clean_prompt = prompt.rstrip(" ").rstrip(".!?, \n\t")

            target_clean = target.replace("(", "").replace(")", "").strip()
            if not target_clean.startswith(" "):
                target_clean = " " + target_clean

            import re as _re
            _trailing = _re.compile(
                r'\s*\(?'
                + _re.escape(target_clean.strip())
                + r'\)?\s*$',
                _re.IGNORECASE,
            )
            clean_prompt = _trailing.sub("", clean_prompt).rstrip()
            if not clean_prompt.lower().endswith("answer:"):
                if clean_prompt.endswith(":"):
                    clean_prompt = clean_prompt.rstrip(":") + "Answer:"
                else:
                    clean_prompt = clean_prompt + "\n\nAnswer:"

            try:
                result = generate_llama_graph(
                    tl_model=tl_model,
                    tokenizer=tokenizer,
                    prompt=clean_prompt,
                    target_token=target_clean,
                    cfg=cfg,
                    item_id=iid,
                    condition=cond,
                    graph_dir=graph_dir,
                )
                if result is not None:
                    n_new += 1
                else:
                    n_skipped += 1
            except Exception as e:
                logger.error(f"  [{iid}/{cond}] Failed: {e}")
                n_skipped += 1

    logger.info(f"Done. New graphs: {n_new} | Skipped/errors: {n_skipped}")
    logger.info(f"Graphs saved to: {graph_dir}")
    logger.info("")
    logger.info("Next: run scripts/20_graph_census.py --graph-dir artifacts/graphs_llama")
    logger.info("      to verify non-degeneracy before full campaign.")


if __name__ == "__main__":
    main()
