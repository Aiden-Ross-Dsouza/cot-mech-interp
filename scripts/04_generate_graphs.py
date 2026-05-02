"""
scripts/04_generate_graphs.py
Main attribution-graph campaign.

For every row in the Regime A/B/C pairs JSONL files, generates attribution
graphs for BOTH conditions (prompt0 and prompt1) and saves them as:
  artifacts/graphs/{item_id}_{condition}.json

Fully resumable: skips any graph that already exists on disk.
Checkpoints every N graphs (cfg.graph_gen.checkpoint_every).

Usage:
    python scripts/04_generate_graphs.py --config config.yaml [--pilot] [--regime A|B|C]
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import torch
from pathlib import Path
from typing import Iterator, List

# Fix for CUDA OOM and fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import jsonlines
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config, Config
from src.model_utils import load_main_model
from src.graph_utils import (
    generate_attribution_graph, save_graph, graph_exists, graph_path
)


PAIR_FILES = {
    "A": ["regime_A_pairs.jsonl"],
    "B": ["regime_B_truncate.jsonl", "regime_B_addmistake.jsonl"],
    "C": ["regime_C_hint.jsonl"],
}


def iter_pair_rows(cfg: Config, regimes: List[str], pilot: bool) -> Iterator[dict]:
    """Yield rows from all requested regime pair files."""
    pairs_dir = Path(cfg.paths.pairs)
    count = 0
    limit = cfg.pilot.n_items * 3 if pilot else None  # ×3 for the 3 regimes

    for regime in regimes:
        for fname in PAIR_FILES.get(regime, []):
            fpath = pairs_dir / fname
            if not fpath.exists():
                logger.warning(f"Pair file not found: {fpath} — skipping.")
                continue
            with jsonlines.open(fpath) as reader:
                for row in reader:
                    yield row
                    count += 1
                    if limit and count >= limit:
                        return


def generate_pair(tl_model, tokenizer, row: dict, cfg: Config) -> int:
    """Generate graphs for both conditions in a row. Returns number of new graphs written."""
    graph_dir = Path(cfg.paths.graphs)
    iid = row["item_id"]
    target = row.get("target_token", "")
    n_new = 0

    for cond_key in ("condition0", "condition1"):
        cond = row[cond_key]
        prompt_key = "prompt0" if cond_key == "condition0" else "prompt1"
        prompt = row[prompt_key]

        if graph_exists(iid, cond, graph_dir):
            continue

        # Carefully strip the target token and any trailing formatting.
        clean_prompt = prompt.rstrip()
        
        # If it ends with "**", strip that first
        if clean_prompt.endswith("**"):
            clean_prompt = clean_prompt[:-2]
            
        target_for_attr = target if target else "A"
        # Clean target of any rogue parentheses that might have sneaked in from the dataset (e.g. '(A)')
        target_for_attr = target_for_attr.replace("(", "").replace(")", "").strip()
        
        # Always prepend space to target_for_attr because we will strip all spaces from the prompt end
        if not target_for_attr.startswith(" "):
            target_for_attr = " " + target_for_attr

        # M2 fix: Strip any trailing answer token from the prompt so attribution
        # is performed on the position *before* the answer is given.
        # CoTs from Gemma-2B may end with "Answer: A", "Answer: (A)", or "Answer:(A)".
        # We need to strip all of these variants, not just the bare letter.
        import re as _re
        # Pattern: optional whitespace, optional '(', the target letter, optional ')'
        _trailing = _re.compile(
            r'\s*\(?'
            + _re.escape(target_for_attr.strip())
            + r'\)?\s*$',
            _re.IGNORECASE,
        )
        clean_prompt = _trailing.sub("", clean_prompt).rstrip()
        
        # Standardize trigger: Ensure prompt always ends with "Answer:" (NO trailing space)
        # We verified that Gemma 2 expects exactly "Answer:" to output " A" or " B".
        clean_prompt = clean_prompt.rstrip(" ")
        if not clean_prompt.lower().endswith("answer:"):
            if clean_prompt.endswith(":"):
                clean_prompt = clean_prompt.rstrip(":") + "Answer:"
            else:
                clean_prompt = clean_prompt + "\n\nAnswer:"

        try:
            torch.cuda.empty_cache() # Clear VRAM before starting heavy attribution
            graph = generate_attribution_graph(
                model=None,
                tokenizer=tokenizer,
                prompt=clean_prompt,
                target_token=target_for_attr,
                cfg=cfg,
                item_id=iid,
                condition=cond,
                tl_model=tl_model,
            )
            save_graph(graph, graph_path(iid, cond, graph_dir))
            n_new += 1
        except Exception as e:
            logger.error(f"  [{iid}/{cond}] Graph generation failed: {e}")

    return n_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--regime", default="ABC",
        help="Which regimes to process (e.g. 'A', 'BC', 'ABC')"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    regimes = list(args.regime.upper())

    # Load model once using the same logic as circuit-tracer
    import torch
    from circuit_tracer.replacement_model.replacement_model_transformerlens import TransformerLensReplacementModel
    from circuit_tracer.utils.hf_utils import load_transcoder_from_hub
    
    logger.info(f"Loading {cfg.models.main.name} and transcoders…")
    transcoder_set, _ = load_transcoder_from_hub(
        cfg.transcoders.hf_repo,
        device=torch.device(cfg.models.main.device),
        dtype=torch.float16,
    )
    # Manually filter the transcoder set to save VRAM.
    # First, read d_model and d_sae from a real transcoder so our dummy returns correct-shaped tensors.
    d_model = None
    d_sae = None
    for tc in transcoder_set.transcoders:
        if hasattr(tc, 'W_dec'):
            d_sae = tc.W_dec.shape[0]   # 16384 features
            d_model = tc.W_dec.shape[1] # 2304 residual stream dim
            break
            
    if d_model is None:
        raise ValueError("FATAL: Could not detect d_model/d_sae from transcoders. Dummy injection failed!")
        
    logger.info(f"  Detected d_model = {d_model}, d_sae = {d_sae} from transcoders")

    class DummyTranscoder(torch.nn.Module):
        """No-op transcoder that satisfies the full SingleLayerTranscoder interface.
        Returns correctly-shaped zero tensors so layer reconstructions can be stacked."""
        def __init__(self, d_model, d_sae):
            super().__init__()
            self._d_model = d_model
            self._d_sae = d_sae
        def encode(self, x, apply_activation_function=True):
            return torch.zeros(*x.shape[:-1], self._d_sae, device=x.device, dtype=x.dtype)
        def encode_sparse(self, x, zero_positions=slice(0, 1)):
            empty = torch.zeros(x.shape[0], self._d_sae, device=x.device, dtype=x.dtype)
            active_encoders = torch.zeros(0, self._d_model, device=x.device, dtype=x.dtype)
            return empty.to_sparse(), active_encoders
        def decode(self, acts, input_acts=None):
            return torch.zeros(acts.shape[0], self._d_model, device=acts.device, dtype=acts.dtype)
        def decode_sparse(self, sparse_acts, input_acts=None):
            n_pos = sparse_acts.shape[0]
            reconstruction = torch.zeros(n_pos, self._d_model, device=sparse_acts.device, dtype=sparse_acts.dtype)
            scaled_decoders = torch.zeros(0, self._d_model, device=sparse_acts.device, dtype=sparse_acts.dtype)
            return reconstruction, scaled_decoders
        def compute_skip(self, x): return torch.zeros_like(x)
        def forward(self, x): return x

    if hasattr(transcoder_set, "transcoders") and d_model is not None:
        for i in range(len(transcoder_set.transcoders)):
            if i not in cfg.transcoders.layers:
                transcoder_set.transcoders[i] = DummyTranscoder(d_model, d_sae)
        logger.info(f"  ✓ Replaced unused transcoders with dummies (active layers: {cfg.transcoders.layers})")

    torch.cuda.empty_cache()

    tl_model = TransformerLensReplacementModel.from_pretrained_and_transcoders(
        cfg.models.main.name,
        transcoder_set,
        device=cfg.models.main.device,
        dtype=torch.float16,
    )
    tokenizer = tl_model.tokenizer

    rows = list(iter_pair_rows(cfg, regimes, args.pilot))
    logger.info(f"Total pairs to process: {len(rows)}")

    n_new_total = 0
    checkpoint_every = cfg.graph_gen.checkpoint_every
    t_start = time.time()

    pbar = tqdm(rows, desc="Generating graphs")
    for i, row in enumerate(pbar):
        iid = row["item_id"]
        pbar.set_postfix(item=iid)
        n_new = generate_pair(tl_model, tokenizer, row, cfg)
        n_new_total += n_new

        if (i + 1) % checkpoint_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            eta_h = (len(rows) - i - 1) / rate / 3600 if rate > 0 else float("inf")
            logger.info(
                f"Checkpoint {i+1}/{len(rows)} — "
                f"{n_new_total} new graphs — "
                f"ETA: {eta_h:.1f}h"
            )

    logger.info(f"\nDone. {n_new_total} new graphs written to {cfg.paths.graphs}")


if __name__ == "__main__":
    main()
