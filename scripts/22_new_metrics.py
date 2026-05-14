"""
scripts/22_new_metrics.py
Compute v2.1 metrics: ED (Error-Detection Localisation) and HTIR
(Hint-Token Influence Ratio) for all pairs.

Implements §2.2 and §2.3 of research_plan_v2.md under model-agnostic rules R1–R6.

ED (§2.2):
    Measures fraction of attribution shift concentrated near mistake insertion point.
    Requires knowing mistake token position — inferred from prompt string diff +
    tokenizer if available, otherwise via character-to-token approximation.

HTIR (§2.3):
    Measures fraction of total influence passing through hint-token positions.
    Hint positions computed from the biased prompt using the marker string.

Reads:
    data/pairs/regime_B_addmistake.jsonl
    data/pairs/regime_C_hint.jsonl
    artifacts/graphs/*.json
    artifacts/behavioral/aoc_lanham.parquet
    artifacts/behavioral/turpin_flips.parquet

Writes:
    artifacts/agd/ed_results.parquet        — ED per B_mistake pair
    artifacts/agd/htir_results.parquet      — HTIR per C pair
    analysis/results_new_metrics.json       — correlation tables + statistics

Usage:
    python scripts/22_new_metrics.py --config config.yaml
    python scripts/22_new_metrics.py --config config.yaml --f-window 0.10
    python scripts/22_new_metrics.py --config config.yaml --f-ablation
      (run ED at f ∈ {0.05, 0.10, 0.20, 0.30} for cross-model ablation)
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.graph_utils import load_graph
from src.pano import compute_ed, compute_htir, get_position_influence
from src.stats import spearman_with_ci

try:
    from scipy.stats import mannwhitneyu
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

import jsonlines

# ─────────────────────────────────────────────────────────────────────────────
# Token-position helpers
# ─────────────────────────────────────────────────────────────────────────────

MISTAKE_MARKER = "Wait, actually that's wrong."
HINT_MARKER    = "[Hint: I think the answer is"

# Gemma average chars per token (empirical from Gemma tokenizer on English text)
CHARS_PER_TOKEN_DEFAULT = 3.8


def try_load_tokenizer(model_name: str = "google/gemma-2-2b-it"):
    """Try to load a HuggingFace tokenizer (no GPU needed). Returns None on failure."""
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        logger.info(f"Loaded tokenizer: {model_name}")
        return tok
    except Exception as e:
        logger.warning(f"Could not load tokenizer ({e}). Using character approximation.")
        return None


def get_mistake_token_pos_and_cot_len(
    prompt0: str,
    prompt1: str,
    tokenizer=None,
    chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
) -> Tuple[Optional[int], int]:
    """Return (mistake_token_pos, cot_len_tokens).

    Uses tokenizer if available; falls back to character approximation.
    cot_len_tokens is the length of prompt0 in tokens.
    """
    char_pos = prompt1.find(MISTAKE_MARKER)
    if char_pos < 0:
        return None, max(1, int(len(prompt0) / chars_per_token))

    if tokenizer is not None:
        tokens0 = tokenizer.encode(prompt0)
        tokens1 = tokenizer.encode(prompt1[:char_pos])
        token_pos = len(tokens1)
        cot_len = len(tokens0)
    else:
        token_pos = int(char_pos / chars_per_token)
        cot_len   = max(1, int(len(prompt0) / chars_per_token))
    return token_pos, cot_len


def get_hint_token_positions(
    prompt1: str,
    tokenizer=None,
    chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
) -> List[int]:
    """Return list of token positions for the hint phrase in the biased prompt."""
    start_char = prompt1.find(HINT_MARKER)
    if start_char < 0:
        return []
    end_char = prompt1.find("]", start_char)
    if end_char < 0:
        end_char = start_char + 60  # fallback

    if tokenizer is not None:
        tok_start = len(tokenizer.encode(prompt1[:start_char]))
        tok_end   = len(tokenizer.encode(prompt1[:end_char + 1]))
        return list(range(tok_start, tok_end))
    else:
        tok_start = int(start_char / chars_per_token)
        tok_end   = int(end_char   / chars_per_token)
        return list(range(tok_start, tok_end + 1))


# ─────────────────────────────────────────────────────────────────────────────
# ED computation over all B_mistake pairs
# ─────────────────────────────────────────────────────────────────────────────

def compute_ed_for_pairs(
    pairs: List[Dict],
    graph_dir: Path,
    tokenizer=None,
    f_values: List[float] = (0.10,),
    chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
) -> pd.DataFrame:
    """Compute ED at each f value for all B_mistake pairs.

    Returns DataFrame with columns: item_id, ed_f{f} for each f, plus metadata.
    """
    rows = []
    for pair in tqdm(pairs, desc="ED computation"):
        iid = pair["item_id"]
        p0 = graph_dir / f"{iid}_clean.json"
        p1 = graph_dir / f"{iid}_addmistake.json"

        row: Dict[str, Any] = {
            "item_id": iid,
            "base_item_id": pair.get("base_item_id"),
            "task": pair.get("task"),
            "flipped": pair.get("flipped"),
        }

        if not p0.exists() or not p1.exists():
            for f in f_values:
                row[f"ed_f{f:.2f}"] = np.nan
            rows.append(row)
            continue

        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            prompt0 = pair.get("prompt0", "")
            prompt1 = pair.get("prompt1", "")
            token_pos, cot_len = get_mistake_token_pos_and_cot_len(
                str(prompt0), str(prompt1), tokenizer, chars_per_token
            )
            row["mistake_token_pos_approx"] = token_pos
            row["cot_len_tokens_approx"]    = cot_len

            if token_pos is None:
                for f in f_values:
                    row[f"ed_f{f:.2f}"] = np.nan
            else:
                for f in f_values:
                    ed = compute_ed(g0, g1, token_pos, cot_len, f=f)
                    row[f"ed_f{f:.2f}"] = ed
        except Exception as e:
            logger.warning(f"ED failed {iid}: {e}")
            for f in f_values:
                row[f"ed_f{f:.2f}"] = np.nan

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# HTIR computation over all Regime C pairs
# ─────────────────────────────────────────────────────────────────────────────

def compute_htir_for_pairs(
    pairs: List[Dict],
    graph_dir: Path,
    tokenizer=None,
    chars_per_token: float = CHARS_PER_TOKEN_DEFAULT,
) -> pd.DataFrame:
    """Compute HTIR for all Regime-C pairs."""
    rows = []
    for pair in tqdm(pairs, desc="HTIR computation"):
        iid = pair["item_id"]
        # Hint graph is condition1 = "hint"; filenames use the hint-specific naming
        base_id = pair.get("base_item_id", iid.replace("_hint", ""))
        hint_choice = pair.get("hint_choice", "")
        # Try multiple filename patterns
        p1_candidates = [
            graph_dir / f"{iid}_hint.json",
            graph_dir / f"{base_id}_hint{hint_choice}.json",
            graph_dir / f"{iid}_hint{hint_choice}.json",
        ]
        p1 = next((p for p in p1_candidates if p.exists()), None)

        row: Dict[str, Any] = {
            "item_id": iid,
            "base_item_id": base_id,
            "task": pair.get("task"),
            "flipped": pair.get("flipped"),
            "unfaithful_flip": pair.get("unfaithful_flip"),
            "hint_choice": hint_choice,
        }

        if p1 is None:
            row["htir"] = np.nan
            row["n_hint_positions"] = 0
            rows.append(row)
            continue

        try:
            g1 = load_graph(p1)
            prompt1 = pair.get("prompt1", "")
            hint_positions = get_hint_token_positions(str(prompt1), tokenizer, chars_per_token)
            row["n_hint_positions"] = len(hint_positions)
            row["hint_tok_start"] = hint_positions[0] if hint_positions else np.nan
            row["htir"] = compute_htir(g1, hint_positions)
        except Exception as e:
            logger.warning(f"HTIR failed {iid}: {e}")
            row["htir"] = np.nan
            row["n_hint_positions"] = 0

        rows.append(row)

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ed(ed_df: pd.DataFrame, aoc_df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Correlate ED with aoc_mistake, compare to aoc_truncate columns."""
    results: Dict[str, Any] = {}
    ed_cols = [c for c in ed_df.columns if c.startswith("ed_f")]

    for ed_col in ed_cols:
        col_res: Dict[str, Any] = {}
        if aoc_df is not None:
            merged = ed_df.merge(aoc_df, on="item_id", how="inner")
            aoc_cols = [c for c in merged.columns if c.startswith("aoc_")]
            for aoc_c in aoc_cols:
                valid = merged[[ed_col, aoc_c]].dropna()
                if len(valid) < 10:
                    continue
                rho, ci_lo, ci_hi, p = spearman_with_ci(valid[ed_col].values, valid[aoc_c].values)
                col_res[aoc_c] = {
                    "rho": round(rho, 4), "p": round(p, 6),
                    "ci": [round(ci_lo, 4), round(ci_hi, 4)], "n": len(valid),
                }
        results[ed_col] = col_res

    # Summary stats on default f=0.10
    default_col = "ed_f0.10"
    if default_col in ed_df.columns:
        valid_ed = ed_df[default_col].dropna()
        results["summary"] = {
            "n_valid": int(len(valid_ed)),
            "mean":    float(valid_ed.mean()),
            "std":     float(valid_ed.std()),
            "median":  float(valid_ed.median()),
            "n_nan":   int(ed_df[default_col].isna().sum()),
        }
    return results


def analyze_htir(htir_df: pd.DataFrame) -> Dict[str, Any]:
    """Compute HTIR statistics and AUROC for predicting flips."""
    results: Dict[str, Any] = {}

    valid = htir_df[["htir", "flipped"]].dropna()
    results["n_valid"] = int(len(valid))
    results["n_nan"]   = int(htir_df["htir"].isna().sum())
    results["mean_htir_flip"]    = float(valid[valid["flipped"]]["htir"].mean()) if valid["flipped"].any() else float("nan")
    results["mean_htir_no_flip"] = float(valid[~valid["flipped"]]["htir"].mean()) if (~valid["flipped"]).any() else float("nan")
    results["mean_htir_overall"] = float(valid["htir"].mean())
    results["std_htir"]  = float(valid["htir"].std())
    results["htir_threshold_0.05_fraction"] = float((valid["htir"] > 0.05).mean())

    if HAS_SKLEARN and len(valid) > 10 and valid["flipped"].nunique() == 2:
        try:
            auroc = roc_auc_score(valid["flipped"].astype(int), valid["htir"])
            results["auroc_htir_vs_flip"] = round(float(auroc), 4)
        except Exception as e:
            results["auroc_error"] = str(e)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--f-window", type=float, default=0.10, help="ED window fraction (default 0.10)")
    parser.add_argument("--f-ablation", action="store_true", help="Run ED at f∈{0.05,0.10,0.20,0.30}")
    parser.add_argument("--no-tokenizer", action="store_true", help="Skip tokenizer load, use char approx")
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir    = Path(args.graph_dir or cfg.paths.graphs)
    agd_dir      = Path(cfg.paths.agd)
    behavioral_dir = Path(cfg.paths.behavioral)
    analysis_dir = Path(cfg.paths.analysis)
    pairs_dir    = Path(cfg.paths.pairs)

    f_values = [0.05, 0.10, 0.20, 0.30] if args.f_ablation else [args.f_window]
    logger.info(f"ED f-values: {f_values}")

    # Load tokenizer
    tokenizer = None if args.no_tokenizer else try_load_tokenizer(cfg.models.main.name)

    # Load behavioral ground truth
    aoc_df = None
    aoc_path = behavioral_dir / "aoc_lanham.parquet"
    if aoc_path.exists():
        aoc_df = pd.read_parquet(aoc_path)
        # Normalise item_id: base_item_id for B_mistake pairs maps to AOC item
        if "base_item_id" not in aoc_df.columns and "item_id" in aoc_df.columns:
            aoc_df = aoc_df.rename(columns={"item_id": "base_item_id"})
            aoc_df["item_id"] = aoc_df["base_item_id"]
        logger.info(f"AOC loaded: {len(aoc_df)} items")

    # ── ED on B_mistake pairs ─────────────────────────────────────────────────
    logger.info("\n=== Computing ED on Regime-B mistake pairs ===")
    mistake_pairs = []
    mistake_file = pairs_dir / "regime_B_addmistake.jsonl"
    if mistake_file.exists():
        with jsonlines.open(mistake_file) as reader:
            mistake_pairs = list(reader)
    logger.info(f"Loaded {len(mistake_pairs)} B_mistake pairs")

    ed_df = compute_ed_for_pairs(mistake_pairs, graph_dir, tokenizer, f_values)

    # Merge with AOC using base_item_id
    ed_for_analysis = ed_df.copy()
    if aoc_df is not None and "base_item_id" in ed_df.columns:
        aoc_merge = aoc_df[["item_id"] + [c for c in aoc_df.columns if c.startswith("aoc_")]].copy()
        aoc_merge = aoc_merge.rename(columns={"item_id": "base_item_id"})
        ed_for_analysis = ed_df.merge(aoc_merge, on="base_item_id", how="left")

    ed_stats = analyze_ed(ed_for_analysis, None)  # already merged above

    # Save ED results
    agd_dir.mkdir(parents=True, exist_ok=True)
    ed_out = agd_dir / "ed_results.parquet"
    ed_df.to_parquet(ed_out, index=False)
    logger.info(f"ED results → {ed_out}  ({len(ed_df)} rows)")

    # ── HTIR on Regime-C pairs ────────────────────────────────────────────────
    logger.info("\n=== Computing HTIR on Regime-C hint pairs ===")
    hint_pairs = []
    hint_file = pairs_dir / "regime_C_hint.jsonl"
    if hint_file.exists():
        with jsonlines.open(hint_file) as reader:
            hint_pairs = list(reader)
    logger.info(f"Loaded {len(hint_pairs)} Regime-C pairs")

    htir_df = compute_htir_for_pairs(hint_pairs, graph_dir, tokenizer)
    htir_stats = analyze_htir(htir_df)

    # Save HTIR results
    htir_out = agd_dir / "htir_results.parquet"
    htir_df.to_parquet(htir_out, index=False)
    logger.info(f"HTIR results → {htir_out}  ({len(htir_df)} rows)")

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n=== Summary ===")
    default_col = f"ed_f{args.f_window:.2f}"
    if default_col in ed_stats:
        for aoc_c, s in ed_stats[default_col].items():
            logger.info(f"  ED({args.f_window}) × {aoc_c}: ρ={s['rho']:.3f}, p={s['p']:.4f}, n={s['n']}")
    if "summary" in ed_stats:
        s = ed_stats["summary"]
        logger.info(f"  ED valid: {s['n_valid']}, mean={s['mean']:.3f}, nan={s['n_nan']}")

    logger.info(f"  HTIR valid: {htir_stats.get('n_valid')}, mean={htir_stats.get('mean_htir_overall', 'n/a'):.4f}")
    logger.info(f"  HTIR AUROC vs flip: {htir_stats.get('auroc_htir_vs_flip', 'n/a')}")
    logger.info(f"  Mean HTIR flip={htir_stats.get('mean_htir_flip', 'n/a'):.4f}, "
                f"no-flip={htir_stats.get('mean_htir_no_flip', 'n/a'):.4f}")

    # ── Write combined JSON ───────────────────────────────────────────────────
    output = {
        "ed": ed_stats,
        "htir": htir_stats,
        "config": {
            "f_values": f_values,
            "graph_dir": str(graph_dir),
            "n_mistake_pairs": len(mistake_pairs),
            "n_hint_pairs": len(hint_pairs),
            "tokenizer_used": tokenizer is not None,
        },
    }
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_json = analysis_dir / "results_new_metrics.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nFull results → {out_json}")


if __name__ == "__main__":
    main()
