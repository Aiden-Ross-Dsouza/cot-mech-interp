"""
scripts/21_deep_graph_analysis.py
Deep graph understanding before finalising v2.1 metrics.

Implements §4.2–§4.6 of research_plan_v2.md:
  §4.2  Backbone vs. item-specific concept split + filtered GRACE-T
  §4.3  Per-depth-band GRACE-T on Regime-B pairs × AOC components
  §4.5  Mistake-locality sanity check (position-resolved attribution shift)
  §4.6  Hint-token sanity check (per-position influence in Regime-C biased graphs)

Reads:
    artifacts/graphs/*.json
    artifacts/agd/pano_pairs_with_editdist.parquet   (pairs + GRACE scores)
    artifacts/behavioral/aoc_lanham.parquet

Writes:
    analysis/deep_graph_analysis.json
    analysis/figures/band_grace_vs_aoc.png   (if matplotlib available)
    analysis/figures/mistake_locality_sanity.png
    analysis/figures/hint_token_sanity.png

Usage:
    python scripts/21_deep_graph_analysis.py --config config.yaml
    python scripts/21_deep_graph_analysis.py --config config.yaml --n-sanity 10
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
from src.pano import (
    strip_position,
    graph_to_pano_node_set,
    compute_pano_bands,
    compute_pano_filtered,
    build_concept_frequency_map,
    split_concepts_by_frequency,
    get_position_influence,
    get_node_position,
    normalize_graph_influence,
)
from src.stats import spearman_with_ci

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    logger.warning("matplotlib not available — figures will be skipped")


# ─────────────────────────────────────────────────────────────────────────────
# §4.2 — Backbone vs. item-specific split
# ─────────────────────────────────────────────────────────────────────────────

def build_backbone_split(
    graph_dir: Path,
    k: int = 64,
    backbone_thresh: float = 0.50,
    specific_thresh: float = 0.10,
) -> Tuple[Dict[str, float], set, set, set]:
    """Load all graphs, compute concept frequencies, split backbone/item-specific."""
    graph_files = sorted(graph_dir.glob("*.json"))
    logger.info(f"Building backbone split from {len(graph_files)} graphs...")
    graphs = []
    for gf in tqdm(graph_files, desc="Loading graphs"):
        try:
            graphs.append(load_graph(gf))
        except Exception as e:
            logger.warning(f"Skip {gf.name}: {e}")

    freq_map = build_concept_frequency_map(graphs, k=k)
    backbone, item_specific, middle = split_concepts_by_frequency(
        freq_map, backbone_thresh, specific_thresh
    )
    logger.info(
        f"Concept frequencies: total={len(freq_map)}, "
        f"backbone(>{backbone_thresh:.0%})={len(backbone)}, "
        f"item-specific(<{specific_thresh:.0%})={len(item_specific)}, "
        f"middle={len(middle)}"
    )
    return freq_map, backbone, item_specific, middle


def compute_filtered_grace_for_pairs(
    pairs_df: pd.DataFrame,
    graph_dir: Path,
    k: int,
    backbone: set,
    item_specific: set,
) -> pd.DataFrame:
    """Compute backbone-only and item-specific-only GRACE for each pair."""
    rows = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Filtered GRACE"):
        iid = row["item_id"]
        c0, c1 = row["condition0"], row["condition1"]
        p0 = graph_dir / f"{iid}_{c0}.json"
        p1 = graph_dir / f"{iid}_{c1}.json"
        if not p0.exists() or not p1.exists():
            rows.append({"pano_div_backbone": np.nan, "pano_div_item_specific": np.nan})
            continue
        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            rows.append({
                "pano_div_backbone":      compute_pano_filtered(g0, g1, k, backbone),
                "pano_div_item_specific": compute_pano_filtered(g0, g1, k, item_specific),
            })
        except Exception as e:
            logger.warning(f"Filtered GRACE failed {iid}: {e}")
            rows.append({"pano_div_backbone": np.nan, "pano_div_item_specific": np.nan})
    filtered_df = pd.DataFrame(rows, index=pairs_df.index)
    return pd.concat([pairs_df, filtered_df], axis=1)


# ─────────────────────────────────────────────────────────────────────────────
# §4.3 — Depth-band GRACE on Regime-B pairs
# ─────────────────────────────────────────────────────────────────────────────

def compute_band_grace_for_pairs(
    pairs_df: pd.DataFrame,
    graph_dir: Path,
    k: int = 64,
) -> pd.DataFrame:
    """Compute GRACE-T per depth band for each pair."""
    rows = []
    for _, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Band GRACE"):
        iid = row["item_id"]
        c0, c1 = row["condition0"], row["condition1"]
        p0 = graph_dir / f"{iid}_{c0}.json"
        p1 = graph_dir / f"{iid}_{c1}.json"
        if not p0.exists() or not p1.exists():
            rows.append({f"pano_div_{b}": np.nan for b in ("early", "mid", "late")})
            continue
        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            band_result = compute_pano_bands(g0, g1, k=k)
            rows.append({f"pano_div_{b}": band_result[f"pano_div_{b}"] for b in ("early", "mid", "late")})
        except Exception as e:
            logger.warning(f"Band GRACE failed {iid}: {e}")
            rows.append({f"pano_div_{b}": np.nan for b in ("early", "mid", "late")})
    band_df = pd.DataFrame(rows, index=pairs_df.index)
    return pd.concat([pairs_df, band_df], axis=1)


def correlate_band_grace_with_aoc(merged_df: pd.DataFrame) -> Dict[str, Any]:
    """Spearman ρ of each depth-band GRACE with each AOC component."""
    aoc_cols = [c for c in merged_df.columns if c.startswith("aoc_")]
    results = {}
    for band in ("early", "mid", "late", "full"):
        grace_col = "pano_div" if band == "full" else f"pano_div_{band}"
        if grace_col not in merged_df.columns:
            continue
        band_res = {}
        for aoc_col in aoc_cols:
            valid = merged_df[[grace_col, aoc_col]].dropna()
            if len(valid) < 10:
                continue
            rho, ci_lo, ci_hi, p = spearman_with_ci(valid[grace_col].values, valid[aoc_col].values)
            band_res[aoc_col] = {"rho": round(rho, 4), "p": round(p, 6), "ci": [round(ci_lo, 4), round(ci_hi, 4)], "n": len(valid)}
        results[band] = band_res
    return results


# ─────────────────────────────────────────────────────────────────────────────
# §4.5 — Mistake-locality sanity check
# ─────────────────────────────────────────────────────────────────────────────

MISTAKE_MARKER = "Wait, actually that's wrong."


def find_mistake_char_pos(prompt0: str, prompt1: str) -> Optional[int]:
    """Find the character position of the mistake insertion in prompt1."""
    pos = prompt1.find(MISTAKE_MARKER)
    return pos if pos >= 0 else None


def compute_position_shift_profile(
    graph0: Dict[str, Any],
    graph1: Dict[str, Any],
) -> Dict[int, float]:
    """Position-resolved attribution shift |inf(p,G1) - inf(p,G0)| for all positions."""
    inf0 = get_position_influence(graph0, normalized=True)
    inf1 = get_position_influence(graph1, normalized=True)
    all_pos = sorted(set(inf0) | set(inf1))
    return {p: abs(inf1.get(p, 0.0) - inf0.get(p, 0.0)) for p in all_pos}


def sanity_check_mistake_locality(
    pairs_df: pd.DataFrame,
    graph_dir: Path,
    n_samples: int = 10,
    chars_per_token: float = 4.0,  # rough approximation
) -> Dict[str, Any]:
    """Check whether attribution shifts concentrate near mistake positions.

    Returns per-item locality fractions and aggregate statistics.
    """
    mistake_pairs = pairs_df[pairs_df["regime_label"] == "B_mistake"].head(n_samples)
    results = []
    for _, row in mistake_pairs.iterrows():
        iid = row["item_id"]
        p0 = graph_dir / f"{iid}_clean.json"
        p1 = graph_dir / f"{iid}_addmistake.json"
        if not p0.exists() or not p1.exists():
            continue
        try:
            g0 = load_graph(p0)
            g1 = load_graph(p1)
            prompt0 = row.get("prompt0", "")
            prompt1 = row.get("prompt1", "")
            char_pos = find_mistake_char_pos(str(prompt0), str(prompt1))
            if char_pos is None:
                logger.warning(f"Mistake marker not found in {iid}")
                continue
            token_pos = int(char_pos / chars_per_token)
            cot_len = max(len(str(prompt0)), 1)
            cot_len_tokens = int(cot_len / chars_per_token)
            shift_profile = compute_position_shift_profile(g0, g1)
            total_shift = sum(shift_profile.values())
            # Window = 10% of CoT
            half_w = max(1, int(np.ceil(0.10 * cot_len_tokens / 2)))
            local_shift = sum(
                v for p, v in shift_profile.items()
                if token_pos - half_w <= p <= token_pos + half_w
            )
            locality_frac = local_shift / total_shift if total_shift > 0 else float("nan")
            results.append({
                "item_id": iid,
                "mistake_char_pos": char_pos,
                "mistake_token_pos_approx": token_pos,
                "cot_len_tokens_approx": cot_len_tokens,
                "total_shift": total_shift,
                "local_shift": local_shift,
                "locality_fraction": locality_frac,
                "n_positions": len(shift_profile),
            })
            logger.info(f"  {iid}: locality={locality_frac:.3f} (token~{token_pos}, window±{half_w})")
        except Exception as e:
            logger.warning(f"Sanity check failed {iid}: {e}")

    if not results:
        return {"error": "no valid pairs"}
    df = pd.DataFrame(results)
    return {
        "n_items": len(df),
        "locality_fraction_mean": float(df["locality_fraction"].mean()),
        "locality_fraction_std":  float(df["locality_fraction"].std()),
        "locality_fraction_median": float(df["locality_fraction"].median()),
        "items": results,
        "interpretation": (
            "If locality_fraction_mean >> 1/n_positions, the ED metric should work. "
            f"Mean={df['locality_fraction'].mean():.3f}, "
            f"expected-random={1/(df['n_positions'].mean()):.4f}"
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# §4.6 — Hint-token sanity check
# ─────────────────────────────────────────────────────────────────────────────

HINT_MARKER = "[Hint: I think the answer is"


def find_hint_char_range(prompt1: str) -> Optional[Tuple[int, int]]:
    """Find character start/end of the hint phrase in the biased prompt."""
    start = prompt1.find(HINT_MARKER)
    if start < 0:
        return None
    # Find closing bracket
    end = prompt1.find("]", start)
    if end < 0:
        end = start + 60  # fallback: 60 chars max
    return (start, end)


def sanity_check_hint_token(
    pairs_df: pd.DataFrame,
    graph_dir: Path,
    n_samples: int = 10,
    chars_per_token: float = 4.0,
    n_flip: int = 5,
    n_no_flip: int = 5,
) -> Dict[str, Any]:
    """Check whether hint-token positions show elevated influence in flipped items.

    Compares per-position influence at hint positions for flipped vs non-flipped items.
    """
    c_pairs = pairs_df[pairs_df["regime_label"] == "C"].copy()
    flip_items = c_pairs[c_pairs.get("flipped", False) == True].head(n_flip) if "flipped" in c_pairs.columns else c_pairs.head(n_flip)
    no_flip_items = c_pairs[c_pairs.get("flipped", False) == False].head(n_no_flip) if "flipped" in c_pairs.columns else c_pairs.iloc[n_flip: n_flip + n_no_flip]

    def analyze_group(group_df: pd.DataFrame, label: str) -> List[Dict]:
        group_results = []
        for _, row in group_df.iterrows():
            iid = row["item_id"]
            p1 = graph_dir / f"{iid}_hint.json"
            if not p1.exists():
                # Try hintA, hintB, hintC, hintD
                for suffix in ["hintA", "hintB", "hintC", "hintD"]:
                    candidate = graph_dir / f"{iid.replace('_hint', '')}_{suffix}.json"
                    if candidate.exists():
                        p1 = candidate
                        break
            if not p1.exists():
                continue
            try:
                g1 = load_graph(p1)
                prompt1 = row.get("prompt1", "")
                char_range = find_hint_char_range(str(prompt1))
                if char_range is None:
                    continue
                char_start, char_end = char_range
                tok_start = int(char_start / chars_per_token)
                tok_end   = int(char_end / chars_per_token)
                hint_positions = list(range(tok_start, tok_end + 1))
                pos_inf = get_position_influence(g1, normalized=True)
                hint_inf = sum(pos_inf.get(p, 0.0) for p in hint_positions)
                htir = hint_inf  # already normalized, sums to fraction of total
                group_results.append({
                    "item_id": iid,
                    "label": label,
                    "flipped": bool(row.get("flipped", False)),
                    "hint_char_range": char_range,
                    "hint_token_range": (tok_start, tok_end),
                    "hint_influence_fraction": htir,
                })
                logger.info(f"  {label} {iid}: HTIR={htir:.4f}, hint_pos=[{tok_start},{tok_end}]")
            except Exception as e:
                logger.warning(f"Hint sanity failed {iid}: {e}")
        return group_results

    flip_results   = analyze_group(flip_items, "flip")
    noflip_results = analyze_group(no_flip_items, "no_flip")
    all_results = flip_results + noflip_results

    if not all_results:
        return {"error": "no valid hint pairs"}

    flip_htir   = [r["hint_influence_fraction"] for r in flip_results]
    noflip_htir = [r["hint_influence_fraction"] for r in noflip_results]

    return {
        "n_flip": len(flip_htir),
        "n_no_flip": len(noflip_htir),
        "mean_htir_flip":    float(np.mean(flip_htir)) if flip_htir else float("nan"),
        "mean_htir_no_flip": float(np.mean(noflip_htir)) if noflip_htir else float("nan"),
        "items": all_results,
        "interpretation": (
            "If mean_htir_flip >> mean_htir_no_flip, HTIR is a valid instrument. "
            "If both are near zero, hint operates below the feature level."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting helpers
# ─────────────────────────────────────────────────────────────────────────────

def plot_band_correlations(band_corr: Dict[str, Any], fig_dir: Path) -> None:
    if not HAS_MPL:
        return
    aoc_cols = ["aoc_truncate_25", "aoc_truncate_50", "aoc_truncate_75", "aoc_mistake", "aoc_composite"]
    bands = [b for b in ("early", "mid", "late", "full") if b in band_corr]
    data = {b: [band_corr[b].get(a, {}).get("rho", float("nan")) for a in aoc_cols] for b in bands}

    x = np.arange(len(aoc_cols))
    width = 0.2
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, band in enumerate(bands):
        ax.bar(x + i * width, data[band], width, label=band)
    ax.set_xticks(x + width * len(bands) / 2)
    ax.set_xticklabels([c.replace("aoc_", "") for c in aoc_cols], rotation=30, ha="right")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("GRACE-T per depth band × AOC components")
    ax.legend()
    ax.axhline(0, color="black", linewidth=0.5)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(fig_dir / "band_grace_vs_aoc.png", dpi=150)
    plt.close(fig)
    logger.info(f"Figure saved → {fig_dir}/band_grace_vs_aoc.png")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--graph-dir", default=None)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--n-sanity", type=int, default=10, help="Items per sanity check")
    parser.add_argument("--skip-backbone", action="store_true", help="Skip §4.2 backbone split (saves time)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    graph_dir   = Path(args.graph_dir or cfg.paths.graphs)
    agd_dir     = Path(cfg.paths.agd)
    analysis_dir = Path(cfg.paths.analysis)
    fig_dir     = analysis_dir / "figures"

    # Load pairs
    pairs_parquet = agd_dir / "pano_pairs_with_editdist.parquet"
    if not pairs_parquet.exists():
        pairs_parquet = agd_dir / "pano_pairs.parquet"
    if not pairs_parquet.exists():
        logger.error("No pano_pairs parquet found — run script 13 first.")
        sys.exit(1)
    pairs_df = pd.read_parquet(pairs_parquet)
    logger.info(f"Loaded {len(pairs_df)} pairs")

    # Load AOC
    aoc_parquet = Path(cfg.paths.behavioral) / "aoc_lanham.parquet"
    aoc_df = None
    if aoc_parquet.exists():
        aoc_df = pd.read_parquet(aoc_parquet)
        logger.info(f"Loaded AOC scores for {len(aoc_df)} items")
    else:
        logger.warning("AOC parquet not found — skipping correlation analyses")

    results: Dict[str, Any] = {}

    # ── §4.2: Backbone split ──────────────────────────────────────────────────
    if not args.skip_backbone:
        logger.info("\n=== §4.2 Backbone vs. item-specific split ===")
        freq_map, backbone, item_specific, middle = build_backbone_split(graph_dir, k=args.k)

        results["backbone_split"] = {
            "n_total_concepts": len(freq_map),
            "n_backbone":       len(backbone),
            "n_item_specific":  len(item_specific),
            "n_middle":         len(middle),
            "top_backbone_concepts": sorted(backbone, key=lambda c: freq_map[c], reverse=True)[:20],
        }

        # Compute filtered GRACE on B_trunc pairs
        b_trunc = pairs_df[pairs_df["regime_label"] == "B_trunc"].copy()
        if len(b_trunc) > 0:
            b_trunc = compute_filtered_grace_for_pairs(b_trunc, graph_dir, args.k, backbone, item_specific)
            if aoc_df is not None:
                b_trunc = b_trunc.merge(
                    aoc_df[["item_id"] + [c for c in aoc_df.columns if c.startswith("aoc_")]],
                    on="item_id", how="left",
                )
                filt_corr = {}
                for variant in ("pano_div", "pano_div_backbone", "pano_div_item_specific"):
                    if variant not in b_trunc.columns:
                        continue
                    for aoc_col in [c for c in b_trunc.columns if c.startswith("aoc_")]:
                        valid = b_trunc[[variant, aoc_col]].dropna()
                        if len(valid) < 10:
                            continue
                        rho, ci_lo, ci_hi, p = spearman_with_ci(valid[variant].values, valid[aoc_col].values)
                        filt_corr.setdefault(variant, {})[aoc_col] = {
                            "rho": round(rho, 4), "p": round(p, 6), "n": len(valid)
                        }
                results["backbone_split"]["filtered_grace_correlations"] = filt_corr
                logger.info("Filtered GRACE correlations:")
                for v, v_res in filt_corr.items():
                    for aoc_c, stats in v_res.items():
                        logger.info(f"  {v} × {aoc_c}: ρ={stats['rho']:.3f}, p={stats['p']:.4f}")

    # ── §4.3: Depth-band GRACE ────────────────────────────────────────────────
    logger.info("\n=== §4.3 Depth-band GRACE on Regime-B pairs ===")
    b_pairs = pairs_df[pairs_df["regime_label"].isin(["B_trunc", "B_mistake"])].copy()
    b_pairs = compute_band_grace_for_pairs(b_pairs, graph_dir, k=args.k)

    if aoc_df is not None:
        b_pairs = b_pairs.merge(
            aoc_df[["item_id"] + [c for c in aoc_df.columns if c.startswith("aoc_")]],
            on="item_id", how="left",
        )
        band_corr = correlate_band_grace_with_aoc(b_pairs)
        results["band_grace_correlations"] = band_corr
        logger.info("Band GRACE × AOC correlations:")
        for band, band_res in band_corr.items():
            for aoc_c, stats in band_res.items():
                logger.info(f"  {band} × {aoc_c}: ρ={stats['rho']:.3f}, p={stats['p']:.4f}")
        plot_band_correlations(band_corr, fig_dir)

    # ── §4.5: Mistake-locality sanity ─────────────────────────────────────────
    logger.info("\n=== §4.5 Mistake-locality sanity check ===")

    # Load pair data with prompts for locality check
    import jsonlines
    pairs_data_dir = Path(cfg.paths.pairs)
    mistake_rows = []
    mistake_file = pairs_data_dir / "regime_B_addmistake.jsonl"
    if mistake_file.exists():
        with jsonlines.open(mistake_file) as reader:
            for row in reader:
                row["regime_label"] = "B_mistake"
                mistake_rows.append(row)
    mistake_df = pd.DataFrame(mistake_rows)

    if len(mistake_df) > 0:
        locality_results = sanity_check_mistake_locality(
            mistake_df, graph_dir, n_samples=args.n_sanity
        )
        results["mistake_locality_sanity"] = locality_results
        logger.info(f"Locality fraction mean: {locality_results.get('locality_fraction_mean', 'n/a'):.3f}")
        logger.info(locality_results.get("interpretation", ""))

    # ── §4.6: Hint-token sanity ───────────────────────────────────────────────
    logger.info("\n=== §4.6 Hint-token sanity check ===")

    hint_rows = []
    hint_file = pairs_data_dir / "regime_C_hint.jsonl"
    if hint_file.exists():
        with jsonlines.open(hint_file) as reader:
            for row in reader:
                row["regime_label"] = "C"
                hint_rows.append(row)
    hint_df = pd.DataFrame(hint_rows)

    if len(hint_df) > 0:
        htir_sanity = sanity_check_hint_token(
            hint_df, graph_dir, n_samples=args.n_sanity,
            n_flip=args.n_sanity // 2, n_no_flip=args.n_sanity // 2,
        )
        results["hint_token_sanity"] = htir_sanity
        logger.info(f"Mean HTIR (flip): {htir_sanity.get('mean_htir_flip', 'n/a')}")
        logger.info(f"Mean HTIR (no-flip): {htir_sanity.get('mean_htir_no_flip', 'n/a')}")
        logger.info(htir_sanity.get("interpretation", ""))

    # ── Write results ─────────────────────────────────────────────────────────
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / "deep_graph_analysis.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
