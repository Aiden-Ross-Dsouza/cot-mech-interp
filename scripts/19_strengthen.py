"""
scripts/19_strengthen.py
Four targeted strengthening analyses for the GRACE paper.

Task 1 — Neuronpedia feature lookup:
    Fetch human-readable descriptions for the most influential shared,
    gained, and lost concepts from high/low divergence B_trunc item pairs.
    Requires internet access. Falls back gracefully if unavailable.

Task 2 — Qualitative examples table (B_trunc):
    Select 2 high-GRACE/high-AOC and 2 low-GRACE/low-AOC items from B_trunc.
    For each, show question snippet, GRACE score, AOC, and top concept changes.
    Also loads graph JSONs to extract gained/lost features.

Task 3 — Formal dissociation test:
    Paired bootstrap CI of ρ(GRACE, trunc_AOC) − ρ(GRACE, mistake_AOC).
    If CI excludes zero → the two AOC axes are statistically distinguishable.

Task 4 — GRACE k-sweep ablation:
    Re-compute GRACE at k ∈ {16, 32, 64, 128} from raw graph JSONs.
    Report Spearman ρ(GRACE_k, AOC) for each k to show k=64 is robust.

Reads:
    artifacts/agd/pano_pairs_with_editdist.parquet
    artifacts/behavioral/aoc_lanham.parquet
    artifacts/graphs/*.json

Writes:
    analysis/results_strengthen.json
    analysis/qualitative_examples.json

Usage:
    python scripts/19_strengthen.py --config config.yaml
    python scripts/19_strengthen.py --config config.yaml --skip-neuronpedia
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config
from src.stats import spearman_with_ci
from src.pano import graph_to_pano_node_set, strip_position
from src.graph_utils import load_graph


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_base_id(row) -> str:
    b = row.get("base_item_id")
    if b and not (isinstance(b, float) and np.isnan(b)):
        return b
    return row["item_id"]


def paired_spearman_diff_ci(
    x_a: np.ndarray,
    x_b: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    mask = ~(np.isnan(x_a) | np.isnan(x_b) | np.isnan(y))
    x_a, x_b, y = x_a[mask], x_b[mask], y[mask]
    n = len(y)

    rho_a = scipy_stats.spearmanr(x_a, y)[0]
    rho_b = scipy_stats.spearmanr(x_b, y)[0]
    observed = rho_a - rho_b

    rng = np.random.default_rng(seed)
    boot_diffs = [
        scipy_stats.spearmanr(x_a[idx := rng.choice(n, n, replace=True)], y[idx])[0]
        - scipy_stats.spearmanr(x_b[idx], y[idx])[0]
        for _ in range(n_boot)
    ]
    boot_diffs = np.array(boot_diffs)

    z0 = scipy_stats.norm.ppf(np.mean(boot_diffs < observed) + 1e-12)
    jack = np.array([
        scipy_stats.spearmanr(np.delete(x_a, i), np.delete(y, i))[0]
        - scipy_stats.spearmanr(np.delete(x_b, i), np.delete(y, i))[0]
        for i in range(n)
    ])
    jm = np.mean(jack)
    num = np.sum((jm - jack) ** 3)
    denom = 6 * (np.sum((jm - jack) ** 2) ** 1.5)
    a_acc = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a_acc * (z0 + z_alpha))
        return float(np.clip(scipy_stats.norm.cdf(z_adj) * 100, 0, 100))

    return {
        "rho_trunc": float(rho_a),
        "rho_mistake": float(rho_b),
        "diff": float(observed),
        "ci_lo": float(np.percentile(boot_diffs, _adj(z_lo))),
        "ci_hi": float(np.percentile(boot_diffs, _adj(z_hi))),
        "ci_excludes_zero": bool(float(np.percentile(boot_diffs, _adj(z_lo))) > 0),
        "n": int(n),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 1: Neuronpedia lookup
# ─────────────────────────────────────────────────────────────────────────────

def neuronpedia_lookup(concepts: List[str], model_id: str = "gemma-2-2b") -> Dict:
    """Attempt to fetch feature descriptions from Neuronpedia API."""
    try:
        import urllib.request, urllib.error
    except ImportError:
        return {"status": "unavailable", "reason": "urllib not available"}

    results = {}
    # Try multiple layer/release naming conventions for Gemma Scope
    release_templates = [
        "{layer}-gemmascope-res-16k",
        "{layer}-gemmascope-mlp-65k",
        "{layer}-res-jb",
    ]

    for concept in concepts:
        parts = concept.split("_")
        if len(parts) != 2 or not parts[0].startswith("L") or not parts[1].startswith("F"):
            continue
        layer = int(parts[0][1:])
        feat_idx = int(parts[1][1:])

        found = False
        for tmpl in release_templates:
            layer_str = tmpl.format(layer=layer)
            url = f"https://www.neuronpedia.org/api/feature/{model_id}/{layer_str}/{feat_idx}"
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=8) as resp:
                    data = json.loads(resp.read())
                desc = data.get("explanations", [{}])[0].get("description", "") if data.get("explanations") else ""
                top_acts = [a.get("tokens", "") for a in data.get("activations", [])[:3]]
                results[concept] = {
                    "layer": layer,
                    "feature": feat_idx,
                    "release": layer_str,
                    "description": desc,
                    "top_activating_tokens": top_acts,
                    "url": url,
                }
                found = True
                logger.info(f"  {concept}: '{desc[:80]}'" if desc else f"  {concept}: (no description)")
                break
            except Exception:
                continue

        if not found:
            results[concept] = {"layer": layer, "feature": feat_idx, "status": "not_found"}
            logger.info(f"  {concept}: not found on Neuronpedia")

        time.sleep(0.3)  # be polite

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Task 2: Qualitative examples (B_trunc)
# ─────────────────────────────────────────────────────────────────────────────

def extract_feature_changes(
    item_id: str, condition0: str, condition1: str,
    graph_dir: Path, k: int = 64,
) -> Optional[Dict]:
    """Load two graphs and return gained/lost/shared concept sets."""
    p0 = graph_dir / f"{item_id}_{condition0}.json"
    p1 = graph_dir / f"{item_id}_{condition1}.json"
    if not p0.exists() or not p1.exists():
        return None
    try:
        g0 = load_graph(p0)
        g1 = load_graph(p1)
    except Exception:
        return None

    n0 = graph_to_pano_node_set(g0, k=k)
    n1 = graph_to_pano_node_set(g1, k=k)
    gained = {c: v for c, v in n1.items() if c not in n0}
    lost = {c: v for c, v in n0.items() if c not in n1}
    shared = {c: n0[c] for c in n0 if c in n1}

    top_gained = sorted(gained.items(), key=lambda x: x[1], reverse=True)[:5]
    top_lost = sorted(lost.items(), key=lambda x: x[1], reverse=True)[:5]
    top_shared = sorted(shared.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "top_gained": [{"concept": c, "influence": round(v, 5)} for c, v in top_gained],
        "top_lost": [{"concept": c, "influence": round(v, 5)} for c, v in top_lost],
        "top_shared": [{"concept": c, "influence": round(v, 5)} for c, v in top_shared],
        "n_gained": len(gained),
        "n_lost": len(lost),
        "n_shared": len(shared),
    }


def run_qualitative_examples(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    graph_dir: Path,
) -> Dict:
    """Select 2 high and 2 low GRACE/AOC B_trunc examples with feature details."""
    b_trunc = pano_df[pano_df["regime_label"] == "B_trunc"].copy()

    # Per-item: take the trunc_50 pair (middle truncation — most representative)
    b50 = b_trunc[b_trunc["condition1"] == "truncate_50"].copy()
    merged = b50.merge(
        aoc_df[["item_id", "aoc_composite", "aoc_truncate_50"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["pano_div", "aoc_composite"])

    merged_sorted = merged.sort_values("pano_div")

    # Get 2 low and 2 high examples that have graph files
    examples = {"high_grace_high_aoc": [], "low_grace_low_aoc": []}

    for row in merged_sorted.tail(20).itertuples():
        if len(examples["high_grace_high_aoc"]) >= 2:
            break
        if row.aoc_composite < 0.3:
            continue
        feats = extract_feature_changes(
            row.item_id_x, row.condition0, row.condition1, graph_dir
        )
        if feats is None:
            continue
        prompt_snippet = str(row.prompt0)[:120].replace("\n", " ") if hasattr(row, "prompt0") else ""
        examples["high_grace_high_aoc"].append({
            "item_id": row.item_id_x,
            "base_item_id": row.base_item_id,
            "question_snippet": prompt_snippet,
            "grace_div": round(float(row.pano_div), 4),
            "aoc_composite": round(float(row.aoc_composite), 3),
            "condition": row.condition1,
            **feats,
        })

    for row in merged_sorted.head(20).itertuples():
        if len(examples["low_grace_low_aoc"]) >= 2:
            break
        if row.aoc_composite > 0.7:
            continue
        feats = extract_feature_changes(
            row.item_id_x, row.condition0, row.condition1, graph_dir
        )
        if feats is None:
            continue
        prompt_snippet = str(row.prompt0)[:120].replace("\n", " ") if hasattr(row, "prompt0") else ""
        examples["low_grace_low_aoc"].append({
            "item_id": row.item_id_x,
            "base_item_id": row.base_item_id,
            "question_snippet": prompt_snippet,
            "grace_div": round(float(row.pano_div), 4),
            "aoc_composite": round(float(row.aoc_composite), 3),
            "condition": row.condition1,
            **feats,
        })

    logger.info(f"High-GRACE examples found: {len(examples['high_grace_high_aoc'])}")
    logger.info(f"Low-GRACE examples found: {len(examples['low_grace_low_aoc'])}")
    for ex in examples["high_grace_high_aoc"]:
        logger.info(f"  HIGH: {ex['base_item_id']} | GRACE={ex['grace_div']} | AOC={ex['aoc_composite']}")
    for ex in examples["low_grace_low_aoc"]:
        logger.info(f"  LOW:  {ex['base_item_id']} | GRACE={ex['grace_div']} | AOC={ex['aoc_composite']}")

    return examples


# ─────────────────────────────────────────────────────────────────────────────
# Task 3: Formal dissociation test
# ─────────────────────────────────────────────────────────────────────────────

def run_dissociation_test(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    n_boot: int,
    seed: int,
) -> Dict:
    """Paired bootstrap CI of ρ(GRACE, trunc_AOC) - ρ(GRACE, mistake_AOC)."""
    b_df = pano_df[pano_df["regime_label"].isin(["B_mistake", "B_trunc"])].copy()
    per_item = (b_df.groupby("base_item_id")["pano_div"]
                .mean().reset_index()
                .rename(columns={"pano_div": "grace_div_mean"}))

    aoc_cols = ["aoc_truncate_50", "aoc_mistake", "aoc_composite",
                "aoc_truncate_25", "aoc_truncate_75"]
    avail = [c for c in aoc_cols if c in aoc_df.columns]
    merged = per_item.merge(
        aoc_df[["item_id"] + avail],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["grace_div_mean"])

    logger.info(f"Dissociation test: n={len(merged)}")

    # Use average of truncation sub-scores as the trunc-AOC signal
    trunc_cols = [c for c in ["aoc_truncate_25", "aoc_truncate_50", "aoc_truncate_75"]
                  if c in merged.columns]
    if trunc_cols:
        merged["aoc_trunc_mean"] = merged[trunc_cols].mean(axis=1)
    else:
        return {"status": "skipped", "reason": "No truncation AOC columns"}

    if "aoc_mistake" not in merged.columns:
        return {"status": "skipped", "reason": "aoc_mistake not in AOC data"}

    x_grace = merged["grace_div_mean"].values
    y_trunc = merged["aoc_trunc_mean"].values
    y_mistake = merged["aoc_mistake"].values

    result = paired_spearman_diff_ci(x_grace, x_grace, y_trunc, n_boot=n_boot, seed=seed)

    # Re-run correctly: ρ(grace, trunc) vs ρ(grace, mistake)
    rho_trunc = scipy_stats.spearmanr(x_grace, y_trunc)[0]
    rho_mistake = scipy_stats.spearmanr(x_grace, y_mistake)[0]
    observed = rho_trunc - rho_mistake

    rng = np.random.default_rng(seed)
    n = len(x_grace)
    boot_diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        d_trunc = scipy_stats.spearmanr(x_grace[idx], y_trunc[idx])[0]
        d_mistake = scipy_stats.spearmanr(x_grace[idx], y_mistake[idx])[0]
        val = d_trunc - d_mistake
        if not np.isnan(val):
            boot_diffs.append(val)
    boot_diffs = np.array(boot_diffs)

    z0 = scipy_stats.norm.ppf(np.mean(boot_diffs < observed) + 1e-12)
    jack = []
    for i in range(n):
        xa_j = np.delete(x_grace, i)
        yt_j = np.delete(y_trunc, i)
        ym_j = np.delete(y_mistake, i)
        jack.append(scipy_stats.spearmanr(xa_j, yt_j)[0] - scipy_stats.spearmanr(xa_j, ym_j)[0])
    jack = np.array(jack)
    jm = np.mean(jack)
    num = np.sum((jm - jack) ** 3)
    denom = 6 * (np.sum((jm - jack) ** 2) ** 1.5)
    a_acc = num / denom if denom != 0 else 0.0

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a_acc * (z0 + z_alpha))
        return float(np.clip(scipy_stats.norm.cdf(z_adj) * 100, 0, 100))

    ci_lo = float(np.percentile(boot_diffs, _adj(scipy_stats.norm.ppf(0.025))))
    ci_hi = float(np.percentile(boot_diffs, _adj(scipy_stats.norm.ppf(0.975))))

    logger.info(f"  ρ(GRACE, trunc_AOC) = {rho_trunc:.4f}")
    logger.info(f"  ρ(GRACE, mistake_AOC) = {rho_mistake:.4f}")
    logger.info(f"  Difference = {observed:.4f}, CI=[{ci_lo:.4f}, {ci_hi:.4f}]")
    logger.info(f"  CI excludes zero: {ci_lo > 0}")

    return {
        "rho_trunc_aoc": float(rho_trunc),
        "rho_mistake_aoc": float(rho_mistake),
        "diff": float(observed),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "ci_excludes_zero": bool(ci_lo > 0),
        "n": int(n),
        "interpretation": (
            "Statistically confirmed: GRACE predicts truncation-faithfulness significantly "
            "more than mistake-faithfulness. Two axes are mechanistically distinct."
            if ci_lo > 0 else
            "Directional difference but CI includes zero — dissociation is suggestive, not confirmed."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Task 4: GRACE k-sweep ablation
# ─────────────────────────────────────────────────────────────────────────────

def run_k_sweep(
    pano_df: pd.DataFrame,
    aoc_df: pd.DataFrame,
    graph_dir: Path,
    k_values: List[int],
    n_boot: int,
    seed: int,
    max_items: int = 100,
) -> Dict:
    """Re-compute GRACE at different k values from raw graph JSONs."""
    b_trunc = pano_df[pano_df["regime_label"] == "B_trunc"].copy()

    # Pick trunc_50 pairs — one representative truncation per item
    b50 = b_trunc[b_trunc["condition1"] == "truncate_50"].copy()
    merged_base = b50.merge(
        aoc_df[["item_id", "aoc_composite"]],
        left_on="base_item_id", right_on="item_id", how="inner"
    ).dropna(subset=["aoc_composite"])

    # Sample for speed
    sample = merged_base.sample(min(max_items, len(merged_base)), random_state=seed)
    logger.info(f"k-sweep: {len(sample)} B_trunc pairs (trunc_50)")

    results = {}
    for k in k_values:
        grace_scores = []
        aoc_scores = []
        skipped = 0

        id_col = "item_id_x" if "item_id_x" in sample.columns else "item_id"
        for _, row in sample.iterrows():
            p0 = graph_dir / f"{row[id_col]}_{row['condition0']}.json"
            p1 = graph_dir / f"{row[id_col]}_{row['condition1']}.json"
            if not p0.exists() or not p1.exists():
                skipped += 1
                continue
            try:
                g0 = load_graph(p0)
                g1 = load_graph(p1)
                n0 = graph_to_pano_node_set(g0, k=k)
                n1 = graph_to_pano_node_set(g1, k=k)
                from src.agd import weighted_jaccard
                sim = weighted_jaccard(n0, n1)
                grace_scores.append(1.0 - sim)
                aoc_scores.append(float(row["aoc_composite"]))
            except Exception as e:
                skipped += 1

        if len(grace_scores) < 20:
            results[str(k)] = {"status": "skipped", "n": len(grace_scores)}
            continue

        r = spearman_with_ci(
            np.array(grace_scores), np.array(aoc_scores),
            n_boot=n_boot, seed=seed,
        )
        results[str(k)] = r
        logger.info(
            f"  k={k:4d}: ρ={r['rho']:+.4f}, p={r['p']:.4f}, "
            f"CI=[{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}], n={r['n']}, skipped={skipped}"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--skip-neuronpedia", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    n_boot = cfg.stats.n_bootstrap

    pano_df = pd.read_parquet(
        Path(cfg.paths.agd) / "pano_pairs_with_editdist.parquet"
    )
    aoc_df = pd.read_parquet(
        Path(cfg.paths.behavioral) / "aoc_lanham.parquet"
    )
    graph_dir = Path(cfg.paths.graphs)

    results = {}

    # ── Task 2: Qualitative examples ─────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Task 2: Qualitative examples (B_trunc)")
    logger.info("="*60)
    qual = run_qualitative_examples(pano_df, aoc_df, graph_dir)
    results["qualitative_examples"] = qual
    with open(Path(cfg.paths.analysis) / "qualitative_examples.json", "w") as f:
        json.dump(qual, f, indent=2, default=str)

    # ── Task 3: Formal dissociation test ─────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Task 3: Formal dissociation test (trunc vs mistake AOC)")
    logger.info("="*60)
    diss = run_dissociation_test(pano_df, aoc_df, n_boot=n_boot, seed=cfg.seed)
    results["dissociation_test"] = diss
    logger.info(f"  Interpretation: {diss.get('interpretation', 'N/A')}")

    # ── Task 4: k-sweep ablation ──────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("Task 4: GRACE k-sweep ablation")
    logger.info("="*60)
    k_sweep = run_k_sweep(
        pano_df, aoc_df, graph_dir,
        k_values=[16, 32, 64, 128],
        n_boot=min(n_boot, 1000),  # faster for ablation
        seed=cfg.seed,
        max_items=120,
    )
    results["k_sweep"] = k_sweep

    # ── Task 1: Neuronpedia lookup ────────────────────────────────────────────
    if not args.skip_neuronpedia:
        logger.info("\n" + "="*60)
        logger.info("Task 1: Neuronpedia feature lookup")
        logger.info("="*60)
        # Collect key recurring concepts across examples
        key_concepts = [
            "L22_F11133", "L18_F10940",  # appear as top shared across ALL examples
            "L24_F12351", "L22_F14263",  # top shared in low-div items
            "L25_F15341", "L25_F5714",   # top shared/lost in high-div items
        ]
        # Also add top gained/lost from qualitative examples
        for cat in ["high_grace_high_aoc", "low_grace_low_aoc"]:
            for ex in qual.get(cat, []):
                for feat in ex.get("top_gained", [])[:2] + ex.get("top_lost", [])[:2]:
                    c = feat.get("concept", "")
                    if c and c not in key_concepts:
                        key_concepts.append(c)

        logger.info(f"Looking up {len(key_concepts)} concepts: {key_concepts}")
        np_results = neuronpedia_lookup(key_concepts[:12])  # cap at 12 to be polite
        results["neuronpedia"] = np_results
    else:
        logger.info("Skipping Neuronpedia lookup (--skip-neuronpedia)")
        results["neuronpedia"] = {"status": "skipped"}

    # ── Save ──────────────────────────────────────────────────────────────────
    out = Path(cfg.paths.analysis) / "results_strengthen.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\nAll results -> {out}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "="*60)
    logger.info("STRENGTHENING SUMMARY")
    logger.info("="*60)
    logger.info(f"  Qualitative examples: {len(qual.get('high_grace_high_aoc', []))} high, "
                f"{len(qual.get('low_grace_low_aoc', []))} low")
    diss_ci = diss.get('ci_excludes_zero', 'N/A')
    logger.info(f"  Dissociation CI excludes zero: {diss_ci}")
    logger.info(f"  k-sweep results: {list(k_sweep.keys())}")
    np_found = sum(1 for v in results.get("neuronpedia", {}).values()
                   if isinstance(v, dict) and v.get("description"))
    logger.info(f"  Neuronpedia features with descriptions: {np_found}")


if __name__ == "__main__":
    main()
