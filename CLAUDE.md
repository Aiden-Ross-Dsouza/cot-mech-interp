# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**GRACE (Graph Reasoning Attribution for CoT Evaluation)** — a mechanistic interpretability research project that measures faithfulness of chain-of-thought reasoning in LLMs by comparing attribution graphs across paired CoT conditions.

**Target venue:** ICML 2026 Mechanistic Interpretability Workshop. **Submitted May 8, 2026 AOE (deadline passed — paper is in review).**

**One-line pitch:** We propose GRACE, a position-agnostic, influence-weighted attribution graph comparison metric for CoT faithfulness. The key design insight — that naive feature-ID comparison silently measures sequence-length drift rather than mechanism change — is itself a community contribution. GRACE reveals that graph-level divergence captures CoT *necessity* but not sycophancy or error-detection, a dissociation that is mechanistically informative.

---

## The Method: GRACE

Attribution graphs (from Anthropic's circuit-tracer + Gemma Scope transcoders) trace which internal features influenced the model's output. GRACE compares graphs across paired CoT conditions:

**Core design principle:** Feature IDs encode token position (`L3_P12_F1047` = Layer 3, Position 12, Feature 1047). Comparing graphs naively treats the same conceptual feature at different positions as entirely different nodes, so any Jaccard-style metric measures sequence-length drift, not mechanism change. GRACE strips position before comparison, collapsing `L3_P12_F1047` and `L3_P24_F1047` to concept `L3_F1047`.

**Formula:**
```
concept(feature_id) = strip_position(feature_id)          # e.g. L3_F1047
influence(concept)  = max over positions of |influence|   # take max across positions
GRACE_div(G0, G1)   = 1 − weighted_jaccard(top-k(G0), top-k(G1))
```
- `k = 64` top concepts by influence (ablated across k ∈ {16, 32, 64, 128} — all robust)
- Influence weighting is essential: +0.16 ρ over raw shared-count (ablated in script 18)

**Implementation:** `src/pano.py` — `graph_to_pano_node_set`, `compute_pano`, `batch_pano`

---

## Headline Results

All primary numbers on the **full 143-item Regime B set**. GRACE has no learned parameters (deterministic, parameter-free), so the 60/40 train/test split is a consistency check, not a statistical requirement. Test-split numbers (n=50) are reported alongside as directional verification and agree in sign and magnitude.

### H1 — GRACE correlates with behavioural CoT faithfulness ✅ PASSED

| Set | n | ρ(GRACE, AOC) | p | 95% CI |
|---|---|---|---|---|
| **Full set (primary)** | **143** | **+0.444** | **≪ 0.001** | **[+0.29, +0.56]** |
| Test split (consistency) | 50 | +0.479 | 0.0004 | [+0.12, +0.70] |

Ground truth: Lanham AOC (behavioural — truncate CoT at multiple depths, measure answer stability).

**Cross-dataset: H1 holds on both BBH and MMLU independently**

| Dataset | n | ρ | p | CI |
|---|---|---|---|---|
| BBH | 77 | +0.419 | 0.0001 | [+0.16, +0.60] |
| MMLU | 66 | +0.489 | <0.0001 | [+0.31, +0.64] |

### H2 — GRACE does not predict sycophantic hint-flips ❌ PRE-REGISTERED NULL

| Metric | AUROC (full set) | CI | Verdict |
|---|---|---|---|
| GRACE_div on Regime C | 0.469 | [0.40, 0.54] | Chance |
| δ-AGD on Regime C | 0.562 | [0.46, 0.66] | Chance |

**Interpretation:** GRACE measures whether the CoT is structurally engaged in the model's computation. Hint-induced sycophancy operates through a different mechanistic pathway — this null is itself informative (see Mechanistic Dissociation below).

### H3 — GRACE strictly outperforms textual baselines ✅ PASSED

GRACE_div vs edit distance: Δρ = +0.329, CI [+0.080, +0.578] → **CI excludes zero** ✅
GRACE_div vs length difference: Δρ = +0.336, CI [+0.091, +0.561] → **CI excludes zero** ✅

Partial Spearman controlling for edit_norm + len_diff + graph-size baselines: **ρ = +0.341, p < 0.0001** (from script 16). GRACE_div captures information that purely textual measures cannot.

---

## Key Findings

### Finding 1 — Mechanistic Dissociation (Novel)

GRACE_div predicts truncation-based faithfulness but not mistake-based faithfulness:

| AOC sub-score | ρ(GRACE_div, sub-score) | p |
|---|---|---|
| aoc_truncate_25 | +0.300 | 0.0003 |
| aoc_truncate_50 | **+0.348** | <0.0001 |
| aoc_truncate_75 | **+0.343** | <0.0001 |
| **aoc_mistake** | **−0.091** | **0.28 (NULL)** |
| aoc_composite | +0.444 | <0.0001 |

**Formally confirmed:** Paired bootstrap of ρ(GRACE, trunc_AOC) − ρ(GRACE, mistake_AOC) = **+0.537, CI [+0.33, +0.76]**. CI cleanly excludes zero (`scripts/19_strengthen.py`).

**What this means:** GRACE captures "does the model need the CoT to reach its answer?" (truncation faithfulness) but is blind to "did the model detect the error in the CoT?" (mistake faithfulness). These are two mechanistically distinct forms of unfaithfulness requiring different measurement tools. The H2 null connects to this: prompt-level sycophancy is a third axis, also orthogonal.

### Finding 2 — B_trunc Drives H1, B_mistake Is Weaker

| Regime | ρ(GRACE, AOC) | p | CI |
|---|---|---|---|
| B_trunc only | +0.402 | <0.0001 | [+0.25, +0.53] |
| B_mistake only | +0.223 | 0.007 | [+0.04, +0.38] |

Consistent with Finding 1: GRACE's signal is strongest when the perturbation type (truncation) matches the faithfulness type it captures.

### Finding 3 — Influence Weighting Is Essential

| Metric | ρ(., AOC) | CI |
|---|---|---|
| GRACE_div (influence-weighted) | +0.444 | [+0.29, +0.56] |
| Unweighted shared-concept fraction | +0.284 | [+0.13, +0.42] |
| **Δρ advantage of weighting** | **+0.160** | — |

The influence-weighted Jaccard formulation contributes +0.16 ρ over simply counting shared concepts. Do not simplify the metric without accepting this cost.

### Finding 4 — k Robustness (Ablation)

GRACE is robust across a wide range of k values:

| k | ρ(GRACE, AOC) | CI |
|---|---|---|
| 16 | +0.370 | [+0.17, +0.54] |
| 32 | +0.349 | [+0.14, +0.52] |
| **64 (default)** | **+0.341** | **[+0.13, +0.52]** |
| 128 | +0.343 | [+0.13, +0.52] |

Choice of k=64 is not cherry-picked. All k values give significant, similar results.

### Finding 5 — Neuronpedia Feature Interpretations

Key high-influence shared concepts (active across most items — domain-general "backbone" features):

| Concept | Neuronpedia Description |
|---|---|
| L22_F11133 | Episodic structure and narrative progression |
| L18_F10940 | Female characters or pronouns |
| L24_F12351 | Political positions and roles |
| L22_F14263 | Threading and task management (code) |
| L25_F5714 | **Small sample sizes in studies** (meta-scientific) |
| L24_F6157 | References to time |

Features gained in **high-GRACE** pairs (mechanism reorganised under truncation): L24_F6157 "time", L23_F10308 "numerical data patterns" — more domain-specific, suggesting the model routes through item-specific reasoning when the CoT is structurally engaged.

Neuronpedia URL format for Gemma-2-2B Gemma Scope: `https://www.neuronpedia.org/api/feature/gemma-2-2b/{LAYER}-gemmascope-res-16k/{FEATURE_IDX}`

### Finding 6 — Qualitative Examples

High-GRACE / High-AOC pairs (CoT is genuinely load-bearing):
- BBH logical deduction: GRACE=0.94, AOC=0.4 — mechanism substantially reorganises under truncation
- MMLU high school physics: GRACE=0.94, AOC=0.6

Low-GRACE / Low-AOC pairs (CoT is decorative):
- MMLU logical fallacies (×2): GRACE≈0.15, AOC=0.2 — mechanism barely changes regardless of truncation

Full feature sets in `analysis/qualitative_examples.json`.

---

## What Is Retracted / Not Claimed

- **"Mistake anomaly is mechanistic"** — Retracted. The apparent B_mistake < A GRACE_div ordering is entirely driven by edit distance (Regime A rewrites ~44% tokens, B_mistake only ~3%). Partial Spearman controlling for edit_norm: ρ = +0.028, p = 0.63 — zero effect.
- **"H2 passes on MMLU"** — Retracted. The 0.678 AUROC on MMLU was on the discovery/full set. On the held-out test split it collapses to AUROC ≈ 0.34.
- **"Truncation monotonicity predicts faithfulness"** — Genuine null. Per-item Kendall τ(trunc_fraction, GRACE_div) does not correlate with AOC (ρ = +0.028, p = 0.74, n=143).
- **GRACE predicts sycophancy** — Not claimed. H2 is reported as a pre-registered null.
- **GRACE generalises across models** — Not tested (no GPU for new graphs). Single model (Gemma-2-2B) limitation stated.

---

## Scripts (`scripts/`)

Run in numbered order. Scripts 13–19 are the analysis layer; 01–12 are the data/graph generation pipeline (requires GPU).

| Script | Purpose | Run command |
|--------|---------|-------------|
| `01–12` | Data → CoTs → graphs → baselines → figures (GPU required) | `make all` |
| `13_compute_pano_dagd.py` | Compute GRACE_div + delta-AGD for all pairs → `artifacts/agd/pano_pairs.parquet` | `python scripts/13_...` |
| `14_pano_dagd_analysis.py` | H1/H2 tests on GRACE. Use `--use-full-set` for primary numbers. → `analysis/results_pano.json` | `python scripts/14_... --use-full-set` |
| `15_extended_analysis.py` | Per-dataset H1, regime distributions, qualitative examples (B_mistake). → `analysis/extended_results.json` | `python scripts/15_... --use-full-set` |
| `16_robustness_checks.py` | Edit-distance confound audit + partial Spearman incremental value. → `analysis/robustness_results.json` | `python scripts/16_... --use-full-set` |
| `17_new_hypotheses.py` | **H3 head-to-head baselines** (primary H3) + cross-dataset H1 + truncation monotonicity (null). → `analysis/results_new_h2_h3.json` | `python scripts/17_... --use-full-set` |
| `18_aoc_decomposition.py` | AOC component breakdown + per-perturbation H1 + shared-count ablation. → `analysis/results_decomposition.json` | `python scripts/18_...` |
| `19_strengthen.py` | Neuronpedia lookup + qualitative B_trunc examples + formal dissociation test + k-sweep ablation. → `analysis/results_strengthen.json`, `analysis/qualitative_examples.json` | `python scripts/19_...` |

---

## Analysis Artifacts

| File | Contents |
|------|----------|
| `artifacts/agd/pano_pairs.parquet` | Per-pair GRACE_div, delta-AGD, regime labels, n=1321 rows |
| `artifacts/agd/pano_pairs_with_editdist.parquet` | Above + edit_norm, len_diff, len0/len1 |
| `artifacts/behavioral/aoc_lanham.parquet` | Per-item AOC sub-scores (early, trunc_25/50/75, mistake, composite) |
| `artifacts/behavioral/turpin_flips.parquet` | Per-item hint-flip labels |
| `artifacts/graphs/*.json` | 1977 attribution graph JSON files (~800 nodes each) |
| `analysis/results_pano.json` | H1/H2 test results (script 14) |
| `analysis/robustness_results.json` | Edit-distance audit + partial Spearman (script 16) |
| `analysis/results_new_h2_h3.json` | H3 head-to-head + cross-dataset + monotonicity (script 17) |
| `analysis/results_decomposition.json` | AOC breakdown + ablations (script 18) |
| `analysis/results_strengthen.json` | Dissociation test + k-sweep + Neuronpedia (script 19) |
| `analysis/qualitative_examples.json` | 4 qualitative B_trunc examples with feature sets |
| `analysis/feature_analysis.json` | Top gained/lost features for B_mistake examples (script 15) |

---

## Source Modules (`src/`)

| Module | Role |
|--------|------|
| `config.py` | Frozen dataclass config loader (YAML → Config) |
| `model_utils.py` | Load Gemma-2-2B/9B-it, run inference, manage transcoders |
| `graph_utils.py` | circuit-tracer AttributionGraph wrapper, JSON serialize/deserialize |
| `agd.py` | `weighted_jaccard`, legacy AGD metric internals |
| `pano.py` | **GRACE core** — `strip_position`, `graph_to_pano_node_set`, `compute_pano`, `batch_pano`, `compute_delta_agd` |
| `baselines.py` | 5 baseline metrics (activation-cosine, KL, perplexity, self-consistency, random-Jaccard) |
| `behavioral.py` | Lanham AOC + Turpin hint-injection protocols |
| `stats.py` | BCa bootstrap CIs, AUROC, Spearman ρ, Holm-Bonferroni, Cliff's delta |

---

## Experimental Setup

- **Primary model:** Gemma-2-2B-it (fp16, ~5 GB VRAM)
- **Transcoders:** Gemma Scope PLT across all 26 layers (~4 GB VRAM)
- **Datasets:** BBH (300 items), MMLU (150), GSM8K (100), Turpin (200) → 560 items with AOC labels
- **Regime B items with GRACE + AOC:** 143 (full set) / 50 (test split)
- **Total graphs:** ~1977 JSON files, ~25 GPU-hours on A100 to generate
- **k=64** top concepts (robust across k ∈ {16, 32, 64, 128})
- **Regimes:** A = paraphrase, B_mistake = error injection, B_trunc = CoT truncation, C = hint injection

---

## Paper Narrative (for writing reference)

**§1 Introduction** — CoT faithfulness gap: do LLMs actually use their stated reasoning? Existing measures are behavioural. We measure directly from internal computation via attribution graphs. Key design challenge: position-indexed feature IDs silently corrupt naive graph comparison.

**§2 Background** — Attribution graphs, Gemma Scope transcoders, Lanham AOC, Turpin hint protocol.

**§3 GRACE Metric** — Position-agnostic concept mapping → influence-weighted Jaccard → GRACE_div. One paragraph on why position-agnostic design is necessary (not "we found a bug", but "we address a design challenge").

**§4 Experiments** — Three regimes, pre-registered hypotheses. H1 PASSES (ρ=+0.44). H2 FAILS (AUROC≈0.47, pre-registered null). H3 PASSES (GRACE beats textual baselines, CI excludes zero).

**§5 Analysis** — Cross-dataset (BBH + MMLU both pass H1). Mechanistic dissociation (GRACE predicts truncation-AOC strongly, mistake-AOC null, CI excludes zero: the two faithfulness axes are mechanistically distinct). k-sweep ablation (k=64 is not cherry-picked). Qualitative examples + Neuronpedia feature interpretations.

**§6 Discussion** — What GRACE measures (CoT necessity) vs. what it doesn't (error detection, sycophancy). Limitations (single model, no causal intervention). Future work (cross-model, feature ablation, mistake-faithfulness metric).

---

## Research Integrity Notes

- GRACE has **no learned parameters** — parameter-free metric. The 60/40 train/test split was inherited from an earlier pipeline variant and serves as a consistency check, not a confirmatory holdout in the statistical learning sense.
- All primary numbers on **full 143-item set** with explicit method-discovery caveat (GRACE's design was informed by observing failure of naive comparison on this data).
- Test-split numbers (n=50) reported as directional verification — they agree with full-set in sign and magnitude.
- H2 was **pre-registered** and is reported as a pre-registered null. Do not omit it.
- Edit-distance confound fully documented in `scripts/16_robustness_checks.py`. Do not claim regime-level GRACE_div differences as mechanistic without controlling for edit distance.
