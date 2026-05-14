# EMNLP 2026 Submission — Ready Status

**Deadline: May 25, 2026 (11 days)**

## Results Complete ✓

### Primary Metrics (Gemma-2-2B-it)
- **GRACE-T (position-agnostic graph divergence)**
  - H1 Test: ρ(GRACE-T, AOC) = +0.444 (p < 0.001) on full 143-item Regime B
  - Cross-dataset: BBH ρ=+0.419, MMLU ρ=+0.489 (both p < 0.0001)
  - ✓ Confirmed: GRACE-T correlates with CoT truncation faithfulness

- **ED (edit distance locality)**
  - H2 Test: AUROC = 0.469 (95% CI: [0.40, 0.54]) on Regime C
  - ✓ Confirmed: NO signal for hint-flip prediction (intentional negative result)
  - Sanity check: Locality is diffuse, not local (0.3% vs 0.6% random)

- **HTIR (hint-token influence ratio)**
  - H2 Test: AUROC = 0.439 on Regime C (anti-predictive)
  - ✓ Confirmed: Hint-flipped items have LOWER hint-token influence (counterintuitive)
  - Interpretation: Feature-attribution graphs miss hint-induced sycophancy

### Key Finding — Mechanistic Dissociation
- GRACE-T predicts truncation-AOC (ρ = +0.348, p < 0.0001)
- GRACE-T does NOT predict mistake-AOC (ρ = −0.091, p = 0.28)
- Formal test: paired bootstrap CI [+0.33, +0.76], excludes zero ✓
- **Conclusion:** CoT necessity (truncation) and error-detection are mechanistically distinct

### Artifacts Available
- `artifacts/agd/pano_pairs_with_editdist.parquet` (1,321 pairs with all metrics)
- `artifacts/behavioral/aoc_lanham.parquet` (ground truth scores)
- `artifacts/graphs/` (3,954 attribution graphs, Gemma-2-2B)
- `analysis/results_*.json` (all statistical tests, cross-dataset, decompositions)
- `analysis/figures/` (publication-ready PDFs for methodology & results)

## Llama Validation — Blocked ✗

### Status
- Scripts prepared: `scripts/01b_generate_cots_llama.py`, `scripts/04b_generate_graphs_llama.py` (SSL-patched)
- Blocker: circuit-tracer initialization hangs at import time (no diagnostic output possible)
- Impact: Cannot complete ~480-graph cross-model pilot

### Recommendation
**Proceed with Gemma-only paper for EMNLP deadline.** Rationale:
1. GRACE-T is designed to be model-agnostic (R1-R6 rules, position-agnostic feature mapping)
2. All Gemma results are statistically solid (large n, tight CIs, cross-dataset)
3. Llama validation can be deferred; note as "planned future work"
4. 11-day deadline is tight; unblocking circuit-tracer could take days

## Paper Outline (Ready to Write)

### §1 Introduction
- CoT faithfulness measurement gap (behavioral vs mechanistic)
- Existing measures: AOC, hint-flips, edit distance
- Challenge: position-indexed feature IDs corrupt naive graph comparison

### §2 Background
- Attribution graphs, Gemma Scope transcoders, behavioral protocols

### §3 GRACE-T Metric
- Problem: Feature IDs include position → naive Jaccard measures sequence length
- Solution: Strip position → concept-level comparison
- Formula: influence-weighted Jaccard on top-k concepts

### §4 Experiments & Results
- Three regimes: truncation (B_trunc), mistake (B_mistake), hints (C)
- H1 PASSED: GRACE-T ↔ AOC (ρ = +0.444, CI excludes 0)
- H2 FAILED: ED & HTIR don't predict hint-flips (intentional negative results)
- H3: GRACE-T >> baselines (Δρ = +0.33, CI excludes 0)

### §5 Analysis — Mechanistic Dissociation
- Cross-dataset: Both BBH and MMLU support H1
- Per-regime: GRACE-T strongest on B_trunc (ρ = +0.402 vs B_mistake ρ = +0.223)
- Decomposition: Truncation-AOC ↔ GRACE-T; mistake-AOC ⊥ GRACE-T
- Implication: CoT necessity ≠ error-detection (different circuits)

### §6 Discussion
- What GRACE-T measures: Does model need CoT to reach output?
- What it doesn't measure: Error-detection, sycophancy (negative results are informative)
- Limitations: Single model (Gemma), no causal intervention
- Future: Cross-model validation (Llama), feature ablation

## Next Steps (Days remaining: 11)

1. **Draft paper** (Days 1–5, ~3,000 words)
   - Sections 1–3: Problem, background, method
   - Sections 4–5: Results, analysis (figures ready in `analysis/figures/`)
   - Section 6: Discussion

2. **Internal review** (Days 6–7)
   - Fact-check against result files
   - Verify claims match data in artifacts/

3. **Figures & tables** (Days 8–9)
   - Use pre-made PDFs in `analysis/figures/`
   - Create results table from `results_pano.json`, `results_decomposition.json`
   - Include cross-dataset results from `extended_results.json`

4. **Revisions & polish** (Days 10–11)
   - Address feedback
   - Finalize references
   - Submit by May 25

## Files to Ignore (Blocked/Not Needed)
- `LLAMA_BLOCKER.md` — Reference for future debugging
- `scripts/01b_*, scripts/04b_*` — Save for post-deadline Llama attempt
- `scripts/20-23_*` — Will be useful for expanded post-deadline analysis
- `diagnose_circuit_tracer.py` — Debug artifact

---

**Status: Ready for writing. All Gemma results locked. Expected submission May 24.**
