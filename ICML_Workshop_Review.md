# ICML 2026 Mechanistic Interpretability Workshop — Detailed Review

## Paper: "Attribution-Graph Divergence (AGD) / PANO: A Mechanism-Level Faithfulness Metric for Chain-of-Thought Reasoning"

---

> [!IMPORTANT]
> **Overall Verdict: BORDERLINE ACCEPT → LEAN ACCEPT for the workshop, but only if the paper is framed correctly.** The work has genuine contributions but also serious weaknesses. The difference between acceptance and rejection will come down entirely to framing, honesty, and narrative discipline. See §8 for the detailed conditional verdict.

---

## 1. Summary of the Submission

The authors propose Attribution-Graph Divergence (AGD), a metric that compares pruned attribution graphs (from Anthropic's circuit-tracer + Gemma Scope transcoders) across paired counterfactual conditions to quantify mechanism-level CoT faithfulness. After discovering that naive AGD saturates at ~0.97 due to position-indexed feature IDs, they introduce **PANO (Position-Agnostic Node Overlap)** — a position-stripped variant — and **δ-AGD** — a per-item differential baseline. The work tests three pre-registered hypotheses on Gemma-2-2B-it across BBH, MMLU, GSM8K, and Turpin-style hint injection datasets (~2,100 attribution graphs).

**Headline results (held-out test split):**
- **H1 (PANO_div ↔ Lanham AOC):** ρ = +0.479, p = 0.0004, n = 50 — **PASSED**
- **H2 (hint-flip prediction):** AUROC = 0.337 (δ-AGD) / 0.367 (PANO_div) — **FAILED**
- **H3 (incremental value):** Partial Spearman ρ = +0.341, p < 0.0001; but ΔR² CI crosses zero — **MIXED**

The paper also reports the discovery that the original (position-indexed) AGD produced an *inverted* H1 correlation (ρ = −0.475), which the authors trace to a position-encoding confound, and conducts edit-distance confound audits that retract several initial claims.

---

## 2. Strengths

### S1. Genuinely Novel Research Question
This is, to my knowledge, the **first attempt to operationalize attribution graphs as a quantitative faithfulness metric**. Zhao et al.'s CRV (ICLR 2026 oral) uses attribution graphs to verify *correctness*, not *faithfulness* — a clearly distinct axis. The distinction between "did the model reason correctly?" and "did the model reason *as it claims*?" is well-articulated and important. No other published work closes this exact gap.

**This is the paper's strongest selling point for the workshop.**

### S2. Methodological Contribution: The Position-Indexing Bug
The discovery that position-indexed feature IDs (`L3_P12_F1047`) make *any* graph-Jaccard metric measure token-position drift rather than mechanism change is a **genuinely useful methodological contribution**. This affects anyone using circuit-tracer for graph comparison. PANO is a clean, principled, easy-to-adopt fix. This is the kind of "save others from our mistake" contribution that the mech-interp community particularly values.

### S3. Honest Self-Correction and Retractions
The CLAUDE.md and robustness checks show a level of intellectual honesty that is refreshing:
- The "mistake anomaly" (B_mistake < A) is explicitly retracted after the edit-distance audit shows ρ_partial = 0.028, p = 0.63
- H2's initial 0.678 AUROC on MMLU (full set) is retracted after the test-split analysis collapses it
- The post-hoc nature of PANO is explicitly disclosed

This gives the paper credibility. Reviewers notice when authors fight against their own results.

### S4. Pre-Registration and Train/Test Discipline
The `prereg.md` with explicit thresholds, the 60/40 stratified item-level split, the hyperparameter tuning restricted to training half — this is above the median rigor for workshop papers. The Holm-Bonferroni correction and BCa bootstrap (5000 resamples) are appropriate.

### S5. Robustness Infrastructure
The edit-distance confound audit (`scripts/16_robustness_checks.py`) is excellent reviewer-defense infrastructure. Partial Spearman controlling for edit distance, graph size, and legacy AGD (ρ = +0.341, p < 0.0001) is a strong signal that PANO_div is capturing something beyond textual similarity.

### S6. Comprehensive Pipeline
18 scripts, clear data flow, reproducibility via `make all` — this is well-engineered for a 2-person team on a 14-day timeline.

---

## 3. Weaknesses

### W1. 🔴 CRITICAL: H2 Fails Badly — AUROC Below Chance
The hint-flip prediction result (AUROC = 0.337 for δ-AGD, 0.367 for PANO_div) is not just a null — it's **worse than random**. An AUROC below 0.5 means the metric is *anti-predictive*: higher PANO_div is associated with items that did NOT flip. This needs explanation, not just reporting. Possible interpretations:
- PANO_div reflects *resistance* to manipulation (high graph divergence when hints are present but the model resists → no flip)
- The hint-injection pairs compare biased vs. unbiased *prompts* with *different CoTs*, so PANO_div is measuring prompt-driven mechanism change, not CoT-driven change
- The Regime C sample is underpowered (n_pos=25 for δ-AGD, n_pos=48 for PANO_div)

**A reviewer will ask:** "If your metric cannot predict the most dramatic form of unfaithfulness (Turpin hint-flips), what does it actually measure?"

### W2. 🔴 CRITICAL: H1 Test Split n = 50 Is Small
The headline H1 result (ρ = +0.479) is based on only **50 items**. The 95% CI is [+0.12, +0.70] — that's a 0.58-wide interval. While the CI excludes zero and is statistically significant, this is fragile:
- The full-set result (n = 143, ρ = +0.444) is tainted by discovery overlap
- A reviewer could argue that with n = 50, a single cluster of outliers could drive the correlation

For a workshop paper, this is *acceptable* but not *comfortable*. The wide CI will draw scrutiny.

### W3. 🟡 MAJOR: PANO Was Developed Post-Hoc — Discovery vs. Confirmation Tension
The paper's credibility rests on a delicate narrative:
1. Original AGD (position-indexed) was pre-registered
2. It failed (inverted correlation, ceiling saturation)
3. PANO was developed *after* seeing the failure
4. PANO is then tested on the held-out split

This is disclosed honestly — but a skeptical reviewer will note that the held-out split was "held out" from the *original* AGD analysis, not from PANO development. The PANO fix was designed knowing what the data looked like. The test split gives *some* protection (PANO wasn't tuned on test items), but it's not a clean confirmatory design.

**Mitigation needed in the paper:** Frame PANO explicitly as "exploratory method + held-out validation" rather than "confirmatory replication." The current CLAUDE.md does this correctly; the paper must too.

### W4. 🟡 MAJOR: No Cross-Model Validation
All results are on Gemma-2-2B-it. The research plan mentions a Llama-3.2-1B robustness check — **was this run?** I don't see it in the results. Single-model results on a 2B model weaken the generalizability claim considerably. CRV (Zhao et al.) tests on multiple model families.

### W5. 🟡 MAJOR: The Edit-Distance Confound Undermines Regime-Level Claims
The robustness analysis shows:
- Paraphrase (Regime A): edit_norm_mean = 0.436
- Mistake (B_mistake): edit_norm_mean = 0.030
- Truncation (B_trunc): edit_norm_mean = 0.271
- Hint (C): edit_norm_mean = 0.074

The partial correlation of regime → PANO_div controlling for edit_norm is ρ = 0.028 (p = 0.63) — **regime has zero effect after controlling for edit distance**. This means the regime-level distributions in the paper (the violin/box plots) are *entirely* driven by how many tokens changed, not by the type of perturbation.

This is candidly disclosed, but it significantly narrows what the paper can claim. The only surviving claim is the within-Regime-B per-item H1 correlation, where edit distance is approximately constant.

### W6. 🟡 MAJOR: δ-AGD Is Acknowledged as Contaminated
The paper states: "δ-AGD inherits whatever residual edit-distance contamination the baseline has." This is honest but means δ-AGD is not a clean metric for H2. The paper effectively has only one working metric (PANO_div on H1) for one hypothesis.

### W7. 🟢 MINOR: No Feature Interpretation
The feature-level qualitative analysis shows which concepts were gained/lost, but there's no attempt to *interpret* what these features represent (e.g., by looking them up in Neuronpedia or running autointerp). For a mech-interp workshop, reviewers may expect at least a few example features identified by name/function.

### W8. 🟢 MINOR: Ablation Results Are Incomplete
The ablation file shows `null` for most H1 ρ values in the k_sweep and alpha_sweep. This suggests the ablations were run only partially or encountered errors. Incomplete ablations weaken the robustness story.

### W9. 🟢 MINOR: No Comparison with NLDD or Other Mechanistic Metrics
Ye et al. (2026) recently introduced NLDD (Normalized Logit Difference Decay) as a per-token causal faithfulness measure. Chen et al. (2025) "How does Chain of Thought Think?" use SAEs for feature-level causal analysis. The paper should at least discuss these conceptually, even if runtime comparison isn't feasible.

---

## 4. Positioning Against State of the Art

### 4.1 Competitive Landscape (as of May 2026)

| Paper | Approach | Axis | Tool | Scale | Key Result |
|-------|----------|------|------|-------|------------|
| **Zhao et al. (CRV, ICLR 2026 oral)** | Classifier on graph structural features | Correctness | circuit-tracer + transcoders | Multi-model | AUROC 92.47 on arithmetic; causal intervention |
| **Ye et al. (2026)** | NLDD — token-level logit difference decay | Faithfulness | Activation patching | Multi-model | Reasoning horizon at 70-85% chain length |
| **Chen et al. (2025)** | SAE feature-level causal study | Faithfulness | SAEs + patching | Multi-model, multi-scale | Scale threshold for CoT structural influence |
| **Anthropic (2025)** | Behavioral + internal monitoring | Faithfulness | Extended thinking traces | Claude 3.7 Sonnet, DeepSeek R1 | Models mention hints only 25-39% of time |
| **Han et al. (2026)** | Pearl front-door on reasoning traces | Faithfulness | Causal DAGs | Theoretical + empirical | Formal causal framework |
| **This paper (AGD/PANO)** | Graph-pair divergence metric | Faithfulness | circuit-tracer + transcoders | Gemma-2-2B only | ρ = +0.479 (n=50) for PANO vs AOC |

### 4.2 Honest Positioning Assessment

**Where this paper sits:** Below CRV in terms of result strength (CRV has 92.47 AUROC; this paper has ρ = 0.479 on a different task). Below NLDD in terms of multi-model validation. But on a different axis (faithfulness vs. correctness) that no other graph-based work has quantified.

**The niche:** No one else has done paired-graph comparison for faithfulness. The closest is behavioral (Lanham, Turpin) or single-graph analysis (CRV). The counterfactual structure (same item, different CoT conditions) is unique.

**The risk:** The niche is narrow enough that a reviewer might ask "is this a paper or a failed experiment with one salvaged positive?" The answer depends entirely on framing.

---

## 5. Framing Assessment: Three Possible Paper Narratives

### Narrative A: "First Mechanism-Level Faithfulness Metric" (Original Plan)
- ❌ Too strong for the results. H2 failed. H3 is mixed. Only H1 passed, on n=50, with a post-hoc metric.

### Narrative B: "Position-Indexing Pitfall + PANO Fix" (Methodological Contribution)
- ✅ **This is the strongest framing.** Lead with: "Naive graph comparison fails because of position-indexed feature IDs. Here's why, here's the fix (PANO), and here's evidence it works (H1 passes, partial Spearman survives edit-distance controls)."
- The methodological insight is clean, novel, and reusable.
- The H1 positive result becomes a validation of the fix, not the primary contribution.
- The H2 failure becomes secondary: "We demonstrate the fix on H1; hint-flip prediction remains an open challenge."

### Narrative C: "Pre-Registered Negative with Diagnosis and Partial Recovery" (Honest Mixed Result)
- ✅ Also strong, but harder to write compellingly in 4 pages.
- "We pre-registered AGD. It failed. We diagnosed the failure (position indexing). We proposed PANO. PANO partially recovered H1 but not H2. The edit-distance confound is itself a contribution."

**My recommendation: Narrative B** with elements of C for the H2 discussion.

---

## 6. Detailed Reviewer Simulation (ICML Mech-Interp Workshop)

### Reviewer 1 (Senior, Anthropic-adjacent, SAE/circuit expert)

> **Score: 5/10 (Borderline)**
>
> The paper addresses an interesting question — can attribution graphs quantify CoT faithfulness? — but the execution raises concerns.
>
> The position-indexing discovery is a genuine service to the community. Anyone doing graph comparison with circuit-tracer will benefit from knowing this pitfall exists and that PANO fixes it.
>
> However, the core results are thin: one passing hypothesis (H1, ρ=0.48, n=50) with a wide CI, one clear failure (H2), and a mixed H3. The post-hoc development of PANO weakens the confirmatory framing. The restriction to Gemma-2-2B is a limitation — this should at least be validated on one other model family.
>
> The edit-distance audit is well done and the retractions are honest. But the fact that regime-level differences disappear after controlling for edit distance is a significant blow to the paper's broader claims about mechanism-level measurement.
>
> I'd like to see: (1) interpretation of at least 3-5 features by name, (2) cross-model validation on one other model, (3) a clearer discussion of why H2 fails sub-chance.
>
> **Recommendation:** Weak accept if framed as methodological contribution; reject if framed as "we built a new faithfulness metric."

### Reviewer 2 (Safety-oriented, interested in CoT monitoring)

> **Score: 6/10 (Lean Accept)**
>
> This paper tackles a question I care about deeply: can we use interpretability tools to detect when CoT is unfaithful? The answer is "partially" — H1 passes with a moderate correlation, but H2 fails for sycophancy/hint-flip detection, which is the more safety-relevant task.
>
> The position-indexing bug discovery is valuable. The edit-distance confound analysis raises the bar for future work in this space. The pre-registration and honest retractions are commendable.
>
> I'm concerned about n=50 for the headline result and single-model validation. But for a workshop paper, this is acceptable as initial evidence.
>
> **Recommendation:** Accept as a short paper. The community needs to know about the position-indexing pitfall, and the partial H1 success justifies further investigation.

### Reviewer 3 (Methodology-focused, statistics-aware)

> **Score: 4/10 (Lean Reject)**
>
> The statistical integrity is above average (pre-registration, BCa bootstrap, Holm-Bonferroni, partial correlations). But the results don't support the paper's ambition.
>
> The headline H1 result (ρ=0.479) is on n=50 with CI [0.12, 0.70]. With this range, I cannot rule out that the true correlation is as low as ρ=0.12 — a very weak effect. The full-set result (ρ=0.444, n=143) is tainted by discovery-phase overlap.
>
> H2 is below chance, which is actively problematic. δ-AGD is acknowledged as contaminated. The paper effectively has one metric, one hypothesis, one model, and a small sample.
>
> The position-indexing insight is a useful note, but PANO is a trivial fix (strip the `_P\d+_` from the string) — it's not clear this rises to the level of a primary contribution.
>
> **Recommendation:** Reject. Needs cross-model validation and a larger held-out sample to be convincing.

### Expected Meta-Review Decision
With a 5, 6, and 4, the paper falls in the **borderline** zone. Workshop acceptance rates for ICML mech-interp tend to be 30-50%. This paper would likely be in the discussion pile, not an automatic accept or reject.

---

## 7. What Would Strengthen the Paper for Submission

### 7.1 Must-Do (Before Submission)
1. **Frame as Narrative B** — Lead with the position-indexing pitfall and PANO as a methodological contribution
2. **Explain H2 sub-chance explicitly** — Don't just report it; hypothesize why higher PANO_div predicts non-flips (e.g., high graph divergence reflects the model *resisting* the hint, not succumbing to it)
3. **Report the full-set H1 clearly labeled as discovery** — Don't hide it, but clearly mark it as sensitivity analysis
4. **Cite NLDD (Ye et al. 2026) and CRV's ICLR acceptance** — Position against these explicitly

### 7.2 Should-Do (If Time Permits Before May 8 AOE)
1. **Interpret 3-5 features** — Use Neuronpedia or autointerp to give names to the top gained/lost features in the qualitative examples
2. **Fix the ablation nulls** — The h1_rho nulls in the k_sweep and alpha_sweep look like bugs, not true nulls
3. **Add a paragraph on why PANO might not work for H2** — Regime C involves *prompt* changes (biased vs. unbiased), not *CoT* changes; PANO measures CoT mechanism shift, which is the wrong axis for hint detection

### 7.3 Would-Be-Nice (For a Stronger Submission)
1. **Cross-model validation** on Llama-3.2-1B or Qwen (even 50 items)
2. **LIWD metric** from your proposed metrics doc — it's independent and could add a second dimension
3. **Direct comparison with a simple baseline** like activation-cosine on the same H1 task — does PANO beat it?

---

## 8. Final Verdict

### Is This Publishable at ICML 2026 Mech-Interp Workshop?

**Conditional YES — with the right framing.**

| Aspect | Assessment |
|--------|-----------|
| **Novelty** | ✅ First paired-graph faithfulness metric; position-indexing bug discovery |
| **Significance** | ⚠️ Moderate — one passing hypothesis, narrow results |
| **Soundness** | ✅ Pre-registration, proper statistics, honest retractions |
| **Clarity** | ⚠️ Depends on framing; current docs have mixed narratives |
| **Reproducibility** | ✅ Excellent pipeline, seeds, make targets |
| **Workshop Fit** | ✅ Explicitly in scope: "rigorous negative results," "standard empirical work," methodological contributions |

### What Tips the Balance

**In favor of acceptance:**
- The workshop *explicitly invites* rigorous negative results and replications/critiques
- The position-indexing pitfall is a genuine community service
- The partial Spearman surviving edit-distance controls (ρ = +0.341, p < 0.0001) is real signal
- The pre-registration and retractions demonstrate scientific maturity
- This is non-archival — the bar is appropriately lower than ICLR/NeurIPS main

**Against acceptance:**
- H2 fails badly (below chance)
- H1 is n=50 with wide CI
- Single model (Gemma-2-2B)
- PANO was developed post-hoc
- No feature interpretation
- CRV (which uses the same tools) was accepted as an oral at ICLR — comparison will be unfavorable in reviewers' minds

### Bottom Line

> [!TIP]
> **If you frame this as "we discovered a critical pitfall in graph comparison (position indexing), built a fix (PANO), and show it recovers a moderate faithfulness signal that survives edit-distance controls — but hint-flip prediction remains unsolved" — this is an honest, useful, workshop-grade contribution.**
>
> If you try to frame it as "we built a new faithfulness metric that works" — reviewers will focus on H2's failure and the narrow H1 evidence, and you risk rejection.

### Probability Estimate

| Framing | P(Accept) |
|---------|-----------|
| Narrative B (methodological) | **55-65%** |
| Narrative C (honest mixed) | **40-50%** |
| Narrative A (strong positive) | **15-25%** |

---

## 9. Critical Issues That Could Sink the Paper

> [!CAUTION]
> These are the landmines to avoid:

1. **Do NOT claim regime-level PANO_div differences are mechanistic.** The robustness check kills this (ρ_partial = 0.028 after edit-distance control).
2. **Do NOT report the full-set H1 (n=143) as the primary result.** It overlaps with PANO development data.
3. **Do NOT hide the H2 sub-chance result.** Report it, explain it, and frame the open question.
4. **Do NOT compare favorably with CRV's numbers.** CRV measures correctness with AUROC 92.47; you measure faithfulness with ρ 0.479. These are different tasks. A reviewer who thinks you're claiming superiority to an ICLR oral will not be kind.
5. **Do NOT claim PANO is "confirmatory."** It was developed post-hoc. The test-split provides validation, not confirmation.

---

## 10. One-Paragraph Paper Pitch (If I Were Writing the Abstract)

> Attribution graphs — pruned computational traces from circuit-tracer and Gemma Scope transcoders — offer a natural substrate for measuring whether an LLM's chain-of-thought faithfully reflects its internal reasoning. We discover that naive graph comparison metrics are dominated by a **position-indexing artifact**: feature IDs that encode token position cause any Jaccard-style metric to measure sequence-length drift rather than mechanism change. We introduce **PANO (Position-Agnostic Node Overlap)**, which strips position before comparison, and show it recovers a moderate positive correlation with Lanham's behavioral faithfulness metric (Spearman ρ = +0.48, p < 0.001, held-out split) — a signal that survives controlling for edit distance, graph size, and legacy metrics (partial ρ = +0.34, p < 0.0001). However, PANO does not predict Turpin-style hint-induced answer flips (AUROC = 0.37), suggesting that mechanism-level and prompt-level unfaithfulness are distinct phenomena requiring different tools. We release our pipeline and edit-distance audit as resources for future graph-comparison research.

---

## Appendix: Workshop Alignment Check

The ICML 2026 Mech-Interp Workshop explicitly welcomes:
- ✅ "Standard empirical work" — yes
- ✅ "Rigorous negative results" — H2 qualifies
- ✅ "Open-source software and tools" — the pipeline is releasable
- ✅ "Critiques/failed replications of past work" — the position-indexing finding is a "critique" of naive graph comparison

The deadline is **May 8, 2026 AOE**. You are within the window.
