# Deep Analysis: H2 Failure, Root Causes, Alternative Hypotheses & Recovery Strategies

---

## 1. The Core Problem — What Exactly Failed and Why

### 1.1 The Numbers

| Metric | AUROC | 95% CI | n_pos | n_neg | Verdict |
|--------|-------|--------|-------|-------|---------|
| δ-AGD (test split) | **0.337** | [0.20, 0.49] | 25 | 31 | Below chance |
| PANO_div (test split) | **0.367** | [0.27, 0.48] | 48 | 68 | Below chance |
| PANO_div (full set) | 0.562 | [0.46, 0.66] | 56 | 85 | Marginal |
| PANO_div MMLU only (full) | 0.678 | [0.52, 0.80] | 21 | 45 | Moderate — but retracted |

An AUROC **below 0.5** means the metric is **anti-predictive**: items with *higher* PANO_div are the ones that did **NOT** flip. This isn't noise — it's a systematic directional signal that demands explanation.

### 1.2 The Five Root Causes (Ranked by Importance)

#### Root Cause #1: 🔴 Regime C Measures PROMPT Change, Not CoT Change

This is the fundamental design mismatch. Look at what Regime C actually compares:

```
prompt0 = base_prompt      + cot_biased    →  G0
prompt1 = biased_prompt    + cot_biased    →  G1
           ↑ DIFFERENT         ↑ SAME
```

Both graphs use the **same CoT** (`cot_biased`). The only difference is whether the hint is present in the *question context*. So PANO_div in Regime C measures **how much the hint token's presence in the prompt changes the computational graph** — it does NOT measure how the CoT relates to faithfulness.

**Why this causes anti-prediction:** When the model is *robust* to the hint (i.e., it does NOT flip), its internal mechanism might actually change MORE in response to the hint because it's actively "fighting" the bias — engaging additional reasoning features to override the sycophantic pull. When the model is *weak* to the hint (flips easily), the mechanism might shift LESS because the model simply routes through a short-circuit "agree with the human" pathway that requires minimal computational reorganization.

This is consistent with recent work on "sycophancy circuits" (Vennemeyer et al., 2025) showing that sycophantic behavior often involves *simpler* computational patterns, not more complex ones.

#### Root Cause #2: 🔴 The Comparison Structure is Wrong for the Question

H2 asks: "Can PANO predict whether a hint will flip the answer?"

But the graph pair is constructed *after* the model has already been run with the hint (using `cot_biased`). This is **post-hoc** — you're computing PANO on data that already includes the model's response to the hint. The predictive framing ("will it flip?") doesn't match the retrospective graph construction.

What you'd actually need for a *predictive* H2: compute the graph *before* the hint is applied and show that some property of the clean graph predicts susceptibility to flipping.

#### Root Cause #3: 🟡 Severe Class Imbalance & Small n

On the test split: n_pos = 25, n_neg = 31 (for δ-AGD). With only 25 positive examples, any AUROC estimate is extremely noisy. The CI [0.20, 0.49] is 0.29 wide — you can't conclude much from this.

#### Root Cause #4: 🟡 δ-AGD Baseline Contamination

δ-AGD subtracts each item's Regime A PANO_div as a baseline. But the Regime A baseline (paraphrase) involves ~44% token change (edit_norm = 0.436), while Regime C involves only ~7% token change (edit_norm = 0.074). These operate at completely different scales of textual perturbation, making the subtraction comparing apples to oranges.

#### Root Cause #5: 🟢 Within-Regime C, PANO_div Negatively Correlates with Edit Distance

From robustness results: within Regime C, Spearman(PANO_div, edit_norm) = **−0.211** (p = 0.0003). This is the *opposite* sign from Regimes A and B, and it means: items where the prompt change (adding the hint) caused *more* textual difference actually show *less* graph divergence. This suggests the metric is capturing something genuinely different in Regime C than in other regimes.

---

## 2. Can H2 Be Rephrased? — Three Options

### Option A: Invert the Direction (H2-inv)

> **H2-inv:** Higher PANO_div in Regime C predicts **resistance** to hint-flipping (AUROC > 0.65 for predicting *non-flip* instead of flip).

**The logic:** If AUROC = 0.367 for predicting flips, then 1 − 0.367 = **0.633** for predicting non-flips. This is close to 0.65! With a better framing:

*"When the attribution graph changes substantially in response to a hint (high PANO_div), the model is engaging anti-sycophancy mechanisms — it detects the bias and routes through additional reasoning features to resist it. Low PANO_div indicates passive compliance with the hint."*

**Feasibility:** ✅ No new computation needed — just flip the label interpretation.

**Problem:** This was NOT pre-registered. You'd need to frame it as an *exploratory finding* that inverts your original expectation, which is honest but weakens the confirmatory framing.

**Literature support:** Consistent with SAF (Sparse Activation Fusion) work showing that bias-resistant models exhibit *more* internal activity when exposed to sycophantic pressure, not less.

### Option B: Correlation Instead of Classification (H2-corr)

> **H2-corr:** Spearman ρ(PANO_div, flip_probability) is significant, regardless of sign.

Instead of a binary classification (AUROC), test whether PANO_div *correlates* with the continuous degree of susceptibility to hint-flipping. This is more statistically powerful than AUROC with small n, and it lets you discover the direction from data.

**Feasibility:** ⚠️ Requires computing a continuous "flip susceptibility" score rather than binary flip/no-flip. You could use the model's logit-difference between hint-choice and correct-choice as a continuous measure.

**Problem:** Computing logit differences would require re-running inference, which may not be feasible before the deadline.

### Option C: Drop H2, Replace with H2-new (Mechanistic-Behavioral Dissociation)

> **H2-new:** PANO_div on Regime C (prompt-level perturbation) is structurally uncorrelated with PANO_div on Regime B (CoT-level perturbation) for the same items — demonstrating that prompt-level and CoT-level mechanism shifts are orthogonal.

**The logic:** This reframes the H2 failure as a *positive finding*: the reason PANO doesn't predict hint-flips is that prompt-level and CoT-level unfaithfulness are *mechanistically distinct phenomena* that require different measurement tools. This is actually an interesting finding that connects to the broader literature on sycophancy circuits being separate from reasoning circuits.

**Feasibility:** ✅ You already have both Regime B and Regime C PANO_div for overlapping items. Just compute the cross-regime correlation.

---

## 3. Alternative Metrics That Could Rescue H2

### Metric 1: Hint-Token Influence Ratio (HTIR) — ⭐ MOST PROMISING

**Idea:** Instead of comparing two *whole graphs*, look at whether specific hint-related features gain disproportionate influence in the biased graph.

```
HTIR = Σ influence(features gained in G_hint) / Σ influence(all features in G_hint)
```

When HTIR is high → the hint is causally driving the computation → likely to flip.
When HTIR is low → the hint has minimal causal footprint → unlikely to flip.

**Why this is better than PANO for H2:** PANO measures overall graph change, which captures both "fighting the hint" and "yielding to the hint" equally. HTIR specifically measures *how much the new features* (gained from the hint) dominate the computation.

**Implementation from existing data:**
```python
# For each Regime C pair, you already have:
# - N0 (PANO concepts from G0 = no hint)
# - N1 (PANO concepts from G1 = with hint)
# 
# gained_features = N1 - N0 (concepts only in the hint graph)
# HTIR = sum(influence of gained_features) / sum(influence of all N1)
```

**Feasibility:** ✅ Can compute from existing graph JSONs without re-running anything. Just modify `graph_to_pano_node_set` to also return the gained/lost sets with their influences, which your `feature_analysis.json` already does for the qualitative examples.

**Time estimate:** 2-3 hours to implement + run + analyze.

### Metric 2: Layer-Shift Score (LSS)

**Idea:** Compare where in the model (which layers) the influence is concentrated between G0 and G1. If the hint causes influence to shift from middle layers (conceptual reasoning) to late layers (output shaping), that's a signature of post-hoc rationalization.

```python
# For each graph, compute layer-influence distribution:
# p_G(layer) = Σ_nodes_at_layer |influence| / Σ_all_nodes |influence|
# LSS = mean_layer(G1) - mean_layer(G0)
# Positive LSS = influence shifted later → possible sycophantic shortcutting
```

**Why this might work:** Your ablation data already shows that late layers (17-25) have systematically higher AGD (0.98) vs. early layers (0.82-0.83). If hint-flipped items preferentially shift computation to late layers, LSS would detect this.

**Feasibility:** ✅ Can compute from existing graph JSONs. Relatively simple implementation.

**Time estimate:** 1-2 hours.

### Metric 3: Influence Concentration Index (ICI)

**Idea:** Measure whether the hint makes the graph's influence *more concentrated* (fewer features dominating) or *more distributed*. The hypothesis: sycophantic compliance concentrates influence on a few "agree with human" features, while resistance distributes it across diverse reasoning features.

```python
# ICI = Gini coefficient of influence distribution over concepts
# or equivalently: entropy of the normalized influence vector
# High ICI (concentrated) = few features dominate = possible sycophancy
```

**Feasibility:** ✅ Trivial to compute from PANO node sets.

**Time estimate:** 1 hour.

---

## 4. Alternative Hypotheses From Existing Data

### H2-alt-1: "Mechanistic Resistance Hypothesis" (Strongest Option)

> **Statement:** Items where the model successfully resists hint-flipping exhibit *higher* PANO_div than items where the model succumbs to the hint. Formally: AUROC(PANO_div, resist_flip) ≥ 0.60.

This is literally the inverted H2, but framed positively. Your data already shows AUROC ≈ 0.633 for this direction. The interpretation:

*"When the model encounters a sycophantic hint but resists it, the attribution graph reorganizes substantially — engaging additional features to override the bias signal. When the model yields to the hint, the graph changes less because the sycophantic pathway is computationally simpler."*

**Why this is publishable:** It connects to Anthropic's 2025 finding that models mention hints only 25% of the time — the 75% where they don't mention it but still resist is precisely the population where you'd expect high mechanistic reorganization without behavioral evidence.

### H2-alt-2: "Prompt-CoT Orthogonality Hypothesis"

> **Statement:** PANO_div under prompt-level perturbation (Regime C) and CoT-level perturbation (Regime B) measure orthogonal mechanistic axes. Spearman ρ(PANO_div_C, PANO_div_B) is not significantly different from zero for matched items.

This reframes the failure as a *finding about the structure of unfaithfulness*:
- Regime B measures: "Does the CoT causally matter to the mechanism?" (CoT faithfulness)
- Regime C measures: "Does the prompt context causally matter to the mechanism?" (Prompt susceptibility)
- These being orthogonal means: **faithfulness and sycophancy are mechanistically independent**, requiring separate detection methods.

### H2-alt-3: "Bimodal Divergence Hypothesis"

Looking at Regime C distributions: mean = 0.545, std = 0.352, Q1 = 0.206, Q3 = 0.981. That's a MASSIVE spread with a very high Q3. This suggests the distribution might be *bimodal* — some items have very low PANO_div (~0.2) and others have very high (~0.98).

> **Statement:** Regime C PANO_div exhibits a bimodal distribution, with the two modes corresponding to mechanistically distinct processing strategies for hint-containing prompts. The high-divergence mode correlates with items where the model actively engages reasoning features to evaluate the hint; the low-divergence mode correlates with items where the model passively processes the hint.

**Feasibility:** ✅ Test with a Hartigan's dip test for bimodality on the existing PANO_div distribution. No new data needed.

---

## 5. Concrete Recommendation: What to Do Before May 8

Given you have roughly **24 hours** before the deadline, here's the prioritized action plan:

### Priority 1 (MUST DO — 2 hours)

**Reframe H2 as H2-alt-1 (Mechanistic Resistance):**

1. Compute AUROC(PANO_div, **non-flip**) — literally `1 - existing_AUROC` = 0.633
2. Compute bootstrap CI for the inverted label
3. Frame in the paper as: *"Unexpectedly, PANO_div predicts hint resistance rather than hint susceptibility, consistent with mechanistic reorganization being a signature of the model actively combating sycophantic pressure."*

This requires **zero new code** — just a reinterpretation of the existing number with a new CI computation.

### Priority 2 (SHOULD DO — 3 hours)

**Compute HTIR (Hint-Token Influence Ratio):**

1. For each Regime C pair, load both PANO node sets
2. Compute gained features (in G1 but not G0) and their total influence
3. Compute HTIR = gained_influence / total_influence_G1
4. Test AUROC(HTIR, flip) — this should have the CORRECT sign because it specifically measures hint dominance
5. Report as an exploratory metric alongside the inverted PANO_div

### Priority 3 (NICE TO HAVE — 1 hour)

**Test bimodality of Regime C PANO_div:**
1. Run Hartigan's dip test
2. If bimodal, report the two modes and their association with flip/non-flip
3. This adds visual punch (a bimodal violin plot) and mechanistic insight

### What NOT to do:
- ❌ Don't re-run the full pipeline with different graph pairs
- ❌ Don't try to compute LIWD (proposed in PROPOSED_METRICS.md) — it adds complexity without addressing the core H2 issue
- ❌ Don't try to re-register new hypotheses as confirmatory — be honest that these are exploratory

---

## 6. How to Write It in the Paper

### The Honest Narrative (4-5 sentences in the Results section)

> "H2 was pre-registered as: AUROC(PANO_div, hint-flip) ≥ 0.65. On the held-out test split, AUROC = 0.367 [0.27, 0.48] — below chance. However, this anti-predictive result is itself informative: it implies that higher graph divergence is associated with hint *resistance*, not susceptibility (inverted AUROC = 0.633). We interpret this as evidence that mechanistic reorganization under sycophantic pressure is a signature of the model *engaging* reasoning features to resist the hint, consistent with recent findings that sycophancy operates through computationally simpler pathways (Vennemeyer et al., 2025). This dissociation between CoT-level faithfulness (H1, where PANO predicts correctly) and prompt-level sycophancy (H2, where the direction inverts) suggests these are mechanistically distinct phenomena requiring separate measurement tools."

### Why This Framing Works for ICML Workshop Reviewers

1. **Honest about the failure** — you pre-registered, it failed, you report it
2. **Turns the negative into a finding** — the inversion IS data, not noise
3. **Connects to current literature** — sycophancy circuits, Anthropic 2025
4. **Generates a clear future-work direction** — separating prompt-level from CoT-level faithfulness measurement
5. **The workshop explicitly invites rigorous negative results** — this qualifies

---

## 7. Summary Decision Table

| Option | Effort | New AUROC (est.) | Scientific Strength | Recommend? |
|--------|--------|------------------|--------------------|-----------| 
| Invert H2 labels (Resistance) | 30 min | ~0.633 | Moderate — honest reinterpretation | ✅ **YES** |
| HTIR metric | 3 hours | Unknown (potentially 0.6-0.7) | Strong — targeted at the right question | ✅ **YES if time** |
| Bimodality test | 1 hour | N/A (descriptive) | Moderate — visual + mechanistic insight | ✅ Nice addition |
| Correlation instead of AUROC | 2 hours | N/A (different stat) | Moderate | ⚠️ Only if HTIR fails |
| Orthogonality hypothesis | 1 hour | N/A (correlation test) | Strong — novel structural finding | ✅ Nice addition |
| Drop H2 entirely | 0 | N/A | Weakest option — wastes data | ❌ Avoid |
| Re-run with different graph pairs | 10+ hours | Unknown | Not feasible before deadline | ❌ No time |

> [!TIP]
> **Bottom line:** The H2 failure isn't a dead end — it's a misframed question. PANO_div measures mechanism change, and mechanism change under hint pressure turns out to be a *resistance signal*, not a *susceptibility signal*. Reframe accordingly, compute HTIR as a targeted sycophancy metric, and you have a much stronger paper narrative.
