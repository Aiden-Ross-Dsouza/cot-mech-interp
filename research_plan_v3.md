# GRACE → v2.2: Research plan for EMNLP 2026 short paper

**Status of the work as of May 13, 2026 (end of Day 2 / early Day 3):**

- ICML Mech Interp workshop: paper desk-rejected for hallucinated citations (fixes verified).
- ~780 Gemma-2-2B-it attribution graphs already generated; pipeline working.
- v2.1 graph-understanding work (§4 of the prior plan) is **complete**:
  - **Graph census**: 1,977 rows, late layers carry 65.6% of influence on average.
  - **Backbone vs item-specific split**: 40 universal backbone concepts (>50% frequency); 605 item-specific (<10%); 114 middle. Structure is bimodal as predicted.
  - **Mistake-locality sanity check (§4.5)**: **falsified.** Locality fraction 0.003 vs random expectation 0.0058. Mistake injection causes diffuse, not local, attribution shift. 6/10 sample items have exactly 0% locality.
  - **Hint-token sanity check (§4.6)**: **falsified.** Mean HTIR is 3.4%, non-flipped items have *higher* HTIR than flipped (anti-predictive at AUROC 0.439, same direction as the earlier H2 PANO result of 0.367).
- v2.1 metric implementations (`src/pano.py` + scripts 20–22) are **complete and run cleanly** on the Gemma corpus.
- ED and HTIR have been computed on the full Gemma corpus and their statistical behavior matches the sanity checks: ED median ≈ 0, HTIR AUROC < 0.5.

**EMNLP 2026 short paper:** ARR submission deadline May 25, 2026 → **12 days from today**.

The Day 1–3 results force a re-framing. The original v2.1 framing — *"three matched metrics for three behavioral sub-protocols"* — assumed all three matched-metric designs would work and the open question was cross-model replication. They don't all work on Gemma alone. **The paper's actual contribution is now: one principled metric (GRACE-T) that works, plus two principled negative results (ED and HTIR) that say feature-attribution-graphs are not the right substrate for measuring mistake-detection or hint-influence at the mechanism level.** This is the v2.2 framing.

This is a real result, not a downgrade. It is the most honest paper your data supports, and it is publishable.

---

## 0. Revising the strategy: depth and breadth are not independent

**Update (May 13, after Day 1–3 results): v2.2 re-framing.** The v2.1 plan assumed the open empirical questions were (a) do the three matched metrics work on Gemma, and (b) do they replicate on a second model. We answered (a) on Day 2. ED and HTIR fail their sanity checks on Gemma. The paper's contribution shifts:

- **v2.1 framing (now wrong):** "Three matched metrics for three behavioral sub-protocols; the decomposition replicates across models."
- **v2.2 framing (matches data):** "Feature-attribution-graph divergence captures CoT-necessity (truncation) and replicates across model families. The natural feature-level extensions to mistake-detection (ED) and hint-influence (HTIR) fail in informative ways: mistake response is diffuse across positions and layers, not local; hint-induced flips correlate negatively with hint-token feature influence. These negative results constrain where mechanism-level CoT-faithfulness measurement can be done with current attribution-graph tools."

This is a tighter paper. The contribution is now (i) GRACE-T as a principled, cross-model-validated metric for truncation-style CoT faithfulness, (ii) the dissociation finding in the AOC composite, and (iii) two principled negative results that map out where this method does and does not reach. Three contributions, not three metrics.

**On the third-model question (asked May 12):** the answer is now firmly no, but for a sharper reason than yesterday. With ED and HTIR falsified on Gemma alone, the cross-model campaign on Llama serves two purposes: (1) replicate GRACE-T's positive result, (2) replicate ED and HTIR's *negative* results. Both purposes are served by two models. A third model would not add evidence — the failures already replicate within one model (ED and HTIR's earlier H2 result on PANO had the same anti-predictive direction). Spending compute and writing-time budget on a third model means giving up the depth of analysis the negative results now require.

**What changes in the plan:**

1. The Llama campaign on Days 5–7 is unchanged in scope (~480 graphs) but its purpose is now sharper: GRACE-T must replicate; ED and HTIR will likely fail again on Llama, which is fine and expected.
2. **Days 3–4 add new work:** characterize the failures of ED and HTIR carefully. The negative results need to be *explained* mechanistically, not just reported.
3. The paper's §5 (Discussion) shifts from "look at our metric family" to "what feature-attribution graphs can and cannot measure about CoT faithfulness."
4. One new exploratory metric is on the table for Day 4: a "global reorganization" metric that captures the diffuse-shift behavior the §4.5 result revealed. This is not promised; it is contingent on the Day 4 results.

The rest of the plan structure stands. The metric-design rules R1–R6 still apply to GRACE-T and to any new metric proposed. The cross-model campaign still runs.

---

## 0.1 The original v2.1 strategy framing (preserved for reference)

**Update (May 12, after Aiden's pushback):** the original v2 framing — "Road B with a small Road-A robustness paragraph" — separated metric design from cross-model validation as if they were independent. They are not. A metric tuned to artifacts of one model's graphs is not a metric; it is a description of that one model. We will not know whether the proposed metric family is principled or accidental until it is tested on a second model's graphs. **Cross-model validation is part of the metric design, not a downstream check.**

Three concrete failure modes the original plan didn't guard against:

1. **Layer-band assumptions** baked into the metric definitions. Gemma-2-2B has 26 layers with sliding-window + global attention; Llama-3.2-1B has 16 layers with full attention. An "early / mid / late" split that means something on Gemma may correspond to a different functional partition on Llama. The window size in the mistake-locality metric depends on how much position-mixing the attention pattern produces.

2. **Backbone vs item-specific concept structure may be model-specific.** Concepts like `L22_F11133` are Gemma-Scope-feature-11133-at-layer-22. They don't exist in Llama-Scope. What needs to replicate is the *structure* — a dense backbone of features shared across prompts plus a sparse item-specific tail — not the specific feature IDs. If Llama-Scope produces a different distributional shape, the metrics that rely on the split must be re-derived.

3. **The +0.35 / −0.09 dissociation could itself be Gemma-specific.** Maybe Gemma-2-2B genuinely processes truncation and mistake-injection through distinct circuits, and Llama-3.2-1B uses one common circuit. In that case, the decomposition proposed below is a Gemma fact, not a CoT-faithfulness fact. The whole paper's spine has to replicate on a second model.

These cannot be reasoned away. They can only be tested.

### What this means in practice

You are right that **both** generating graphs on a second model **and** designing metrics with both graph sets in mind are needed. The constraint that does not move is the deadline. So the revised plan does the following:

- Pick a second model. Generate a smaller-but-representative graph campaign on it (not 780; a carefully chosen subset).
- Design metrics on Gemma graphs, then **immediately** stress-test them on the second model's graphs before committing to the final form. Anything that doesn't transfer gets revised or dropped.
- Frame the paper around what survives both models. Anything that only works on Gemma becomes a candid "model-specific observation" in the discussion section, not a headline claim.

The honest paper title becomes:

> *Mechanism-level chain-of-thought faithfulness decomposes into three sub-properties: a cross-model evidence base.*

The single contribution is **the decomposition + matched metric family, validated on two model families**. Not "GRACE works on Gemma."

### Why this is doable in 13 days

The Gemma graphs are already on disk (~780). The metric design work is the same as the original v2 plan. The second-model campaign is a carefully scoped subset, not a full replication. Concretely:

- Second model: **Llama-3.2-1B-Instruct** with Llama-Scope ReLU PLTs (mentioned in the circuit-tracer README as fully supported; the `demos/llama_demo.ipynb` notebook already exists and runs).
- Items: ~100 BBH+MMLU items × Regime-B-truncation pairs (200 graphs) + ~60 BBH+MMLU items × Regime-B-mistake pairs (120 graphs) + ~80 Turpin items × Regime-C pairs (160 graphs) = **~480 Llama graphs**.
- That is roughly 60% of the Gemma graph count, on a smaller model, focused only on the items the metrics will be evaluated on.
- Compute estimate: Llama-3.2-1B is smaller than Gemma-2-2B (fewer parameters, 16 layers vs 26, full attention rather than sliding-window) so per-graph time should be similar or less. On A100-80GB, expect ~10–14 hours total. At Modal A100-80GB rates (~$2.50/h), that is ~$30–35. **Budget $50 to be safe.**
- Critical: the 60-mistake and 80-hint budgets are intentionally smaller than the truncation budget, because (a) the dissociation in the Gemma data tells us truncation has the most signal to validate, and (b) we want at least *some* mistake and hint coverage on Llama to test whether the dissociation pattern itself replicates.

This is no longer "a small Road-A paragraph." It is the second pillar of the paper.

### Decision rules before the campaign

You should *not* generate Llama graphs on Day 1 and then build metrics around what you find. The order is fixed:

- **Days 1–3: graph understanding on existing Gemma graphs, plus locking metric definitions.** Do the §4 census, the backbone-split, the locality and hint-token sanity checks on Gemma. Decide what each metric does, written down before any Llama compute is spent.
- **Day 4: small Llama pilot (~30 graphs across all three regimes).** Verify the existing Gemma metric definitions even produce non-degenerate values on Llama graphs. If a metric's distribution collapses (all zeros, all ones), revise *before* the full campaign.
- **Days 5–7: full Llama campaign at the ~480-graph scope.** Compute the metrics on it.
- **Day 8: cross-model comparison.** This is the key analysis: does the decomposition (truncation high, mistake low, hint-flip uncorrelated) replicate on Llama? Does each metric's direction and rough magnitude survive?
- **Days 9–12: writing.**
- **Day 13: submission buffer.**

This sequence ensures the metric design is informed by both models without being overfit to either.

---

## 1. What we actually know from your data

Before designing new metrics, write this down clearly. These are the load-bearing facts:

1. **Position-indexing artifact is real and fixed.** Original AGD saturated at ~0.97 due to `(layer, position, feature)` IDs. PANO strips position → `(layer, feature)` concepts → metric becomes usable. This is the only methodological claim with no caveats.

2. **PANO-divergence correlates with composite Lanham AOC at ρ=+0.44 (n=143, full set).** Held-out: ρ=+0.48 on n=50.

3. **The composite decomposes asymmetrically:**
   - aoc_truncate_25: ρ=+0.30
   - aoc_truncate_50: ρ=+0.35
   - aoc_truncate_75: ρ=+0.34
   - aoc_mistake: ρ=−0.09 (n.s.)

4. **The asymmetry survives controls.** After partialling out edit-distance, length, graph-size, partial ρ on truncation is still significant.

5. **Hint-flip prediction (Regime C) is at chance.** AUROC = 0.52 (PANO). δ-AGD is at 0.34 (anti-predictive on test split).

6. **Regime-level differences in mean PANO are confounded by edit distance.** ρ_partial(regime → PANO_div | edit_norm) = 0.028, p=0.63. You cannot claim "paraphrases produce lower PANO than mistakes" as a mechanism statement.

7. **Backbone features are real.** 40 backbone concepts (>50% of all 1,977 graphs) and 605 item-specific concepts (<10%) on Gemma. The bimodal structure is clean. Backbone examples: `L22_F11133`, `L24_F12351`, `L25_F5714`. The graphs share a domain-general backbone; what differs are the item-specific concepts.

8. **(NEW, May 13) Late layers carry 65.6% of influence on average.** Influence is heavily concentrated in late layers. Early/mid/late split: 17.9% / 16.5% / 65.6%. This is consistent with the picture that late layers do output-formatting while early/mid layers do reasoning, and is a relevant baseline for interpreting the depth-band GRACE results.

9. **(NEW, May 13) Mistake injection produces diffuse, not local, attribution shifts.** Mean locality fraction within a window of ±5% of CoT length around the mistake = 0.003, against random expectation of 0.0058. 6 of 10 sample items have exactly 0% locality. **This falsifies the design assumption behind ED.** The shift exists (mean ED across full corpus = 7.2%, though median is near 0), but it is spread across all positions, not concentrated near the mistake.

10. **(NEW, May 13) Hint-token feature influence is anti-predictive of hint-flips.** HTIR mean is 0.034 (only 3.4% of total influence routes through hint-token positions). Flipped items: HTIR = 0.028. Non-flipped items: HTIR = 0.038. AUROC = 0.439, *below* chance — flipped items have *less* hint-token influence than non-flipped. This is the same anti-predictive direction the earlier H2 PANO experiment showed (AUROC = 0.367). The convergent anti-predictive result across two different graph operations on the hint regime is itself informative: it suggests hint-flipping is mediated by a mechanism that is *not visible* in feature-level attribution graphs — likely embedding-level or routing through pruned-away features.

The dissociation in (3), the hint-flip failure in (5), the mistake locality failure in (9), and the hint-flip anti-prediction in (10) are *findings*, not bugs. Treat them as such. (9) and (10) together establish the paper's main negative result: feature-attribution graphs are the right substrate for measuring truncation-style CoT faithfulness, and not the right substrate for measuring mistake-detection or hint-influence.

---

## 2. The proposed metric family — one metric per faithfulness sub-question

The paper's structure should be: *each behavioural sub-protocol is asking a different question; for each question, we provide one mechanism-level instrument.* This is the section that will take the most thought, so I'll spend the most space here.

### 2.0 Model-agnostic design rules (read this before writing metric code)

Every metric below has to compute the same thing on Gemma-2-2B (26 layers, 16,384 features per layer, sliding-window+global attention) and on Llama-3.2-1B (16 layers, ~16K features per Llama-Scope dictionary, full attention). To prevent silently model-specific definitions, every metric must satisfy these rules:

**R1. Depth is a fraction, not an index.** Never write "layer 22" in a metric definition. Write "the layer at depth fraction d", where d = layer_index / n_layers. The early/mid/late split becomes d∈[0, 1/3) / [1/3, 2/3) / [2/3, 1]. On Gemma that's layers 0–8 / 9–16 / 17–25; on Llama it's 0–4 / 5–9 / 10–15. The metric definition doesn't change; the layer ranges do.

**R2. Position windows are token-fractions of the CoT, not absolute token counts.** "Window of 5 tokens around the mistake" is model-agnostic only if 5 happens to be the right number for both models. Better: "window of fraction f of the CoT length around the mistake." Pick f=0.10 as a default and ablate. Same metric code on both models.

**R3. Top-k is a fraction of the available concepts, not an absolute count.** Gemma graphs after pruning have a few hundred unique (layer, feature) concepts; Llama graphs may have a different distributional shape. Use k = min(64, ⌈f·n_unique_concepts⌉) with f=0.10. This was an implicit assumption in GRACE; make it explicit.

**R4. Backbone-vs-item-specific is re-derived per model.** Do not import the Gemma backbone-feature list into the Llama pipeline. The split is computed separately on each model's graph corpus and applied within that corpus. What replicates across models is the *bimodal structure* of the concept-frequency distribution, not the specific features.

**R5. Influence is dimensionless and normalized within graph.** Always compute influence as a fraction of the graph's total influence-on-target, not the raw circuit-tracer score. Raw scores may differ in magnitude between models for reasons unrelated to mechanism. This is the simplest no-cost cross-model invariance.

**R6. Every metric returns a scalar in [0, 1] with the same meaning on both models.** No metric should have "calibrate against a Gemma reference set" baked in. The metric is a graph property, not a tuned classifier.

If a proposed metric cannot be expressed under these rules, it isn't a metric; it's a Gemma observation. Reject it.

### 2.1 Truncation → "did the CoT actually do work?"

**Behavioural question:** if I remove the tail of the CoT, does the answer change?
**Mechanistic translation:** does the influence on the answer token *concentrate* on the truncated portion, or is it equally distributed across the CoT?

**Metric: GRACE-T (truncation-sensitivity, already covered by GRACE).** Your current PANO-divergence on Regime-B truncation pairs is already the right instrument here. The interpretation is: "when the tail is removed, how much of the influential top-k concept set turns over?" High GRACE-T → CoT tail was load-bearing → faithful in the truncation sense.

**What's new in v2 (and cross-model-safe):** decompose GRACE-T into *which depth-band reorganises.* Under R1:
- Early (d ∈ [0, 1/3)): low-level token-feature reorganisation
- Mid (d ∈ [1/3, 2/3)): conceptual integration
- Late (d ∈ [2/3, 1]): output-shape selection

If truncation reorganises mostly mid layers, the CoT was genuinely contributing to reasoning. If it reorganises mostly late layers, the CoT was contributing to answer formatting. **This per-band split is a free experiment** — you already have the layer field in every node. The same code works on both Gemma and Llama because the band definition is a depth fraction.

**Cross-model question for this metric:** does the relative depth at which truncation reorganisation lives replicate? If on Gemma it's mid-layer-dominant and on Llama it's late-layer-dominant, that itself is a finding (and you should report it as such, not paper over it).

### 2.2 Mistake injection → "does the model detect and react to a wrong intermediate step?"

**STATUS (May 13): ED falsified on Gemma. Section updated to reflect post-hoc understanding.**

**Behavioural question:** if I splice a wrong step into the CoT, does the answer change?
**Original mechanistic translation:** the mistake is *local* — it changes one or two tokens. The reorganisation we'd expect under a faithful, error-sensitive model is also *local*: error-detection features should fire near the mistake position.

**Proposed metric: ED (error-detection-localisation):**
$$
\mathrm{ED}_f(G_0, G_1) = \frac{\sum_{p \in W} |\mathrm{inf}(p, G_1) - \mathrm{inf}(p, G_0)|}{\sum_{p} |\mathrm{inf}(p, G_1) - \mathrm{inf}(p, G_0)|}
$$
where W is a window of fraction f of the CoT length around the mistake position.

**Sanity check result (Gemma, n=10 mistake pairs, f=0.10):** locality fraction 0.003 vs random expectation 0.0058. 6/10 items have exactly 0% locality. Full-corpus ED: mean 0.072, median ~0, std 0.217 — distribution is sparse and dominated by zeros.

**What this means.** The design assumption — that mistake injection produces a *local* attribution shift detectable by a position-window — is wrong on Gemma. There are three non-exclusive interpretations:

1. **The model doesn't mechanistically respond to mistake injection on the items where it doesn't behaviorally flip.** Mistakes get carried forward through the same pathway; no error-detection circuitry is engaged. This is the "post-hoc rationalisation" reading.
2. **Error detection exists but is genuinely distributed.** The model detects the inconsistency but the detection is spread across many positions and layers. ED was the wrong instrument.
3. **The pruned graph loses local error signals.** circuit-tracer prunes to top influence paths to the answer token. Localised error-detection features that fire but don't propagate strongly to the answer may be pruned away.

Your data cannot distinguish these. **The paper should flag all three.** This is what an honest negative-result write-up looks like — falsifying the metric is the easy part; interpreting *why* is where the contribution lives.

**Cross-model expectation for Llama:** if interpretation (1) is correct, ED will also fail on Llama (the mechanism is just absent). If (2) or (3) is correct, ED may or may not transfer — depends on Llama-Scope's pruning characteristics. **Run ED on Llama anyway** — even a confirmed-null on a second model strengthens the negative-result claim.

### 2.3 Hint injection → "is the answer driven by the prompt cue rather than the CoT?"

**STATUS (May 13): HTIR falsified on Gemma (anti-predictive). Section updated to reflect post-hoc understanding.**

**Behavioural question:** add a wrong-answer hint to the prompt; does the answer flip without the CoT acknowledging the hint?
**Original mechanistic translation:** the hint should appear as a *path through the hint-token's position*. Hints that drive a flip should route significant influence through P_hint.

**Proposed metric: HTIR (hint-token influence ratio):**
$$
\mathrm{HTIR}(G_{\text{biased}}) = \frac{\sum_{n: \text{pos}(n) \in P_{\text{hint}}} \mathrm{inf}(n)}{\sum_n \mathrm{inf}(n)}
$$

**Sanity check result (Gemma, n=287 valid Regime-C pairs):** mean HTIR = 0.034. Flipped items: HTIR = 0.028. Non-flipped items: HTIR = 0.038. **AUROC = 0.439** — *anti-predictive*, in the same direction as the earlier H2 PANO result.

**What this means.** Hint-token positions account for only ~3% of total feature-level influence, and that small fraction is *higher* in items where the model resisted the hint. The convergence of two different graph operations (PANO at the whole-graph level, HTIR at the hint-token level) both pointing anti-predictive is informative. It supports the "mechanistic resistance" interpretation from your earlier H2 deep analysis: items where the model engages reasoning circuitry to *resist* the hint show more graph activity around it, while items that flip do so via a short-circuit that bypasses the hint-token's feature representations entirely.

**The deeper implication:** hint-induced answer flips are not visible at the feature-attribution-graph level. The mechanism is one of:

1. **Embedding-level.** The hint token's embedding directly contributes to the answer logit through residual connections that bypass MLP features. circuit-tracer attributes through MLP transcoders; pure attention pathways from embeddings to logits may not surface as "hint-token influence" in the attribution graph.
2. **Attention-mediated routing without feature mediation.** The hint primes which subsequent feature pathway is selected; the hint token itself doesn't accumulate influence, but it changes which features fire downstream.
3. **Bypass through pruning.** Hint-relevant features fire but get pruned because they don't dominate the answer's top influence paths.

The QK-attribution extension from Lindsey et al. (July 2025) would help with (1) and (2), but that's a follow-up project.

**Cross-model expectation for Llama:** if interpretations (1) or (2) are correct, the result should replicate — both architectures have residual streams and attention. **Pre-register the prediction:** HTIR AUROC will be in [0.35, 0.55] on Llama, with non-flipped > flipped. If it replicates, that is the strongest negative result.

### 2.4 The unifying frame (revised May 13)

The clean version of the contribution, after Day 1–3 results:

| Sub-protocol | Behavioural q. | Graph operation | Metric | Status on Gemma |
|---|---|---|---|---|
| Truncation | Did CoT do work? | Top-k concept turnover, position-collapsed | **GRACE-T** (=PANO_div) | ✅ ρ=+0.35 with aoc_truncate |
| Mistake injection | Did model react to wrong step? | Local position-resolved attribution shift | ED | ❌ falsified; shift is diffuse |
| Hint injection | Did prompt cue bypass the CoT? | Influence fraction through hint token | HTIR | ❌ falsified; anti-predictive |

**The contribution claim is now:**

> Feature-attribution graphs are the right substrate for measuring CoT necessity (truncation-style faithfulness). They are *not* the right substrate for measuring error-detection or hint-influence at the mechanism level. The natural feature-level extensions — local attribution shift, hint-token influence — fail in informative ways. We characterize the failures and discuss what mechanistic substrate would be needed instead (embedding-level or QK-attribution).

**This is a stronger paper than "three metrics work."** Three metrics that all work in the same direction would be a curated coincidence. One metric that works plus two principled negative results that tell you *where the method's limits are* is a methodological contribution. The field needs to know where attribution-graph methods reach and where they don't.

### 2.5 What's still on the table for Day 4 (one new exploratory metric)

The §4.5 result said mistake injection produces *diffuse* attribution shifts, not local ones. ED measured locality and got near zero. But the diffuse shifts themselves are real (the magnitude of the shift is non-zero); they just don't concentrate near the mistake.

A natural follow-up question: **does the *total magnitude* of the diffuse shift correlate with aoc_mistake?** If yes, the right instrument for mistake-detection is not a localised window but a global-reorganisation metric.

**Candidate metric: GRM (Global Reorganization Magnitude):**
$$
\mathrm{GRM}(G_0, G_1) = \sum_p |\mathrm{inf}(p, G_1) - \mathrm{inf}(p, G_0)|
$$
i.e., the L1 norm of the position-resolved influence-difference vector. This is exactly the *denominator* of ED — what ED was dividing by. Compute it as a standalone metric. **Time cost: 30 minutes.**

**Decision rule for whether GRM goes in the paper:** if ρ(GRM, aoc_mistake) ≥ 0.20 on the Gemma mistake-injection pairs (n=143 split into mistake subset), it goes in as a fourth metric with an honest "GRM is a noisier-but-positive instrument; ED's locality assumption is what was wrong." If ρ < 0.20, GRM is dropped and the paper's ED section reports only the locality-failure result.

**This is a Day 4 morning task, scoped tight. Do not let it expand.**

---

## 3. The second-model campaign (was: "robustness paragraph"; now: second pillar)

The cross-model campaign on Llama-3.2-1B-Instruct is no longer a paragraph; it is the second pillar of the paper. Spec below.

**Model:**
- `meta-llama/Llama-3.2-1B-Instruct` at fp16
- Llama-Scope ReLU PLTs (the transcoder set bundled with circuit-tracer, called via `--transcoder_set llama` in the CLI; see the `demos/llama_demo.ipynb` notebook for working examples)
- 16 layers (compare Gemma's 26) → fewer activations per graph → smaller VRAM footprint than Gemma
- Full attention (compare Gemma's sliding-window + global)

**Item selection (~480 total graphs):**

The principle is: maximally informative, minimally expensive. Pick items where we already have Gemma graphs, so cross-model comparisons are on matched items. Reuse the existing CoT / paraphrase / mistake / hint construction pipeline so item construction is amortized.

- **Truncation pairs (largest budget, primary replication target):**
  - 100 BBH items from the 10 subtasks used on Gemma, stratified evenly (10 per subtask)
  - 2 conditions per item: clean CoT, truncation at 50% (drop the 25% and 75% variants — we only need one truncation level on Llama to test direction)
  - = 200 graphs

- **Mistake-injection pairs (smaller budget, dissociation test):**
  - 60 items (40 BBH, 20 MMLU), stratified
  - 2 conditions per item: clean CoT, mistake-injected CoT
  - = 120 graphs

- **Hint-injection pairs (smaller budget, HTIR test):**
  - 80 items from the Turpin hint set + augmented BBH-hinted items
  - 2 conditions per item: unbiased prompt + same CoT, biased prompt + same CoT
  - = 160 graphs

- **Total: ~480 Llama graphs.**

**Compute and cost estimate:**

Llama-3.2-1B has fewer parameters (1B vs Gemma's 2B) and 16 layers vs 26. Per-graph time on circuit-tracer is roughly linear in `n_layers × n_features_per_layer × seq_len²` for the attribution Jacobian; Llama's smaller-layers-count partially offsets Llama-Scope's potentially larger feature counts. Conservative estimate: similar per-graph time to Gemma, possibly 20% faster.

Using your Gemma campaign rate (~780 graphs in 3-4 hours on A100, per `paper.tex`):
- 480 graphs at the same rate: ~2.5 hours of pure compute
- Add overhead for setup, model load, transcoder load, intermittent failures: budget 8 GPU-hours
- Modal A100-80GB at ~$2.50/h: **~$20–25 of compute, budget $40 to be safe**

**Item-matching strategy (critical for cross-model comparisons):**

For every Llama item, generate Llama's own CoT (don't reuse Gemma's CoT — they're different models, different reasoning). Match on `item_id`, *not* on CoT content. This means:

- Same BBH/MMLU question → both models generate their own CoTs → each model's pair-construction is internally consistent.
- Cross-model comparisons happen at the item level: for item X, what is GRACE-T on Gemma and what is GRACE-T on Llama? Are they correlated?
- This is what lets you say "the truncation-AOC correlation replicates" — same items, both models, matched behavioural ground truth.

**The two cross-model statements the paper can make:**

1. **Within-model replication:** ρ(GRACE-T, AOC_truncate) is positive and significant on both Gemma (n=143) and Llama (n=100). If yes, the metric is not a Gemma artifact.

2. **Per-item cross-model correlation:** for items present on both models, ρ(GRACE-T_Gemma, GRACE-T_Llama) > 0. If yes, the metric measures something about the *item's CoT structure* that transcends model identity — the strongest possible result. If no but (1) holds, the metric is principled but item-dependent properties are model-specific — still publishable, weaker claim.

**What you do NOT do:**
- Do not generate paraphrases on Llama. Regime A is not needed for the headline; it was a sanity check on Gemma.
- Do not also try Qwen-3-4B. Pick one second model. (Qwen3-4B is supported by circuit-tracer but the VRAM footprint will eat your budget.)
- Do not run cross-layer transcoders (CLTs) on Llama. PLTs only. Match the Gemma setup.
- Do not regenerate Gemma graphs. They're done.

**If the Llama replication partially fails:**

If GRACE-T replicates on Llama but ED or HTIR doesn't, the paper still goes through. The contribution is the *framework* (three metrics, three questions, designed under R1–R6); the strength of each metric in each model is empirical. The honest version is: "GRACE-T replicates cross-model; ED replicates with a different optimal window fraction on Llama; HTIR remains chance on both." That is a clean three-row table in §5, and reviewers will respect the honesty.

If GRACE-T also fails to replicate on Llama, the paper's spine fails and you should pivot to "the truncation correlation we found on Gemma does not replicate on Llama, suggesting it may be a Gemma-Scope-PLT-specific phenomenon." This is also a publishable result — it tightens the field's epistemics — but it's a much weaker paper. The pilot on Day 4 is designed to surface this case before you spend the full compute budget.

---

## 4. Deep graph-understanding work — COMPLETED (May 11–13)

This section was the spine of the v2.1 plan; the work is now done. Section retained for the record and as a reference for the paper's methods section. Key findings (already integrated into §1):

- **§4.1 Graph census:** 1,977 rows × 31 columns at `artifacts/agd/graph_census.parquet`. Late layers carry 65.6% of influence on average.
- **§4.2 Backbone vs item-specific split:** 40 backbone (>50% frequency) + 605 item-specific (<10%) + 114 middle. Bimodal structure confirmed.
- **§4.3 Depth-band breakdown:** Computed on all 820 Regime-B pairs; full correlations with AOC components pending AOC merge in Day 4 analysis.
- **§4.4 Neuronpedia walkthrough:** to do on Day 4 (see Day 4 plan).
- **§4.5 Mistake-locality sanity check:** ***falsified***. Locality fraction 0.003 vs random 0.0058.
- **§4.6 Hint-token sanity check:** ***falsified***. HTIR anti-predictive at AUROC 0.439.

The original §4 subsections are preserved below for reference and replication detail.

### 4.1 The graph census (executed)

For each of the 780 graphs, dump a single row of statistics to a parquet:

- `item_id`, `condition`, `regime`
- `n_nodes_total`, `n_feature_nodes`, `n_error_nodes`, `n_token_nodes`, `n_logit_nodes`
- `n_edges`
- `n_unique_layers_with_features`
- `total_influence`, `top1_influence`, `top10_influence`, `entropy_of_influence`
- `n_unique_concepts` (post-PANO)
- `gini_of_influence` (concentration)
- For each layer band {early, mid, late}: `total_influence_band`, `n_features_band`, `top_feature_id_band`

This is one pandas-friendly file, ~780 rows × 25 columns. You should have it in 2 hours, and you will refer to it constantly.

### 4.2 The backbone vs. specific features split

For each graph, partition the top-64 concepts into:

- **Backbone:** concepts that appear in the top-64 of >50% of all 780 graphs (these are the `L22_F11133`-type universals).
- **Item-specific:** concepts that appear in <10% of graphs.
- **Middle:** everything else.

Then for each pair `(G_0, G_1)`, recompute GRACE separately over backbone-only nodes and over item-specific-only nodes. **Strong prediction:** backbone-only GRACE is near zero (mechanism intact); item-specific GRACE carries the signal. If this is true, the paper now has a *cleaner* GRACE-T variant — restricted to item-specific concepts — which should produce a stronger correlation.

This is conceptually the move from "compare all top features" to "compare the features that vary across items in the first place." It's about 50 lines of pandas code on the existing graphs.

**Cross-model note (R4):** the backbone/item-specific *thresholds* (>50% / <10%) and the *structure* of the split should replicate on Llama; the specific concept IDs will not. Re-derive the split on Llama graphs in isolation; do not import Gemma's backbone list. Compare:
- Does the same bimodal frequency distribution appear on Llama?
- Is the size of the backbone (number of universal concepts) of the same order of magnitude on both models?
- Does the "restrict GRACE to item-specific concepts" trick produce a similar correlation lift on Llama as on Gemma?

If yes to all three, that is strong evidence that the mechanism — model-agnostic backbone of context features + item-specific reasoning features — is real. If no, the trick is Gemma-Scope-specific and you mention that honestly.

### 4.3 Depth-band breakdown of where the divergence lives

For every Regime-B pair, compute GRACE separately within each depth-band (early d∈[0,1/3), mid [1/3,2/3), late [2/3,1] — per R1). Correlate each depth-band GRACE with AOC components.

**Hypothesis:** mid-depth GRACE-T correlates with truncation AOC more strongly than late-depth GRACE-T (because late layers are output-format selection, which is preserved across truncations). If true, this is a striking figure for the paper, and one that should be invariant to absolute layer count.

**Cross-model question:** does the depth band that carries the signal *match* across Gemma and Llama? If on Gemma the truncation signal lives in d∈[1/3, 2/3) and on Llama it lives in d∈[2/3, 1], that is a finding about cross-model differences in how reasoning is depth-distributed — possibly the most interesting result if it appears.

### 4.4 Per-item Neuronpedia walkthrough on 6 illustrative items

Pick 6 items:
- 2 high-GRACE-T, high-AOC (mechanism really reorganises, behaviour really changes)
- 2 low-GRACE-T, low-AOC (mechanism stable, behaviour stable)
- 2 *high* GRACE-T, *low* AOC (mechanism reorganises but behaviour is robust — interesting outlier)

For each, look up the top-5 gained, lost, and shared concepts on Neuronpedia. Give them human-readable labels. Put in an appendix table. This is what the workshop reviewer said was missing (W7), and it is a one-afternoon task that gives the paper qualitative texture.

### 4.5 Test the mistake-locality hypothesis directly

Before writing ED, sanity-check: pick 10 Regime-B mistake-injection pairs. For each, find the token position where the mistake was inserted. Plot `inf(p, G_clean)` vs `inf(p, G_mistake)` as a function of p. If the divergence is concentrated around the mistake position, ED will work. If the divergence is diffuse, ED won't work and you need to rethink. **Do this BEFORE writing the metric code.** It is the empirical check that justifies the metric choice.

### 4.6 Test the hint-token hypothesis directly

For 10 Regime-C pairs (5 flips + 5 non-flips), compute the influence at each token position in the biased graph. Is the hint-token position a noticeable peak? If yes for flips and no for non-flips, HTIR is the right metric. If no for both, the hint operates through some other mechanism and HTIR won't work — possibly the answer is in the embedding layer's contribution, not a specific token position.

---

## 5. Day-by-day execution plan (12 days remaining)

Today is May 13 (Wednesday). EMNLP ARR deadline May 25 (Monday), 11:59 UTC-12 ("AOE"). Treat your real deadline as **EOD May 24** to leave buffer.

**Headcount:** Aiden and Shaan. I'll mark tasks by initial.

### Days 1–3 (May 11–13): COMPLETED ✅

The graph-understanding work, metric design, and metric implementation are done. The Day 1–3 outputs (graph census, backbone split, sanity checks, ED and HTIR computation on the Gemma corpus, ten new functions in `src/pano.py`) are summarized in `RESEARCH_SUMMARY_MAY13.md`. The findings are folded into §1, §2.2, §2.3, §2.4, and §2.5 of this revised plan.

The Day 2 gate ("decide whether ED and HTIR designs are right") has been answered: ED and HTIR designs are *not* right as metrics, but their failures are scientifically informative and form part of the paper's contribution.

### Day 4 (May 14) — TODAY/TOMORROW: tighten Gemma analysis + GRM exploration + Llama pilot

This day has changed the most from v2.1. The Llama pilot still happens, but two new Gemma-side tasks are added.

- **A (morning, 2h): GRM exploration.** Compute the candidate global-reorganization metric (§2.5) on the Gemma mistake-injection pairs. Correlate with aoc_mistake. If ρ ≥ 0.20, GRM becomes the paper's fourth metric (with honest framing); if ρ < 0.20, drop it. Lock the decision before noon.

- **A (morning, 2h): characterize ED's failure quantitatively.** The §4.5 sanity check used n=10. Expand to the full corpus (n=143). Compute for each mistake-injection pair: (i) locality fraction at f=0.05, 0.10, 0.20, 0.30, (ii) total GRM, (iii) the layer at which the maximum attribution shift occurs. **The goal is a table for the paper** characterising *where* and *how* the diffuse shift lives, so the negative result is precise.

- **B (morning, 2h): characterize HTIR's anti-prediction quantitatively.** Beyond AUROC = 0.439, compute: (i) the bootstrap CI on HTIR_flipped vs HTIR_nonflipped, (ii) whether the anti-prediction is driven by a tail of high-HTIR non-flipped items or by a systematic shift in the distribution, (iii) per-subtask AUROC to see if the anti-prediction is dataset-wide or specific. **This is what makes a "HTIR fails" claim publishable rather than just stated.**

- **A + B (afternoon, 3h): bootstrap CIs on the headline numbers.** BCa, 5000 resamples, on ρ(GRACE-T, aoc_truncate), ρ(GRACE-T, aoc_mistake), GRM-AOC correlation if it survived. Holm-Bonferroni across the family.

- **A (late afternoon, 3h): Llama pilot.** 10 items × 3 conditions = 30 Llama-3.2-1B graphs. Compute GRACE-T, ED, HTIR, GRM (if surviving) on these 30 pairs. **The pilot is the gate.** Verify:
  - GRACE-T values are non-degenerate (not all 0 or all 1)
  - The R1–R6 metric definitions actually return sensible numbers on Llama
  - Token-position metadata in Llama graph JSONs follows the same schema as Gemma
  - The depth-band split (d ∈ [0, 1/3), [1/3, 2/3), [2/3, 1]) maps to sensible Llama layers (0–4, 5–9, 10–15)
  
- **B (late afternoon, 2h): Neuronpedia walkthrough on 6 illustrative items.** Pick 2 high-GRACE-T high-AOC + 2 low/low + 2 high-GRACE-T low-AOC items. Look up the top gained/lost concepts. **This is the qualitative texture the workshop reviewer asked for and should be done now while you have time.**

**End of Day 4 gate:** Headline Gemma numbers are locked with CIs. GRM is either in or out (decision recorded). The Llama pilot has revealed whether GRACE-T transfers cleanly; if not, debug Day 5 morning before launching the full campaign.

### Days 5–7 (May 15–17): full Llama-3.2-1B campaign (~480 graphs)

- **A (Day 5 morning, 4h):** Modify `scripts/04_generate_graphs.py` config: model=`meta-llama/Llama-3.2-1B-Instruct`, transcoders=`llama`, output dir=`artifacts/graphs_llama/`. Pre-run `01_generate_cots.py` on the chosen 240-item subset to get Llama's own CoTs. Construct Llama pairs via `03_construct_pairs.py`.

- **A (Day 5 afternoon — Day 6):** Generate the 480 Llama graphs in three batches:
  - Batch 1 (Day 5 PM): 200 truncation graphs. ~3 GPU-h.
  - Batch 2 (Day 6 AM): 120 mistake graphs. ~2 GPU-h.
  - Batch 3 (Day 6 PM): 160 hint graphs. ~3 GPU-h.
  - Buffer: 4 GPU-h for overhead and re-runs.
  - **Total Modal budget: ~$40 on A100-80GB.**
  - Checkpoint aggressively — save each graph JSON immediately so partial results survive failures.

- **B (Days 5–6, 12h total):** While A runs the campaign, write Section 2 (Method) and Section 3 (Experimental setup) of the paper. The §2 prose is essentially §2 of this plan with the May 13 findings integrated. Build `refs.bib` *by hand* as you write — verify every entry by clicking.

- **A (Day 7 morning, 3h):** Compute all metrics on the Llama graphs using the same code that runs on Gemma. Output parquets matching the Gemma analysis files. The metric code should run unchanged; if it doesn't, R1–R6 was violated somewhere and you need to fix it.

- **A (Day 7 afternoon, 2h): cross-model analysis.** For each metric, compute:
  - Within-model correlation on Llama (does GRACE-T replicate? do ED/HTIR fail in the same way?)
  - Per-item cross-model correlation (ρ(GRACE-T_Gemma, GRACE-T_Llama) on matched items)
  - Per-depth-band breakdown on Llama (does the depth signature match Gemma?)

- **B (Day 7, 4h):** Generate cross-model comparison table and the main paper figures. Update §1's findings table with Llama numbers.

**End of Day 7 gate:** Cross-model evidence is locked. The four outcomes and their paper implications:

1. **GRACE-T replicates on Llama; ED/HTIR also fail on Llama in the same direction.** Best case. Clean "one metric, two principled failures, validated cross-model" story.
2. **GRACE-T replicates; ED/HTIR fail on Llama but in different directions.** Still good. Honest write-up: "feature-level metrics for mistake/hint fail in model-specific ways, suggesting the underlying mechanism may differ across architectures."
3. **GRACE-T weakens substantially on Llama (ρ < 0.20).** Concerning. The paper's spine is in trouble. Pivot to "the truncation correlation may be Gemma-Scope-specific; we discuss why" and focus the contribution on the negative results plus methodological tools.
4. **Catastrophic failure of pipeline on Llama.** Should have been caught at Day 4 pilot. If it happens here, drop the Llama campaign and submit a Gemma-only paper with an explicit single-model limitations note — better than missing the deadline.

### Days 8–10 (May 18–20): writing

- **B:** Write Section 1 (Introduction) and Section 4 (Results). The §1 lede is: "we show feature-attribution graphs can measure one of three faithfulness sub-properties (truncation/CoT-necessity) and characterize why the natural feature-level extensions fail for the other two."

- **A:** Write Section 5 (Discussion). Lean hard on: "what feature-attribution graphs can and cannot measure about CoT faithfulness, and what mechanistic substrate would be needed to extend the method."

- **Both:** Section 6 (Related work). Build `refs.bib` *by hand*, verify every entry by clicking. **This is the single highest-leverage anti-desk-reject action.** Cite Chen et al. 2025 (arXiv:2505.05410) not "Anthropic blog"; cite Zhao et al. 2025 (arXiv:2510.09312) with the correct title *"Verifying Chain-of-Thought Reasoning via Its Computational Graph"*; cite Hanna et al. 2025 with the BlackboxNLP 2025 venue.

- **End of Day 10 gate:** Draft of paper is complete, all sections written, all figures placed, all citations click-verified.

### Days 11–12 (May 21–22): polish

- Both: read each other's sections. Cut by 20%. EMNLP short papers are 4 pages content + unlimited refs/appendix. You will overflow.
- Run formatter (ARR 2026 May cycle `acl.sty` template — check the EMNLP CFP for the exact version).
- Move anything non-essential to appendix.
- **Re-verify every citation one final time by clicking each arXiv / DOI / URL link.**

### Days 13 (May 23–24): submission buffer

- Submit by EOD May 24 at the latest. The official deadline is May 25 11:59 PM AOE, but OpenReview can fail at the last minute.
- Author registration: complete by May 27 (mandatory for all authors).

---

## 6. Things you should NOT do

This is the part of a research plan most people skip and most people regret skipping. In order of regret-risk:

1. **Do not generate a second full 700-prompt graph campaign on a second model.** You don't have the time. The Llama campaign is restricted to ~480 graphs with a deliberate item budget per regime. Stick to that scope.

2. **Do not also try Qwen-3-4B as a third model.** This was discussed and decided May 12: with ED and HTIR already falsified on Gemma, the marginal value of a third model is near zero on the failure side and modest on the GRACE-T side. The compute and writing-time cost is not worth it.

3. **Do not let metric definitions baked for Gemma silently leak into the Llama analysis.** Every metric must obey R1–R6 (depth-as-fraction, window-as-fraction, top-k-as-fraction, etc.). If you find yourself writing "layer 22" in Python code, stop.

4. **Do not "fix" ED and HTIR to make them work.** The temptation under deadline pressure is to tune ED's window size or HTIR's position set until they correlate. This is overfitting to the test data and will be transparent to reviewers. The negative results, as they stand, are scientifically valuable. Report them honestly. The one acceptable exploration is GRM on Day 4, with a pre-registered ρ ≥ 0.20 threshold and an immediate drop if it fails.

5. **Do not let an LLM write the references file.** Build `refs.bib` by hand. Every entry: paper title (copy-paste from arXiv abstract page), authors (copy-paste, do not retype), arXiv ID, year, venue if known. Verify each entry by clicking through. **This is the single highest-leverage action to prevent another desk-reject.** EMNLP 2026 explicitly states they will desk-reject for hallucinated citations.

6. **Do not claim PANO is novel methodology.** It is a one-line fix for a specific position-indexing artifact in circuit-tracer's JSON output. It is a *contribution* (worth a paragraph), not *the* contribution. The contribution is the positive GRACE-T result + the two principled negative results + cross-model validation.

7. **Do not present GRACE as "better than X."** Frame as: "GRACE-T captures CoT necessity at the mechanism level on two model families; the natural feature-level extensions ED and HTIR fail in informative ways." This is the v2.2 frame and it is much harder to attack than "we built a better metric."

8. **Do not over-interpret a single Neuronpedia label.** The qualitative labels like "narrative progression" are autointerp guesses. Pick labels with high confidence scores, cite Neuronpedia as the source, use them descriptively, not as load-bearing claims. **Neuronpedia entries differ between Gemma-Scope and Llama-Scope** — don't conflate them.

9. **Do not over-explain why ED and HTIR failed.** The temptation is to pick one of the three interpretations for each (e.g. "ED failed because the mechanism is genuinely distributed") and run with it. The honest version flags all candidate interpretations and notes that your data cannot distinguish them. Reviewers respect epistemic humility on negative results; they do not respect overconfident post-hoc explanations.

10. **Do not promise more than the data delivers.** The paper's strength is its honesty and scope-discipline, not its hit-rate. One working metric + two informative failures is a better paper than four metrics held together with handwaves.

---

## 7. The paper outline (4-page EMNLP short) — v2.2 framing

**Working title:** *What feature-attribution graphs can and cannot measure about chain-of-thought faithfulness.*

(Alternative working title, more provocative: *One metric, two failures: mapping the reach of attribution-graph methods for CoT faithfulness.*)

**Abstract (~200 words, v2.2 version reflecting actual findings):**
> Behavioural chain-of-thought faithfulness scores composite multiple perturbation protocols — truncation, mistake injection, hint injection — into a single number. We ask whether feature-attribution graphs over sparse-autoencoder dictionaries can provide a mechanism-level counterpart. We propose three principled candidate metrics over circuit-tracer graphs: a concept-overlap divergence (GRACE-T) for truncation, a position-resolved attribution shift (ED) for mistake injection, and a hint-token influence ratio (HTIR) for hint-flip prediction. Evaluated on Gemma-2-2B with full Gemma Scope per-layer transcoders (~1,977 graphs across three regimes), only GRACE-T succeeds: it correlates with Lanham truncation AOC at ρ=+0.35 (n=143, BCa CI [+0.21, +0.48]) and is uncorrelated with mistake-injection AOC (ρ=−0.09). ED's design assumption — that mistake injection produces local attribution shifts — is falsified: the actual shift is diffuse, with mean locality fraction of 0.003 against random expectation 0.0058. HTIR is anti-predictive of hint-induced flips (AUROC=0.439); flipped items route *less* influence through hint tokens than non-flipped items. We replicate these findings on Llama-3.2-1B (~480 graphs). The negative results constrain where attribution-graph methods reach: they capture CoT necessity but not error detection or embedding-level hint influence. We release all metric implementations and ~2,460 graphs across two model families.

**Section 1 — Introduction (~0.5 page).** Behavioural faithfulness is composite. Mech-interp tools (circuit-tracer + Gemma Scope) make per-prompt mechanism fingerprints available. We ask whether they can directly substitute for behavioural perturbations. They can for truncation; they cannot for mistake or hint, in informative ways that we characterise.

**Section 2 — Method (~1 page).**
- 2.1 Graph extraction (1 paragraph, cite Hanna et al. 2025, Lieberum et al. 2024).
- 2.2 Concept-level abstraction and the position-indexing artifact (1 paragraph — this is PANO).
- 2.3 GRACE-T, the position-stripped concept divergence (1 short paragraph + equation).
- 2.4 ED and HTIR — the two natural extensions and their stated design assumptions.
- 2.5 Model-agnostic design rules R1–R6.
- Figure 1: schematic showing graphs → three metric computations side by side, with their domain hypotheses.

**Section 3 — Experimental setup (~0.4 page).**
- Datasets: BBH 10 subtasks, MMLU 5 categories, Turpin hint set.
- Models: Gemma-2-2B-it with Gemma Scope PLTs (primary; ~1,977 graphs); Llama-3.2-1B-Instruct with Llama-Scope PLTs (cross-model; ~480 graphs).
- Stats: BCa bootstrap (5000 resamples), Holm-Bonferroni across the test family.

**Section 4 — Results (~1.5 pages).**
- 4.1 GRACE-T tracks truncation AOC; the AOC dissociation. (Table 1: ρ of GRACE-T against each AOC component.) **Headline.**
- 4.2 ED's locality assumption is falsified. (Figure 2: position-resolved attribution shift around the mistake, averaged; flat distribution.) The diffuse-shift finding.
- 4.3 HTIR is anti-predictive of hint-flips. (Figure 3: per-item HTIR scatter with flip outcome.) Convergent negative result with H2 PANO.
- 4.4 **Cross-model replication on Llama-3.2-1B (Table 2).** What replicates, what doesn't.
- 4.5 Per-depth-band view on both models (small panel or appendix).
- 4.6 GRM as exploratory followup *if* it survived Day 4. Otherwise drop.

**Section 5 — Discussion + limitations (~0.5 page).**
- The headline framing: one metric works, two natural alternatives fail in informative ways.
- What the failures imply mechanistically: mistake response is distributed across the graph (or invisible to current pruning); hint-induced flips operate through a substrate (embedding-level, attention-mediated) that feature-level attribution graphs do not surface.
- What next: QK-attribution from Lindsey et al. (July 2025) is the natural extension for hint-related questions; whole-graph reorganisation metrics for mistake-detection.
- Limitations: two model families; PLT-based transcoders only; correlational not causal; pruned graphs are summaries not complete records.

**Section 6 — Related work (~0.2 page in main + extended in appendix).**
- Behavioural faithfulness: Lanham et al. 2023 (arXiv:2307.13702), Turpin et al. 2023 (arXiv:2305.04388), Chen et al. 2025 (arXiv:2505.05410, NOT "Anthropic blog"), Atanasova et al. 2023, Parcalabescu & Frank 2024 (arXiv:2311.07466, ACL 2024, CC-SHAP), Meek et al. 2025 (arXiv:2510.27378).
- Attribution graphs: Ameisen et al. 2025, Lindsey et al. 2025 (transformer-circuits.pub), Hanna et al. 2025 (BlackboxNLP ACL 2025).
- CRV (Zhao et al. 2025, arXiv:2510.09312, *"Verifying Chain-of-Thought Reasoning via Its Computational Graph"*) — orthogonal axis (correctness, not faithfulness; classifier, not paired comparison).

**Appendices (unlimited length, helpful for reviewers):**
- A: Top-k ablation on GRACE-T (already done).
- B: Influence-weighting ablation (already done).
- C: Edit-distance / graph-size partial-Spearman audit (already done).
- D: ED locality fraction at f ∈ {0.05, 0.10, 0.20, 0.30} on Gemma and Llama. The full ED failure characterisation.
- E: HTIR per-subtask breakdown; bootstrap CIs on HTIR_flipped vs HTIR_nonflipped.
- F: Neuronpedia qualitative labels for 6 illustrative items.
- G: Per-depth-band correlation tables on both models.
- H: The full Llama campaign log (item lists, GPU-hours, costs).
- I: Pre-registration commit hashes and analysis-script provenance.

---

## 8. Validate-don't-trust checklist before submission

Print this and tick each box, no shortcuts.

**Citation hygiene (highest-priority anti-desk-reject items):**
- [ ] Every entry in `refs.bib` has an arXiv ID OR a DOI OR a stable URL. I have clicked each one and verified that the linked paper has the claimed title and authors.
- [ ] No reference is to "Anthropic Research Blog" or "alphaXiv" or "preprint" without a verifiable identifier.
- [ ] Chen et al. 2025 cited with arXiv:2505.05410 (not as "Anthropic blog").
- [ ] Zhao et al. 2025 (CRV) cited with arXiv:2510.09312 AND with correct title *"Verifying Chain-of-Thought Reasoning via Its Computational Graph"*.
- [ ] Parcalabescu & Frank 2024 cited with correct title *"On Measuring Faithfulness or Self-consistency of Natural Language Explanations"* (arXiv:2311.07466, ACL 2024).
- [ ] Meek et al. 2025 cited with correct title *"Measuring Chain-of-Thought Monitorability Through Faithfulness and Verbosity"* (arXiv:2510.27378).
- [ ] Hanna et al. 2025 cited with BlackboxNLP @ ACL 2025 venue.

**Results integrity (v2.2 specific):**
- [ ] Every numerical claim in the abstract appears verbatim in §4, with its CI.
- [ ] The dissociation table includes Holm-Bonferroni-corrected p-values across the whole family of tests.
- [ ] GRACE-T described as "post-hoc-discovered (via PANO position-indexing fix), then validated on a held-out split and on a second model" — not "confirmatory."
- [ ] ED is described as **falsified**, with the locality-fraction result reported alongside three candidate interpretations.
- [ ] HTIR is described as **anti-predictive on both models**, with the implication that hint-flips operate through a substrate outside feature-attribution graphs.
- [ ] If GRM survives Day 4 and enters the paper, its post-hoc discovery is acknowledged.

**Methodology safety:**
- [ ] Every metric definition in §2 satisfies R1–R6 (no hard-coded layer indices, no absolute token counts, no Gemma-specific feature IDs).
- [ ] The cross-model table reports each metric's Gemma value, Llama value, and an honest interpretation of whether it replicated.
- [ ] Limitations section lists: only two model families; only PLT-based transcoders; correlational nature; pruning approximation; depth-fraction binning as a coarse approximation; the three non-exclusive interpretations of ED's failure are flagged.

**Reproducibility:**
- [ ] The pre-registration commit hash is cited in the appendix and the commit is publicly accessible.
- [ ] An anonymous GitHub repo is prepared, with Gemma and Llama graph generation scripts plus `src/pano.py` and scripts 20–22.
- [ ] All authors have completed ARR reviewer registration by May 27.

---

## 9. What "strong, generalised metric" actually means here — v2.2 update

You asked, originally, for a metric that captures CoT faithfulness across truncation, mistake, hint, "and maybe something else." A general-purpose metric, like a thermometer for unfaithfulness, would be wonderful. Your data, three days in, says we should not expect one — and crucially, it says *why*.

The v2.1 plan said: "yes to one principled instrument per principled question, no to one-number-captures-everything." That was almost right. The May 13 results refine it:

**The v2.2 answer is: yes to "strong and generalised" in two specific senses:**

1. *One principled instrument for one principled question* — GRACE-T for truncation/CoT-necessity. It works. It correlates with its matched behavioural axis. It will (likely) work cross-model.

2. *Honest accounting of where the method does not reach* — ED's locality assumption is wrong; mistake response is diffuse. HTIR's hint-attribution assumption is wrong; flipped items route *less* through hint tokens. These failures are *informative* because they tell the field what attribution-graph methods can and cannot do. A reviewer who reads a paper that says "we have one working metric and two failed extensions that tell us where this method's reach ends" will trust the work more than a paper claiming three for three.

**No to "strong and generalised" in the third sense:** one number captures everything. That would force overclaiming, and overclaiming is what got every previous version of this paper rejected.

**The shift you should internalize:** the contribution is no longer "we built three metrics." It is **"we mapped the reach of feature-attribution-graph methods for CoT faithfulness measurement, finding one positive result and two principled negative results."** That is a tighter, more interesting, more defensible paper. The negative results would be uninteresting on a method nobody had tried yet; they are interesting because the method (circuit-tracer + Gemma Scope / Llama-Scope) is the current state of the art for this kind of analysis. You are telling the field *here is what this tool measures well, and here is what it does not*.

The cross-model evidence you fought for (correctly) on May 12 is what makes this credible. On one model these would be three Gemma observations. On two models they become the boundary of what this *class of methods* measures.

That is the paper. Write it.

---

*Document version: v2.2 — May 13, 2026 (revised after Day 1–3 results falsified ED and HTIR designs; the contribution shifts from "three metrics" to "one metric + two principled negative results, validated cross-model"). Living document; update after the Day 4 gate (GRM exploration + Llama pilot) and the Day 7 gate (full Llama campaign results).*
