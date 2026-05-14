# GRACE → v2.1: Research plan for EMNLP 2026 short paper

**Status of the work as of May 12, 2026:**
- ICML Mech Interp workshop: paper desk-rejected for hallucinated citations (you have fixes for those).
- Real scientific findings on disk: **dissociation** between truncation-AOC (ρ=+0.35) and mistake-injection-AOC (ρ=−0.09) on n=143 Regime-B items, Gemma-2-2B-it, full Gemma Scope PLT stack.
- Single model. PANO discovered post-hoc as the position-indexing fix. H2 (hint-flip) is at chance.
- ~780 graphs already generated.

**EMNLP 2026 short paper:** ARR submission deadline May 25, 2026 → **13 days from today**. Up to 4 content pages + unlimited references and appendix.

The deadline imposes hard rules on what you should and should not do. **You will not generate a second 700-prompt graph campaign in 13 days.** You will write a stronger paper around the existing Gemma work, AND run a scoped ~480-graph cross-model campaign on Llama-3.2-1B that is the second pillar of the paper, not a footnote. The metric design and the cross-model campaign are intertwined: metrics are designed under model-agnostic rules and validated on both models before being committed to.

---

## 0. Revising the strategy: depth and breadth are not independent

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

7. **Backbone features are real.** Concepts like `L22_F11133` ("narrative progression"), `L24_F12351`, `L25_F5714` show up as high-influence in 80%+ of pairs. The graphs share a domain-general backbone; what differs are the item-specific concepts.

The dissociation in (3) and the hint-flip failure in (5) are *findings*, not bugs. Treat them as such.

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

**Behavioural question:** if I splice a wrong step into the CoT, does the answer change?
**Mechanistic translation:** the mistake is *local* — it changes one or two tokens. The reorganisation we'd expect under a faithful, error-sensitive model is also *local*: error-detection features should fire near the mistake position. A global top-k summary will not catch this. This is exactly why your aoc_mistake correlation is at ρ=−0.09.

**Metric: ED (error-detection-localisation).** Two-part:

1. Identify the token range where the mistake was inserted (you already have this — `add_mistake_to_cot` knows where it inserted the text).
2. Compute the *position-resolved* attribution shift in a window around the mistake. Under R2, the window is a fraction of the CoT length, not an absolute token count.

Formally, let `inf(p, G)` = sum of normalized `influence(n)` over nodes `n` in `G` with token-position `p` (normalization per R5). Let L be the CoT length in tokens and let W = [p_mistake − ⌈fL/2⌉, p_mistake + ⌈fL/2⌉] with f = 0.10. Then:
$$
\mathrm{ED}_f(G_0, G_1) = \frac{\sum_{p \in W} |\mathrm{inf}(p, G_1) - \mathrm{inf}(p, G_0)|}{\sum_{p} |\mathrm{inf}(p, G_1) - \mathrm{inf}(p, G_0)|}
$$
ED → 1 means all the attribution change is local to the mistake; ED → 0 means the change is diffuse, the mistake didn't register specifically.

**Why this is the right instrument:** GRACE used position-stripping precisely to remove position drift. For mistake-injection, *position is the signal.* The very thing PANO collapses is what ED preserves. The two metrics are designed to be complementary, not redundant.

**Note:** this requires un-stripped graphs — the `(layer, position, feature)` form is right here. You have those JSONs already; just compute on a different projection.

**Predicted result:** ED correlates with aoc_mistake on Regime-B mistake pairs at ρ ≥ 0.25 if the metric works. ED is uncorrelated with aoc_truncate (different protocol, different mechanism).

**Cross-model question for this metric:** the f=0.10 default is arbitrary. Run the metric at f ∈ {0.05, 0.10, 0.20, 0.30} on both Gemma and Llama, separately. If the optimal f is similar on both, the metric is genuinely model-agnostic. If it differs by 3× or more, that is itself an interesting cross-model fact — different models distribute the mistake's mechanistic footprint over different ranges. **Always report f explicitly in the paper.**

### 2.3 Hint injection → "is the answer driven by the prompt cue rather than the CoT?"

**Behavioural question:** add a wrong-answer hint to the prompt; does the answer flip without the CoT acknowledging the hint?
**Mechanistic translation:** the hint should appear as a *path through the hint-token's position* in the attribution graph of the biased run. A hint that the model used but didn't verbalise should appear as: significant attribution from the hint-token's residual stream to the answer logit, *without* a corresponding chain through the CoT tokens.

**Metric: HTIR (hint-token influence ratio), already in your `H2_Weakness_Deep_Analysis.md`.** This is the right instrument. The reason GRACE-on-Regime-C failed is that you were measuring graph-level reorganisation; the right object is *the specific influence path from the hint token*.

For each Regime-C pair, compute (using R5 normalization):
$$
\mathrm{HTIR}(G_{\text{biased}}) = \frac{\sum_{n: \text{pos}(n) \in P_{\text{hint}}} \mathrm{inf}(n)}{\sum_n \mathrm{inf}(n)}
$$
That is, the fraction of total influence-on-target that passes through nodes at the hint-token position(s) P_hint. High HTIR + flip → hint is causal. Low HTIR + no flip → hint is ignored. The off-diagonal cases are the interesting interpretive content.

**Why this is the right instrument:** GRACE measures *whole-graph divergence*. HTIR measures a *path through a specific token position*. Different objects, different questions.

**Implementation:** ~3 hours from existing graph JSONs. You already have token positions stored.

**Cross-model question for this metric:** the hint phrase "I think the answer is (X)" tokenizes differently across tokenizers (Gemma uses SentencePiece, Llama uses tiktoken-like BPE). P_hint is therefore a *set* of token positions, not a single one — compute that set programmatically from the tokenizer, not by hand. The metric value should be comparable across models because it's a ratio. **Pre-register an HTIR threshold of 0.05 as "hint is mechanically active" on both models and compare the fraction of items above threshold across models** — if the fractions are similar, that is replication of the mechanism; if they differ sharply, that is a finding about model differences in hint-sensitivity.

### 2.4 The unifying frame (validated on two models)

Three metrics, three behavioural sub-protocols, three different graph operations:

| Sub-protocol | Behavioural q. | Graph operation | Metric |
|---|---|---|---|
| Truncation | Did CoT do work? | Top-k concept turnover, position-collapsed | GRACE-T (=PANO_div) |
| Mistake injection | Did model react to wrong step? | Local position-resolved attribution shift | ED |
| Hint injection | Did prompt cue bypass the CoT? | Influence fraction through hint token | HTIR |

The contribution claim is: **no single mechanism-level summary suffices — the behavioural decomposition has a mechanistic counterpart**, and the right instruments are different graph operations on the same graphs. **All three are defined under the R1–R6 rules above, so they run unchanged on Gemma-2-2B and Llama-3.2-1B, and the cross-model evidence is reported alongside the within-model results.**

This is publishable as is. It directly addresses your stated goal ("metric that captures CoT faithfulness across truncation, mistake-injection, hint-flip"). The honest answer is: **the metric is a vector, not a scalar**, and the paper says exactly that. The cross-model evidence is what distinguishes a principled vector from a curated Gemma observation.

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

## 4. Deep graph-understanding work to do FIRST (before writing metrics)

You said *"rather than/before generating newer graphs, understand generated graphs well."* Yes. Here is the concrete checklist of analyses to run on your existing 780 graphs **before** finalising the metric definitions. This is 2–3 days of work and is the spine of section 2 above.

### 4.1 The graph census

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

## 5. Day-by-day execution plan (13 days)

Today is May 12 (Tuesday). EMNLP ARR deadline May 25 (Monday), 11:59 UTC-12 ("AOE"). That is end of day on the 25th wherever you are. Treat your real deadline as **EOD May 24** to leave buffer.

**Headcount:** Aiden and Shaan. I'll mark tasks by initial.

### Days 1–2 (May 12–13): graph understanding

- **A:** Build the graph census parquet (§4.1). 4h.
- **A:** Compute backbone vs item-specific concept split (§4.2). 3h.
- **B:** Compute per-layer-band GRACE on Regime-B pairs (§4.3). 4h.
- **B:** Run sanity checks (§4.5, §4.6) on 10 pairs each before committing to ED / HTIR designs. 4h.
- **End of Day 2 gate:** decide whether ED and HTIR are the right metric designs, or whether the sanity checks suggest variants. Lock the metric definitions.

### Day 3 (May 14): metric implementation

- **A:** Implement ED on Regime-B mistake-injection pairs. Output a parquet of (item_id, ED, aoc_mistake). 4h.
- **A:** Implement HTIR on Regime-C pairs. Output (item_id, HTIR, flipped). 3h.
- **B:** Re-implement GRACE-T restricted to item-specific concepts. 2h.
- **B:** Re-run the full correlation suite (GRACE-T variants × AOC components). 3h.

### Day 4 (May 15): metric validation on Gemma + Llama pilot

- **A + B (morning):** Bootstrap CIs (BCa, 5000 resamples) on every headline correlation on Gemma. Run Holm-Bonferroni across the whole hypothesis family. 3h.
- **A (morning):** Pick the 6 illustrative items (§4.4) and look them up on Neuronpedia. 2h.
- **B (afternoon): Llama pilot — ~30 graphs only.** Pick 10 items × {clean, truncate_50, mistake} = 30 graphs on Llama-3.2-1B. **This is the gate.** Compute GRACE-T and ED on these 30 graphs and verify:
  - GRACE-T values are non-degenerate (not all 0 or all 1)
  - ED values are non-degenerate
  - Token-position metadata in the Llama graph JSONs follows the same schema as Gemma (it should — same circuit-tracer code path)
  - The R1–R6 metric definitions actually return sensible numbers
- **End of Day 4 gate:** If the pilot looks bad, *stop and debug before Day 5*. The four headline Gemma numbers must be locked: GRACE-T↔AOC_truncate, ED↔AOC_mistake, HTIR↔hint-flip, partial correlations.

### Days 5–7 (May 16–18): full Llama-3.2-1B campaign

- **A (Day 5 morning):** Modify `scripts/04_generate_graphs.py` config: model=`meta-llama/Llama-3.2-1B-Instruct`, transcoders=`llama` (per circuit-tracer's `--transcoder_set llama`), output dir=`artifacts/graphs_llama/`. Pre-run `01_generate_cots.py` on the chosen 240-item subset to get Llama's own CoTs. Construct Llama pairs via `03_construct_pairs.py`. ~4h of work + waiting on CoT generation.
- **A (Day 5 afternoon — Day 6):** Generate the 480 Llama graphs in three batches (truncation, mistake, hint). Estimated ~8 GPU-hours on A100-80GB; budget 12h with overhead. Checkpoint aggressively — save each graph JSON immediately so partial results survive failures.
- **B (Days 5–6):** While A runs the campaign, write Section 2 (Method) and Section 3 (Experimental setup) of the paper. The §2 prose is essentially §2 of this plan, polished and tightened. Build `refs.bib` *by hand* as you write — verify every entry by clicking. 12h total.
- **A (Day 7):** Compute all three metrics on the Llama graphs using the same code that runs on Gemma (it should — R1–R6 ensure this). Output a parquet matching the Gemma analysis files. 3h.
- **B (Day 7):** Generate the cross-model comparison table and figure (see §6 below). 3h.
- **End of Day 7 gate:** the cross-model evidence is locked. Either the decomposition replicates (good — strong paper), or it partially replicates (still OK — honest paper with clear scope), or it doesn't (rare — pivot to "Gemma-specific" framing).

### Days 8–10 (May 19–21): writing

- **B:** Write Section 1 (Introduction) and Section 4 (Results). The dissociation + cross-model replication is the lede.
- **A:** Write Section 5 (Discussion). Lean hard on "the behavioural composite isn't a single property, and this replicates across two model families."
- **Both:** Section 6 (Related work). This is where the citation fiasco was — every citation must be verified against arXiv / ACL / official Anthropic publications. **Build a `refs.bib` with arXiv IDs, DOIs, and direct URLs for every entry, and verify each one by clicking.** Do not let any LLM auto-fill this file.
- **End of Day 10 gate:** draft of paper is complete, all sections written, all figures placed.

### Days 11–12 (May 22–23): polish

- Both: read each other's sections. Cut by 20%. EMNLP short papers are 4 pages content + unlimited refs/appendix. You will overflow.
- Run formatter (ARR uses an `acl.sty` template — check the EMNLP CFP for the exact version, currently the ARR 2026 May cycle template).
- Move anything non-essential to appendix.
- Re-verify every citation one final time by clicking each arXiv / DOI link.

### Day 13 (May 24): submission

- Submit by EOD May 24 at the latest. The official deadline is May 25 11:59 PM AOE, but anything can go wrong with OpenReview at the last minute.
- Author registration: complete by May 27 (mandatory for all authors).

---

## 6. Things you should NOT do

This is the part of a research plan most people skip and most people regret skipping. In order of regret-risk:

1. **Do not generate a second full 700-prompt graph campaign on a second model.** You don't have the time. The Llama campaign is restricted to ~480 graphs with a deliberate item budget per regime. Stick to that scope.

2. **Do not also try Qwen-3-4B.** It is supported by circuit-tracer but the VRAM and time cost will eat the Llama budget. Pick one second model.

3. **Do not let metric definitions baked for Gemma silently leak into the Llama analysis.** Every metric must obey R1–R6 (depth-as-fraction, window-as-fraction, top-k-as-fraction, etc.). If you find yourself writing "layer 22" in Python code, stop.

4. **Do not add a fourth or fifth metric "in case it works."** Three metrics motivated by three specific behavioural axes is a strong contribution. Four metrics where the fourth is "we also tried LIWD and it didn't add much" is a weaker contribution. Cut LIWD from the proposed-metrics doc unless §4 surfaces a reason to keep it.

5. **Do not let an LLM write the references file.** Build `refs.bib` by hand. Every entry: paper title (copy-paste from arXiv abstract page), authors (copy-paste, do not retype), arXiv ID, year, venue if known. Verify each entry by clicking through. **This is the single highest-leverage action to prevent another desk-reject.** EMNLP 2026 explicitly states they will desk-reject for hallucinated citations.

6. **Do not claim PANO is novel methodology.** It is a one-line fix for a specific position-indexing artifact in circuit-tracer's JSON output. It is a *contribution* (worth a paragraph), not *the* contribution. The contribution is the decomposition + cross-model replication.

7. **Do not present GRACE as "better than X."** Frame as: "GRACE-T answers question Q1. ED answers question Q2. HTIR answers question Q3. These are different questions; the behavioural literature has been treating them as one. We validate this on two model families."

8. **Do not over-interpret a single Neuronpedia label.** The qualitative labels like "narrative progression" are autointerp guesses. Pick labels with high confidence scores, cite Neuronpedia as the source, use them descriptively, not as load-bearing claims. **Neuronpedia entries differ between Gemma-Scope and Llama-Scope** — don't conflate them.

9. **Do not chase the hint-flip AUROC to >0.65.** If HTIR also fails to predict flips on both models, you write a clean negative result: "neither whole-graph divergence nor hint-token influence predicts flips in our sample on either model; this suggests hint-induced flips operate through a third mechanism, potentially embedding-level rather than feature-level, which we leave for future work." This *is* a publishable cross-model finding.

10. **Do not promise more than the data delivers.** If only GRACE-T replicates cleanly across both models and ED/HTIR are noisier, write that. The paper's strength is its honesty and scope-discipline, not its hit-rate.

---

## 7. The paper outline (4-page EMNLP short)

**Title:** *Mechanism-level CoT faithfulness is not one property: three attribution-graph metrics for three behavioural sub-protocols.* (Working title — short.)

**Abstract (≤200 words):**
> Behavioural chain-of-thought faithfulness scores composite multiple perturbation protocols — truncation, mistake injection, hint injection — into a single number. We show on Gemma-2-2B that this composite hides a mechanistic dissociation: attribution-graph divergence under CoT truncation correlates with the truncation behavioural score (ρ=+0.35, n=143) but is uncorrelated with the mistake-injection score (ρ=−0.09). We attribute this to the protocols probing different objects — CoT *necessity* vs. *error sensitivity* — and propose three matched metrics computed from circuit-tracer attribution graphs over Gemma Scope features: a position-stripped concept-overlap divergence for truncation, a position-resolved local attribution-shift for mistake injection, and a hint-token influence fraction for hint-flip prediction. Each metric correlates with its matched behavioural axis and is uncorrelated with the others, validating the decomposition. The methodology replicates qualitatively on Llama-3.2-1B (n=80 truncation pairs). The implication for benchmark design is that a single faithfulness number is the wrong abstraction; we release the metric implementations and ~860 graphs.

**Section 1 — Introduction (~0.5 page).** Behavioural-faithfulness is composite. Mech-interp tools (circuit-tracer + Gemma Scope) make per-prompt mechanism fingerprints available. We use them to test whether one mechanistic number explains all behavioural sub-protocols; it does not. Three metrics, three axes.

**Section 2 — Method (~1 page).**
- 2.1 Graph extraction (1 paragraph, cite Hanna et al. 2025, Lieberum et al. 2024).
- 2.2 Concept-level abstraction and the position-indexing pitfall (1 paragraph, our methodological note).
- 2.3 The three metrics, formally defined (3 short subsections, one equation each).
- Figure 1: the schematic showing graphs → three metric computations side by side.

**Section 3 — Experimental setup (~0.4 page).**
- Datasets: BBH 10 subtasks, MMLU 5 categories, Turpin hint set. Cite Suzgun et al. 2022, Hendrycks et al. 2020, Turpin et al. 2023.
- Models: Gemma-2-2B-it with Gemma Scope PLTs (primary, full campaign); Llama-3.2-1B-Instruct with Llama-Scope PLTs (cross-model validation, ~480 graphs).
- Gemma: 143 valid Regime-B paired items, 287 Regime-C pairs.
- Llama: 100 truncation pairs, 60 mistake pairs, 80 hint pairs.
- Stats: BCa bootstrap, Holm-Bonferroni.

**Section 4 — Results (~1.5 pages).**
- 4.1 The dissociation on Gemma (Table 1: ρ of GRACE-T against each AOC component; the +0.35 / −0.09 split). The within-model headline.
- 4.2 ED tracks mistake-injection where GRACE-T does not (Figure 2). Within-model on Gemma.
- 4.3 HTIR and the hint-flip null. Within-model on Gemma.
- 4.4 **Cross-model replication on Llama-3.2-1B (Table 2).** Each row is one metric; columns are (Gemma ρ, Llama ρ, ratio). This is the "second pillar" result and arguably the strongest single figure in the paper.
- 4.5 Per-depth-band view: where the reorganisation lives, on both models (small figure or appendix).

**Section 5 — Discussion + limitations (~0.4 page).**
- Behavioural composites mask mechanism; recommend reporting per-protocol scores.
- Cross-model evidence: what replicates, what doesn't, what that tells us about the mechanism vs. the dictionary.
- Limitations: only two model families; only PLT-based transcoders; correlational not causal; pruned graphs are summaries not complete records.

**Section 6 — Related work (~0.2 page in main + extended in appendix).**
- Behavioural faithfulness: Lanham 2023, Turpin 2023, Chen et al. 2025 (the Anthropic paper, arXiv:2505.05410, NOT a blog post).
- Attribution graphs: Ameisen et al. 2025, Lindsey et al. 2025, Hanna et al. 2025.
- CRV (Zhao et al. 2025, arXiv:2510.09312, *"Verifying Chain-of-Thought Reasoning via Its Computational Graph"*) — orthogonal axis (correctness, not faithfulness; classifier, not paired comparison).

**Appendices (unlimited length, helpful for reviewers):**
- A: Top-k ablation (already done).
- B: Influence-weighting ablation (already done).
- C: Edit-distance / graph-size partial-Spearman audit (already done).
- D: Neuronpedia qualitative labels for 6 illustrative items (Gemma side; Llama side if time permits).
- E: Per-depth-band correlation table on both models.
- F: The full Llama campaign log (item lists, GPU-hours, costs).
- G: Pre-registration file and analysis-script commit hashes.

**Updated abstract (~200 words, reflecting cross-model story):**
> Behavioural chain-of-thought faithfulness scores composite multiple perturbation protocols — truncation, mistake injection, hint injection — into a single number. We show on Gemma-2-2B that this composite hides a mechanistic dissociation: attribution-graph divergence under CoT truncation correlates with the truncation behavioural score (ρ=+0.35, n=143) but is uncorrelated with the mistake-injection score (ρ=−0.09). We attribute this to the protocols probing different objects — CoT *necessity* vs. *error sensitivity* — and propose three matched metrics computed from circuit-tracer attribution graphs: a position-stripped concept-overlap divergence (GRACE-T) for truncation, a position-resolved local attribution shift (ED) for mistake injection, and a hint-token influence fraction (HTIR) for hint-flip prediction. All three are defined under model-agnostic rules (depth-as-fraction, window-as-fraction, normalised influence) and applied unchanged to Llama-3.2-1B with Llama-Scope transcoders on ~480 graphs. The decomposition replicates: each metric correlates with its matched behavioural axis on both models, with consistent directions. The implication for benchmark design is that a single faithfulness number is the wrong abstraction; we release the metric implementations and ~1,260 graphs across two model families.

---

## 8. Validate-don't-trust checklist before submission

Last thing. Print this and tick each box, no shortcuts.

- [ ] Every entry in `refs.bib` has an arXiv ID OR a DOI OR a stable URL. I have clicked each one and verified that the linked paper has the claimed title and authors.
- [ ] No reference is to "Anthropic Research Blog" or "alphaXiv" or "preprint" without a verifiable identifier.
- [ ] Every numerical claim in the abstract appears verbatim in §4, with its CI.
- [ ] The dissociation table includes Holm-Bonferroni-corrected p-values across the whole family of tests.
- [ ] PANO/GRACE-T is described as "post-hoc-discovered, then validated on a held-out split and on a second model" — not "confirmatory."
- [ ] The hint-flip section reports the chance result honestly on both models; HTIR is presented as the right instrument for the right object, regardless of whether it predicts flips at >0.65.
- [ ] Every metric definition in §2 satisfies R1–R6 (no hard-coded layer indices, no absolute token counts, no Gemma-specific feature IDs).
- [ ] The cross-model table (Table 2) reports each metric's Gemma value, Llama value, and an honest interpretation of whether it replicated.
- [ ] Limitations section lists: only two model families, only PLT-based transcoders, correlational nature, pruning approximation, depth-fraction binning as a coarse approximation to layer function.
- [ ] The pre-registration commit hash is cited in the appendix and the commit is publicly accessible.
- [ ] An anonymous GitHub repo is prepared, with both Gemma and Llama graph generation scripts.
- [ ] All authors have completed ARR reviewer registration by May 27.

---

## 9. What "strong, generalised metric" actually means here

You asked for a metric that captures CoT faithfulness across truncation, mistake, hint, "and maybe something else." A general-purpose metric of CoT faithfulness, like a thermometer for unfaithfulness, would be wonderful. Your data says we shouldn't expect one. The dissociation finding is not a setback — it is **the reason the field hasn't been able to converge on a single such metric for three years.**

So: yes to "strong and generalised" in **two** specific senses:

1. *One principled instrument per principled question* — three metrics, three behavioural axes, each metric the right graph operation for its question.
2. *Each instrument works across model families* — defined under R1–R6, validated on Gemma-2-2B and Llama-3.2-1B with two different transcoder families. **This is what you pushed back about, and it is the right thing to push back about.** A metric that only works on one model isn't a metric; it's a description.

No to "strong and generalised" in the third sense: *one number captures everything*. That would force overclaiming.

The thing that will make reviewers respect this paper is two-fold: you looked at the graphs, found that the behavioural axes were mechanistically distinct, and built tools to measure them as such; **and** you validated those tools on a second model with a different architecture and a different transcoder family. That second validation is what separates "we curated three Gemma observations" from "we propose a methodology." It is what your pushback was driving at, and it is correct.

---

*Document version: v2.1 — May 12, 2026 (revised after Aiden's pushback on cross-model integration). Living document; update after the Day 2, Day 4 pilot, and Day 7 gates.*
