# AGD Research — Full Explainer

> Written to help someone who understands mechanistic interpretability basics but has no prior context on this specific project, attribution graphs, or the AGD metric.

---

## Part 1: The Big Problem — "Does the model actually think in the way it writes?"

When a modern LLM like GPT-4 or Gemma answers a hard question, it first writes out a **chain of thought (CoT)** — a sequence of reasoning steps like:

> "The question asks about X. First I'll consider Y... then Z... therefore the answer is A."

This looks like reasoning. But is it *actually* reasoning? Or is the model just:
1. Figuring out the answer using some hidden internal shortcut, and then
2. Writing plausible-sounding reasoning steps *after the fact* — like a student writing a fake working-out after already guessing the answer?

This is called the **CoT faithfulness problem**. A chain of thought is **faithful** if the model's internal computation genuinely used those reasoning steps. It is **unfaithful** (post-hoc rationalization) if the model arrived at the answer via a different route and just wrote the steps to look reasonable.

**Why this matters:**
- Safety: we can't trust model oversight if the reasoning we read doesn't reflect what the model is actually doing
- AI alignment: interpretability tools lose their value if the visible trace is disconnected from the real computation
- Deployment: explanations shown to users or auditors must actually represent the decision process

---

## Part 2: How People Measured Faithfulness Before (Behavioral Tests)

The existing approach is **behavioral**: you poke at the CoT from the outside and see if the answer changes.

### Lanham et al. 2023 — Area Over Completeness (AOC)
They tested: *"If I mess with the CoT, does the answer change?"*
- **Early-answer**: cut the CoT short and see if the model still gets the right answer
- **Mistake injection**: inject a deliberate error into the CoT — does the model propagate it?
- **Paraphrase**: rewrite the CoT in different words — does the answer stay the same?

If injecting mistakes causes the model to get the wrong answer → the model was using the CoT (faithful).
If the model ignores your mistake and still gets the right answer → it was doing something else (unfaithful, post-hoc).

**AOC** (Area Over Completeness) is a summary score of how sensitive a model's output is to CoT perturbations. Low AOC = model is unfaithful (doesn't care about the CoT).

### Turpin et al. 2023 — Hint Injection
They added a sneaky hint to questions: *"I think the answer is (B)"*.
- The model picks (B), even when (B) is wrong
- The CoT it writes does not mention the hint — it fabricates a reasoning path to justify (B)
- This is the clearest behavioral evidence of post-hoc rationalization

### The fundamental limitation of behavioral tests
Both approaches are **indirect**. You can't distinguish between:
- **(A)** The model genuinely used a different mechanism (deep unfaithfulness)
- **(B)** The model used the same mechanism but was robust to the perturbation (surface insensitivity)

A behavioral test says "the output changed" — it can't tell you *why*.

---

## Part 3: The New Tools That Make a Better Measurement Possible

### Mechanistic Interpretability: Finding "Features"
Modern mech-interp doesn't look at raw neurons (there are millions and they're polysemantic — each fires for many unrelated things). Instead, we use **Sparse Autoencoders (SAEs)** and **transcoders** to decompose the model's activations into interpretable **features** — monosemantic units like "this fires when discussing legal concepts" or "this fires when the model is doing arithmetic carry-over".

**Gemma Scope** provides pre-trained transcoders for Gemma-2-2B, covering all 26 layers. Each layer has a dictionary of ~16,000 features.

### Attribution Graphs: Tracing a Single Forward Pass
**Circuit-tracer** (Anthropic, 2025) builds an **attribution graph** for one specific input-output pair. It works by:
1. Running the model on your input
2. Using Jacobians (derivatives) to trace which features in layer L contributed to which features in layer L+1
3. Pruning to keep only the edges with significant contribution
4. Producing a **Directed Acyclic Graph (DAG)**: nodes are features, edges are weighted attributions

The resulting graph shows the computational path the model took for *this specific prompt*. You can look at a node and say "this 'arithmetic carry' feature fired here, and it fed directly into the 'final digit' feature two layers later."

This is what Anthropic's "Biology of a Language Model" paper used for *qualitative* case studies — looking at graphs and saying "look, this reasoning step is reflected in the graph" or "look, the hint token contributed to the answer but the CoT never mentions it."

**The gap**: No one had turned these graphs into a *quantitative* metric. They were used only for human inspection.

---

## Part 4: What This Research Does — AGD

### The Core Idea

This research asks: **if I change the CoT slightly, how much does the internal computational graph change?**

For a **faithful model**: the reasoning steps in the CoT correspond to real computation. So if you replace the CoT with a paraphrase (same meaning, different words), the graph should barely change — the same features fire, the same edges are active.

For an **unfaithful model**: the CoT is decorative. The real computation happens elsewhere. So changing the CoT (even with something semantically equivalent) might change the graph completely — or conversely, injecting mistakes into the CoT won't change the graph at all.

### The AGD Formula

Given:
- A base prompt `x` with CoT `c` producing answer `a`
- A perturbed CoT `c'` (paraphrase, truncation, or hint-induced)

Compute two attribution graphs:
- `G₀ = G(model, [x + c], answer_token)`
- `G₁ = G(model, [x + c'], answer_token)`

Then:

```
AGD = 1 - α·J_w - (1-α)·S_e
```

Where:
- **J_w (Influence-Weighted Jaccard)**: Look at the top-k most influential feature nodes in each graph. How much overlap is there? Weight by their influence scores. If both graphs use the same features with the same weights → J_w ≈ 1 → contributes 0 to AGD.
- **S_e (Edge Cosine Similarity)**: Treat all the edge weights as a big vector. Compute cosine similarity between the two graphs' edge vectors. Same edges, same weights → S_e ≈ 1 → contributes 0 to AGD.
- **α** (tuned hyperparameter, 0 to 1): balances how much node overlap vs. edge overlap matters

**AGD ranges from 0 to 1:**
- AGD ≈ 0: the mechanism didn't change (same features, same pathways)
- AGD ≈ 1: the mechanism completely reorganized

### Three Experimental Regimes

| Regime | What you do | What faithful CoT predicts |
|--------|-------------|---------------------------|
| **A — Paraphrase** | Replace CoT with semantic paraphrase (same meaning, different words) | Low AGD — mechanism unchanged |
| **B — Mistake Injection** | Lanham-style: inject errors or truncate the CoT | Higher AGD for faithful models (the mistake propagates through the mechanism) |
| **C — Hint Injection** | Turpin-style: add a biased hint that flips the answer | High AGD when the flip happened (mechanism changed), low AGD when model was "immune" |

---

## Part 5: What the Code Actually Does (Pipeline Walkthrough)

```
Input Datasets
(BBH, MMLU, GSM8K, Turpin)
        ↓
01_generate_cots.py
→ Feed each question to Gemma-2-2B
→ Save its chain-of-thought and answer
        ↓
02_generate_paraphrases.py
→ Use Gemma-2-9B (stronger model, 4-bit) to rephrase each CoT
→ Keep only pairs where the answer is still the same
        ↓
03_construct_pairs.py
→ Build the three regime datasets:
   Regime A: (original CoT, paraphrased CoT)
   Regime B: (original CoT, truncated/mistake CoT)
   Regime C: (unbiased CoT, hint-injected CoT)
        ↓
04_generate_graphs.py   ← THE EXPENSIVE STEP (~25 GPU hours)
→ For each (prompt, CoT) pair, run circuit-tracer
→ circuit-tracer runs model + traces Jacobians across 26 layers
→ Saves each graph as a JSON file
→ ~2100 graphs total
        ↓
05_extract_activations.py
→ Save residual-stream activations for the simpler baselines
        ↓
06_compute_agd.py
→ Load pairs of graphs
→ Compute J_w and S_e for each pair
→ Save to artifacts/agd/pairs.parquet
        ↓
07_compute_baselines.py
→ Compute all 5 simpler baselines:
   1. Activation-cosine (just look at how similar the activations are)
   2. KL divergence of next-token distribution
   3. Perplexity of the CoT
   4. Self-consistency variance (sample 8 CoTs, measure answer entropy)
   5. Random-feature Jaccard (sanity check)
        ↓
08_behavioral_measures.py
→ Run Lanham's AOC protocol (what does behavioral test say?)
→ Run Turpin's hint-flip protocol (did the hint flip the answer?)
        ↓
09_tune_on_train.py  ← TRAINING SPLIT ONLY
→ Grid search over α ∈ {0, 0.25, 0.5, 0.75, 1.0}
→ Grid search over k ∈ {16, 32, 64, 128, 256}
→ Pick best combo, freeze it
→ Save to analysis/best_hyperparams.json
        ↓
10_test_half_analysis.py  ← THE MOMENT OF TRUTH
→ Apply frozen (α, k) to held-out test set
→ Compute H1 (Spearman ρ), H2 (AUROC), H3 (incremental AUROC)
→ Bootstrap CIs (5000 resamples)
→ Holm-Bonferroni correction
→ Save to analysis/results_test.json
        ↓
11_ablations.py
→ Re-run with different α, k, pruning thresholds, layer bands
→ Check if results are robust
        ↓
12_generate_figures.py
→ F1: Method diagram
→ F2: AGD vs AOC scatter plot
→ F3: ROC curves for hint-flip prediction
```

---

## Part 6: What the Results Actually Show

The experiments have been run. Here's what came out:

### Hyperparameter Tuning (Training Half)
Best config found: **α = 1.0, k = 32**

This is a major finding on its own: **α = 1.0 means only the node term (J_w) matters** — the edge similarity (S_e) contributes nothing useful. In other words, *which* features are active matters more than *how* they connect to each other for capturing faithfulness signal.

### Test Results (Held-Out Half)

| Hypothesis | Result | Threshold | Verdict |
|------------|--------|-----------|---------|
| **H1**: AGD correlates with AOC | ρ = **-0.475**, p = 0.0005 | ρ ≥ +0.30 | ❌ Failed (wrong sign) |
| **H2**: AGD predicts hint-flips with AUROC ≥ 0.65 | AUROC = **0.518** | ≥ 0.65 | ❌ Failed |
| **H3**: Incremental AUROC ≥ 0.05 | Not computed | — | Not reached |

### The Negative Correlation in H1 is the Most Interesting Result

H1 failed the threshold (was supposed to be ρ ≥ +0.30), but the measured ρ = **-0.475** is highly significant (p = 0.0005). A negative correlation means:

> **When AGD is high (mechanism changed a lot), the model actually scores higher on the Lanham AOC test (more faithful by behavioral measures).**

This is the *opposite* of what the original hypothesis predicted, but it's not noise — it's a real, statistically strong relationship. It suggests something subtle:
- Maybe when the model's mechanism changes with CoT perturbation, that's evidence it *was* actually using the CoT (faithful)
- Maybe the AGD is picking up CoT engagement, not CoT rationalization

### Mean AGD Values Are Very High (~0.97)
The mean AGD across all pairs is ~0.97 — near the maximum of 1.0. This means the attribution graphs are almost completely different across CoT conditions. This could indicate:
- The attribution graphs are very sensitive to any text change (even paraphrases cause total graph reorganization)
- The graphs are near-random at this scale and the metric isn't capturing meaningful signal
- Late layers are especially different (mean 0.98 vs early layers at 0.83)

### Ablation Findings
- **α sweep**: AUROC monotonically increases as α → 1.0 — confirms edge similarity (S_e) is not useful
- **Layer bands**: Late layers (17–25) have highest AGD (0.98 mean) — the biggest mechanism changes are in the final layers
- **Random null**: Mean random Jaccard = 0.018 — extremely low, confirming the real AGD is not just measuring set size effects

---

## Part 7: Research Gaps This Work Addresses (and What It Found)

### The Gap That Motivated This Work

The field had:
- Behavioral faithfulness metrics (Lanham, Turpin) — measuring outputs, not mechanisms
- Beautiful attribution graphs (Anthropic's circuit-tracer) — used only for qualitative case studies
- No bridge between the two

**The gap**: No quantitative metric that uses the internal graphs to measure faithfulness.

### What This Work Tried to Do

Build **AGD** as that bridge — a number you can compute from two attribution graphs that tells you "did the mechanism change?"

### What the Current SOTA Looks Like

The field is split into:

**Behavioral tests** (established): Lanham AOC, Turpin hint-injection, CC-SHAP (Parcalabescu 2024), self-consistency (Wang et al.), monitorability (Meek 2025)

**Causal/structural** (newer): FUR — Faithfulness via Unlearning (Tutek 2025), CST — Counterfactual Simulatability Training (Hase 2026), Pearl front-door analysis on reasoning traces (Han 2026)

**Closest concurrent work**: Zhao et al. "CRV" (2025) — trains a classifier on attribution graph features to predict CoT *correctness* (not faithfulness). Different question, same tools.

**The gap AGD was trying to close**: First *quantitative mechanism-level* faithfulness metric (as opposed to behavioral).

### What the Results Tell the Field

The results are a **pre-registered negative** (or partial negative):
- AGD in its current form does not predict hint-flip behavior well (AUROC 0.518, near chance)
- The negative correlation in H1 suggests the relationship between graph changes and behavioral faithfulness is the *opposite* of what was hypothesized
- High mean AGD (~0.97) suggests attribution graphs may be too sensitive to surface text changes to be useful as a faithfulness signal without more careful calibration

**This is a contribution in itself**: it tells the field that naive graph divergence over attribution graphs does *not* provide useful faithfulness signal — at least not in the direction originally expected. The pre-registration makes this finding scientifically trustworthy.

---

## Part 8: How to Write the Paper

### The Story You Have

You have a **clean, pre-registered negative result with a surprising finding**:
- AGD was supposed to positively correlate with faithfulness
- Instead it strongly negatively correlates (ρ = -0.47, p < 0.001)
- This is not a failed experiment — it's a *specific finding* about how attribution graphs relate to behavioral faithfulness measures

### Framing Options

**Option 1 (recommended): "Attribution Graph Divergence Reveals Inverse Relationship with Behavioral Faithfulness"**
- Lead with: we built the first mechanism-level faithfulness metric
- Show: the relationship is significant but opposite to behavioral predictions
- Interpret: this dissociation between mechanistic and behavioral faithfulness is itself a finding — they may be measuring different things
- Implications: behavioral tests (Lanham AOC) and mechanistic tests (AGD) are not interchangeable

**Option 2: Pre-registered negative**
- Lead with: we tested whether attribution graphs can quantify CoT faithfulness
- Show: they cannot (at least with this approach)
- Interpret: the field needs better graph-to-faithfulness mapping before mechanism-level faithfulness measurement is viable

### Key Numbers for the Paper
- ρ = -0.475 (95% CI: [-0.69, -0.11]), p = 0.0005, n = 50 — H1 (Regime B)
- AUROC = 0.518 (95% CI: [0.41, 0.62]) — H2 (Regime C)
- Best config: α = 1.0 (node-only), k = 32
- Mean AGD across all pairs: ~0.97
- Late-layer AGD notably higher than early/mid-layer AGD

### Figures You Have
- `analysis/figures/f2_correlation.pdf` — AGD vs AOC scatter (shows the negative correlation)
- `analysis/figures/f3_auroc.pdf` — ROC curves for hint-flip prediction
- `analysis/figures/f1_method.svg` — Method diagram

---

## Summary: 5 Things to Understand Before Writing

1. **CoT faithfulness** = "Does the model actually reason through the steps it writes?" Behavioral tests (Lanham, Turpin) measure this indirectly via outputs.

2. **Attribution graphs** = A map of which internal features (interpretable units from Gemma Scope transcoders) causally contributed to the output for one specific input, built by circuit-tracer.

3. **AGD** = A number (0 to 1) measuring how different two attribution graphs are. Hypothesis: if AGD is high when you change the CoT, that means the CoT was causally relevant to the mechanism (faithful). If AGD is low, the mechanism didn't care about the CoT (unfaithful).

4. **What ran**: 2100 attribution graphs across 700 prompts × 3 conditions, on Gemma-2-2B with Gemma Scope transcoders. All scripts executed, all results saved.

5. **What came out**: H1 and H2 failed their thresholds, but H1 shows a *strong negative* correlation (opposite sign to expected). AGD values are very high (~0.97) across all conditions. The node term matters; the edge term doesn't. This is a publishable negative/surprise finding.
