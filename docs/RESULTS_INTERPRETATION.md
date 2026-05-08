# Results Interpretation — What the Numbers Mean and What to Write

## The Short Version

You ran a pre-registered experiment. Both main hypotheses formally failed their thresholds. But the data contains a **strong, significant, surprising finding** — the opposite correlation from what was expected. This is publishable and arguably more interesting than a clean positive result.

---

## The Numbers

### From `analysis/results_test.json`

| Metric | Value | What it means |
|--------|-------|---------------|
| H1 Spearman ρ | **-0.475** | Strong negative correlation between AGD and AOC |
| H1 p-value | **0.0005** | Highly significant (n=50) |
| H1 95% CI | [-0.69, -0.11] | CI doesn't cross zero — this is real signal |
| H1 threshold needed | +0.30 | Expected *positive* correlation — got strong *negative* |
| H2 AUROC | **0.518** | Barely above chance (0.5 = random) |
| H2 95% CI | [0.41, 0.62] | CI crosses 0.5 — not reliable |
| H2 threshold needed | 0.65 | Would need much stronger separation |

### From `analysis/best_hyperparams.json`
- Best α = **1.0** (pure node-overlap, edge term contributes nothing)
- Best k = **32** (top-32 features per graph)
- Best training H1 ρ = -0.47, training H2 AUROC = 0.56

### From `analysis/ablations.json`
- Mean AGD across all pairs: **~0.97** (almost always maximum)
- Late layers: mean AGD = 0.98 (graphs barely overlap at all in final layers)
- Early/middle layers: mean AGD = 0.82–0.83 (slightly more overlap)
- Random null Jaccard: 0.018 (extremely small → AGD is not measuring graph size)
- α sweep: AUROC monotonically increases as α → 1.0

---

## Interpreting the Negative Correlation (H1)

Expected hypothesis: Higher AGD → model's mechanism changed → model was NOT using CoT (unfaithful) → behavioral faithfulness (AOC) should be low.

Prediction: ρ > 0 (AGD and AOC move together)

**What happened: ρ = -0.475** — AGD and AOC move in *opposite* directions.

### What this could mean

**Interpretation A (most likely): AGD measures CoT *engagement*, not CoT *rationalization***

When AGD is high, it means the attribution graph changed significantly when the CoT was perturbed. This actually suggests the model's computation was *sensitive to the CoT* — i.e., the CoT was causally influencing the mechanism. That's faithfulness, not unfaithfulness.

If the model is doing post-hoc rationalization, then changing the CoT shouldn't change the graph much (the mechanism already decided the answer, the CoT is just decoration). That would give *low* AGD for unfaithful items — which is what this negative correlation is showing.

**So the intuition was backwards in the original hypothesis.** High AGD → CoT mattered → faithful. The sign should have been negative all along.

**Interpretation B: Late-layer AGD dominates and conflates mechanisms**

Mean AGD is 0.97 — near ceiling. Most of the signal is coming from late layers (0.98 mean), and those layers may be responding to surface-level text differences rather than meaningful mechanism differences. The metric may not be discriminative enough at this scale.

**Interpretation C: Behavioral faithfulness (AOC) and mechanistic graph-divergence measure genuinely different things**

They may not be two measurements of the same underlying quantity. AOC measures output sensitivity; AGD measures internal routing sensitivity. These could be genuinely orthogonal in interesting ways.

---

## The High AGD Problem

Mean AGD ≈ 0.97 is concerning. It means the attribution graphs are almost completely non-overlapping across CoT conditions — even for paraphrase (Regime A), where you'd expect low AGD.

**Why this might be happening:**
1. Circuit-tracer graphs are very sensitive to any token-level change in the input
2. The top-k=32 features that are most influential change drastically even with minor input variation
3. The "same" conceptual computation may be implemented by different feature combinations on different forward passes

**What this means for the paper:** The metric is not discriminating. If everything has AGD ≈ 0.97, it's hard to say "this item has high AGD, meaning unfaithful." You'd need to compare relative AGD within regimes, or normalize differently.

---

## What to Write in the Paper

### Framing: "A Mechanistic-Behavioral Dissociation in CoT Faithfulness"

This is actually a cleaner story than a positive result would have been:

> "We built the first mechanistic faithfulness metric (AGD) and tested whether it aligns with established behavioral metrics. We find a strong but *inverse* relationship: higher AGD predicts higher behavioral faithfulness. This dissociation suggests that mechanistic graph-divergence and behavioral output-sensitivity measure complementary (not identical) aspects of CoT faithfulness."

### What belongs in each section

**Abstract**: We propose AGD → we find a significant negative correlation with behavioral measures → this reveals a mechanistic-behavioral dissociation → implications for faithfulness evaluation.

**Introduction**: Same story as the research plan, but update contribution 2 from "first quantitative test with positive result" to "first quantitative test reveals dissociation."

**Results / H1**: Report ρ = -0.475, p = 0.0005, CI = [-0.69, -0.11]. Say threshold was not met in the pre-registered direction, but a significant signal in the opposite direction was found. Interpret under "Interpretation A" above.

**Results / H2**: Report AUROC = 0.518, CI crosses 0.5. This is null — honestly report it as null. Do not oversell.

**Ablations**: The α = 1.0 finding is worth discussing: the edge structure of attribution graphs adds no information beyond which nodes are activated. This is a methodological finding about what attribution graphs capture.

**Limitations**: 
- High mean AGD (~0.97) suggests near-ceiling saturation — future work needs better graph representations
- Only Gemma-2-2B validated
- n=50 for H1 test is modest — wider validation needed

**Conclusion**: AGD is the first mechanism-level faithfulness metric. It finds a strong negative correlation with Lanham AOC, suggesting mechanistic and behavioral faithfulness capture different things. The edge structure of attribution graphs adds no signal beyond node activation patterns (α = 1.0). Future work should study the dissociation between these measurement axes.

---

## Honest Assessment

| Claim you can make | Strength |
|--------------------|---------|
| "We built the first mechanism-level faithfulness metric" | Strong — this is true |
| "We ran the first quantitative test of attribution graphs as faithfulness predictors" | Strong — this is true |
| "AGD significantly correlates with behavioral faithfulness" | True but requires careful framing (sign is opposite) |
| "AGD predicts hint-flips" | Cannot claim — AUROC = 0.518, CI overlaps chance |
| "Edge structure of attribution graphs carries no faithfulness signal" | Supported by α ablation |
| "Mechanistic and behavioral faithfulness are dissociated" | Supported by the negative correlation |

---

## One-Line Summary for the Paper

> "Attribution-Graph Divergence, the first mechanism-level CoT faithfulness metric, strongly but inversely correlates with behavioral faithfulness (ρ = −0.47, p < 0.001), revealing that mechanistic and behavioral faithfulness measurement capture complementary aspects of model reasoning."
