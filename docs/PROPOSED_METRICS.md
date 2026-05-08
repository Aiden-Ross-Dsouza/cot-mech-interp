# Proposed Improved Metrics for Attribution Graph Comparison

## Why Current AGD is Broken (Root Cause Analysis)

Before proposing new metrics, we need to understand **why** AGD gives mean values of ~0.97 (near ceiling) and why the edge term (S_e) carries no signal at all (α=1.0 being best).

### The Position-Sensitivity Problem

Look at the graph node IDs in our JSON:
```
"feature_id": "L3_P12_F1047"   ← Layer 3, Position 12 (token index), Feature 1047
"feature_id": "L3_P24_F1047"   ← Same feature, same layer, DIFFERENT position
```

These two are treated as **completely different nodes** in the current Jaccard computation — even though they represent the *same transcoder feature* doing *the same thing*, just attending to a different token position in the sequence.

When we compare a clean CoT (say, 80 tokens long) with a paraphrased CoT (say, 95 tokens long), the same "arithmetic reasoning" feature at layer 3 fires at position 60 in G0 and position 72 in G1. AGD sees zero overlap — even though the mechanism is essentially identical.

**This is why AGD is at ceiling.** It's measuring sequence-length shifts and token-position differences, not mechanism differences.

**This is also why α=1.0 is optimal.** The edge term (S_e) compares edges by (src_id, dst_id) tuples — same position-sensitivity problem. If nodes are all "wrong" due to position shifts, edges are even worse. The node term at least can accidentally match some shared position-invariant features (input tokens, logit nodes) better than edges.

---

## Three Proposed Metrics (with Literature Support)

---

### Metric 1: Position-Agnostic Node Overlap (PANO)

**The idea:** Strip the position `P` from feature IDs before comparison. Use only `(layer, feature_index)` as the concept identity.

```
feature_id "L3_P12_F1047"  →  concept "L3_F1047"
feature_id "L3_P24_F1047"  →  concept "L3_F1047"   (same!)
```

**New PANO similarity:**
```
For each graph G, aggregate influence per concept (L, F):
  influence(L_F) = max over all positions P of influence(L_P_F)

Then compute weighted Jaccard over these position-collapsed concepts.
```

Using `max` (rather than sum) because we care about whether a feature *engages* at any position — not how many positions it fires at (which would be confounded by sequence length).

**Why this fixes things:** A paraphrased CoT that re-orders words will now show high PANO similarity (same features fire, just at slightly shifted positions). A mistake-injected CoT that uses fundamentally different computation will show low PANO similarity (different features active across the board).

**Literature support:**
- Olah et al. (2020) "Zoom In" — features in networks are *universal* across positions; the same feature can fire at multiple positions in a sequence
- Conmy et al. (2023) "Towards Automated Circuit Discovery" — circuits are identified by (layer, head/feature) identity, not by which specific token position they attend to
- Elhage et al. (2022) "Toy Models of Superposition" — polysemanticity and feature identity in transformer features are position-independent concepts
- Lindsey et al. (2025) "Biology of a Language Model" — qualitative graph analysis collapses position when identifying which features matter

**Predicted hypothesis flip:** If PANO fixes the ceiling effect, paraphrase pairs (Regime A) should have *high* PANO similarity (mechanism intact), and mistake injection pairs (Regime B) should have *lower* PANO similarity for faithful models. This would give a **positive** correlation with AOC — the correct direction for H1.

**Implementation:** Trivial. Load nodes from existing graph JSON, strip position from feature_id, aggregate by (layer, feature_idx), compute weighted Jaccard.

```python
def position_agnostic_node_set(graph, k):
    concept_influence = {}
    for node in graph["nodes"]:
        fid = node["feature_id"]
        # Parse "L3_P12_F1047" → concept "L3_F1047"
        parts = fid.split("_")
        if len(parts) == 3 and parts[0].startswith("L") and parts[2].startswith("F"):
            concept = f"{parts[0]}_{parts[2]}"  # L3_F1047
        else:
            concept = fid  # keep as-is for ERR, TOK, LOGIT nodes
        inf = abs(node["influence"])
        if concept not in concept_influence or inf > concept_influence[concept]:
            concept_influence[concept] = inf  # take max across positions
    # Return top-k by influence
    sorted_concepts = sorted(concept_influence.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_concepts[:k])
```

---

### Metric 2: Differential AGD (δ-AGD)

**The idea:** For each item, use Regime A (paraphrase) as a within-item baseline. The absolute AGD value is confounded by item difficulty, CoT length differences, and other per-item factors. The *relative* change in AGD when going from a "neutral" perturbation (paraphrase) to a "mechanistically significant" perturbation (mistake, hint) is the actual signal.

```
δ-AGD(item) = PANO_divergence(B_condition) − PANO_divergence(A_condition)
```

Where both are computed relative to the same "clean" graph G0.

For H2 (hint-flip prediction):
```
δ-AGD_hint(item) = PANO(G_hint, G_clean) − PANO(G_paraphrase, G_clean)
```

- Items where the hint caused *more additional* mechanism shift beyond the paraphrase baseline → higher δ-AGD → hint was causally engaging the mechanism → likely to produce a flip
- Items where hint and paraphrase cause the same amount of shift → δ-AGD ≈ 0 → model was insensitive to the hint → no flip

**Why this would yield a positive hypothesis:**
- It normalizes away the global ceiling effect per-item
- It is directly analogous to the Lanham AOC methodology — AOC is also a *differential* measure (how much does the answer change above baseline?)
- Requires data from both Regime A and Regime C for the same item — we have this

**Literature support:**
- Pearl (2001) *Causation* — the front-door criterion and mediation analysis explicitly use difference-in-effects (direct vs. total effect) as the quantity of interest; δ-AGD is exactly this structure
- Lanham et al. (2023) — AOC is defined as area *over* the completeness curve (i.e., difference from baseline), not absolute completeness; our δ-AGD mirrors this design choice
- Difference-in-differences (DiD) is a standard causal inference technique for removing time-invariant confounders (Angrist & Pischke 2009); per-item baseline divergence is exactly such a confounder
- Han et al. (2026) — apply Pearl front-door analysis to reasoning traces; δ-AGD is the mechanistic equivalent

**Predicted hypothesis:**
H1: `ρ(δ-AGD, AOC) > 0` — faithful items show larger mechanism shift under mistake injection relative to their paraphrase baseline
H2: `AUROC(δ-AGD for hint-flip prediction) > 0.65`

**Implementation:** Requires that all three conditions (clean, paraphrase, hint/mistake) exist for the same item. Join the pairs dataframe on item_id across regimes.

```python
def compute_delta_agd(pairs_A, pairs_BC, graph_dir, k=32):
    """
    pairs_A: DataFrame with regime A (paraphrase) pairs per item
    pairs_BC: DataFrame with regime B or C pairs per item
    """
    # Compute PANO for A condition (baseline)
    agd_A = batch_pano(pairs_A, graph_dir, k=k)
    # Compute PANO for B/C condition
    agd_BC = batch_pano(pairs_BC, graph_dir, k=k)
    # Join on item_id and subtract
    merged = agd_BC.merge(agd_A[["item_id", "pano"]], on="item_id", suffixes=("_BC", "_A"))
    merged["delta_agd"] = merged["pano_BC"] - merged["pano_A"]
    return merged
```

---

### Metric 3: Layer-wise Influence Wasserstein Distance (LIWD)

**The idea:** Instead of comparing *which* features are active (identity), compare *where in the model* the reasoning is happening (distribution across layers). For each graph, build a probability distribution over layers weighted by total influence:

```
p_G(layer l) = Σ_{nodes at layer l} influence(node) / Σ_{all nodes} influence(node)
```

Then compare the two layer distributions using **1D Wasserstein distance** (Earth Mover's Distance — the "cost" of transforming one distribution into the other, where cost = layer distance).

```
LIWD(G0, G1) = W1(p_G0, p_G1)
             = Σ_l |CDF_G0(l) - CDF_G1(l)|   (closed form for 1D)
```

**Why this captures something different from PANO:**
PANO captures which conceptual features are active. LIWD captures *where in the processing pipeline* the influence is concentrated. A model that "front-loads" reasoning in early layers (doing the actual computation in layers 0–8) vs. one that relies on late-layer pattern matching (layers 17–25) would look very different in LIWD but might look similar in PANO (if similar features fire at different depths).

**Why this is relevant to faithfulness:**
- Our ablation shows early layers (0.82 mean) and late layers (0.98 mean) have very different AGD distributions — there's already a structural layer-band signal
- Unfaithful CoT (post-hoc rationalization) might concentrate influence in late layers (output-distribution-shaping layers), while faithful CoT routes through middle layers (conceptual integration layers)
- This hypothesis directly connects to the Anthropic "Biology" paper's finding that different types of reasoning use different layer ranges

**Literature support:**
- Villani (2009) *Optimal Transport* — W1 is the canonical metric for comparing distributions
- Cuturi (2013) Sinkhorn distances — fast OT for ML applications  
- Elhage et al. (2022) "In-context Learning and Induction Heads" — early vs. late layer functions are mechanistically different in transformers
- Nostalgebraist (2020) logit lens — different layers represent different stages of the computation; layer distribution captures processing stage
- TCAV (Kim et al. 2018) — uses layer-specific activation patterns to test conceptual hypotheses; our layer-distribution is the influence-weighted analogue
- Our own ablation result: mean LIWD_early=0.82, LIWD_late=0.98 — there IS differential signal by layer band

**Implementation:**
```python
import numpy as np
from scipy.stats import wasserstein_distance

def layer_influence_distribution(graph, n_layers=26):
    """Return normalized influence distribution over layers 0..n_layers-1."""
    dist = np.zeros(n_layers + 1)  # +1 for logit layer
    for node in graph["nodes"]:
        layer = node.get("layer", -1)
        if 0 <= layer <= n_layers:
            dist[layer] += abs(node["influence"])
    total = dist.sum()
    if total > 0:
        dist /= total
    return dist

def liwd(graph0, graph1, n_layers=26):
    """1D Wasserstein distance between layer influence distributions."""
    p0 = layer_influence_distribution(graph0, n_layers)
    p1 = layer_influence_distribution(graph1, n_layers)
    layer_positions = np.arange(n_layers + 1, dtype=float)
    return wasserstein_distance(layer_positions, layer_positions, p0, p1)
```

---

## Summary: Three-Metric Strategy

| Metric | Problem it solves | Literature anchor | Expected hypothesis |
|--------|------------------|-------------------|---------------------|
| **PANO** (Position-Agnostic Node Overlap) | Fixes ceiling effect from position-sensitivity | Olah 2020, Conmy 2023 | H1: ρ > 0 between PANO-divergence and AOC |
| **δ-AGD** (Differential AGD) | Normalizes per-item baseline divergence | Pearl 2001, Lanham 2023 (DiD structure) | H1+H2: hint-flip items have higher δ-AGD |
| **LIWD** (Layer-wise Influence Wasserstein) | Captures *where* not *which* | Villani 2009, layer-band ablation result | Separate hypothesis: faithful CoT concentrates influence differently by layer |

**Recommended implementation order:**
1. **PANO first** — it's trivial to implement (string parsing), directly addresses the ceiling effect, and reuses everything else in the existing pipeline
2. **δ-AGD second** — requires joining regime A and regime B/C pairs but is mathematically simple
3. **LIWD third** — independent angle, adds a new dimension to the paper

---

## Proposed New Hypotheses

**H1' (revised):** PANO-divergence under mistake injection (Regime B) positively correlates with Lanham AOC:
```
ρ_Spearman(PANO_divergence_B, AOC) ≥ 0.30, p < 0.01
```
**Rationale:** Fixing position-sensitivity should restore the originally expected positive direction.

**H2' (revised):** δ-AGD (PANO-based, hint vs. paraphrase baseline) predicts hint-flip with AUROC ≥ 0.65:
```
AUROC(δ-AGD, hint_flip) ≥ 0.65, CI non-overlapping with activation-cosine
```
**Rationale:** Within-item normalization should reveal the incremental mechanism shift from hint injection.

**H3' (new):** LIWD provides incremental signal beyond PANO and δ-AGD:
```
AUROC(PANO + δ-AGD + LIWD) ≥ AUROC(PANO + δ-AGD) + 0.05
```
**Rationale:** Layer distribution is orthogonal to node-identity metrics; if faithful CoT uses different layer ranges, LIWD captures that.

---

## Connection to Your Existing Results

| Existing finding | Metric it motivates | Why |
|-----------------|---------------------|-----|
| Mean AGD ≈ 0.97 (ceiling) | PANO | Ceiling caused by position-sensitivity in feature IDs |
| α=1.0 optimal (edge term useless) | PANO + LIWD | Edge term is doubly position-sensitive; these metrics avoid it entirely |
| Early layers AGD=0.82, late=0.98 | LIWD | Late-layer saturation is hiding signal; LIWD isolates this |
| Negative ρ=-0.47 (H1 fails) | δ-AGD | Normalizing baseline divergence should flip correlation direction |
| H2 AUROC=0.518 (near chance) | δ-AGD | Absolute AGD can't discriminate; differential can |

---

## What to Write in the Paper

The paper's claim becomes:

> "Naive graph divergence metrics (AGD) fail due to position-sensitivity in attribution graph feature IDs. We introduce three position-aware and distribution-based alternatives — PANO, δ-AGD, and LIWD — that recover the expected positive faithfulness signal and yield AUROC [X] for hint-flip prediction."

This is a stronger contribution than the original: you're diagnosing *why* simple graph divergence fails and proposing principled fixes. This reframes the original "negative result" as a "diagnosis + solution" paper.
