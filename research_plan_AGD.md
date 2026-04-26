# Research Plan: AGD — A Paired-Graph Divergence Metric for Chain-of-Thought Faithfulness

**Target:** ICML 2026 Mechanistic Interpretability Workshop, short paper (4 pages, ICML format).
**Deadline:** 8 May 2026 AOE.
**Team:** 2 people.
**Hardware:** 1 × Quadro RTX 6000 (24 GB).
**Status:** Pivoted away from a too-close concurrent paper (Zhao et al., 2510.09312, "CRV"); see §3.

---

## 0. TL;DR

We propose **Attribution-Graph Divergence (AGD)**: a metric over pairs of pruned attribution graphs that quantifies how much a model's *internal mechanism* shifts when its visible chain-of-thought is perturbed in ways that should be either causally relevant (truncations, mistake injections) or causally inert (semantic-preserving paraphrases, hidden-cue insertions that flip the answer).

The central claim is: **AGD measures CoT faithfulness at the mechanism level, predicts behavioral unfaithfulness signals (Lanham-AOC, Turpin-style hint flips) better than activation-distance baselines, and provides the missing causal-mechanistic counterpart to the existing behavioral faithfulness literature.**

The 4-page paper has three figures and a small table:
- **F1** Method diagram (graph-pair construction + AGD definition).
- **F2** AGD vs. behavioral faithfulness scatter, with bootstrap CIs.
- **F3** AGD's incremental AUROC for hint-flip prediction over baselines.
- **T1** Per-task results across BBH/MMLU/GSM8K, with ablations.

---

## 1. Problem formulation and hypotheses

### 1.1 Why this problem matters

Chain-of-thought reasoning is the dominant inference-time mechanism in modern LMs and the primary surface that humans, monitors, and safety pipelines read. Lanham et al. (2023), Turpin et al. (2023), Anthropic's "Reasoning Models Don't Always Say What They Think" (2025), and Barez et al. (2025, *Chain-of-Thought is Not Explainability*) have established that CoT can be *unfaithful*: the visible reasoning does not always reflect the model's actual computation.

Existing faithfulness measurement is almost entirely *behavioral*: perturb the text, see if the answer changes. This is necessarily indirect — it cannot distinguish:

- **(A)** "model used a different mechanism than the words suggest" (deep unfaithfulness)
- **(B)** "model used the same mechanism but the perturbation was robust enough that the output didn't flip" (surface insensitivity)

Only a *mechanistic* measurement can. Anthropic's circuit-tracing work (Ameisen et al., 2025; Lindsey et al., 2025) provides the first scalable tool — pruned attribution graphs over transcoder features — capable of producing a per-prompt mechanistic fingerprint. To date these graphs have been used for qualitative case studies of unfaithfulness ("Biology of an LLM"). They have **not** been operationalized as a quantitative faithfulness metric.

This is the hole we close.

### 1.2 Formal definitions

Let $M$ be a transformer language model with an associated transcoder dictionary, and let $G(M, x, y)$ be the pruned attribution graph produced by `circuit-tracer` for input $x$ and target output token $y$. Each graph is a DAG with feature-nodes, error-nodes, embedding-nodes and a target-logit node, with weighted directed edges representing linear feature-to-feature attributions.

For a base prompt $x$ with model-generated CoT $c$ and answer $a$, define a **paired-graph experiment** $\mathcal{E} = (x, c, c', y)$ where $c'$ is a perturbation of $c$ and $y$ is the answer token.

We compute $G_0 := G(M, [x \,\|\, c], y)$ and $G_1 := G(M, [x \,\|\, c'], y)$ over a *shared feature space* (same transcoder dictionary), and define:

$$
\mathrm{AGD}(\mathcal{E}) = 1 - \alpha \cdot J_w(\mathcal{N}(G_0), \mathcal{N}(G_1)) - (1-\alpha) \cdot S_e(\mathcal{E}(G_0), \mathcal{E}(G_1))
$$

where:
- $\mathcal{N}(G)$ is the top-$k$ feature node set ranked by influence-on-target (we use $k=64$ as default; ablate);
- $J_w$ is the *influence-weighted Jaccard*: $\sum_{f \in \mathcal{N}_0 \cap \mathcal{N}_1} \min(w_0(f), w_1(f)) \big/ \sum_{f \in \mathcal{N}_0 \cup \mathcal{N}_1} \max(w_0(f), w_1(f))$;
- $S_e$ is an edge-overlap measure: cosine similarity over the vector of (source, target)-pair attributions restricted to the union of top edges;
- $\alpha = 0.5$ default; ablate.

AGD ∈ [0, 1]. AGD ≈ 0 means the two graphs use the same features and edges in the same proportions (mechanism unchanged). AGD → 1 means the mechanism has reorganized.

### 1.3 Hypotheses

**H1 — Faithfulness signal (core).** For prompts where Lanham-style behavioral tests indicate post-hoc / unfaithful reasoning, AGD between the original CoT and a paraphrased-but-same-answer CoT is *higher* than for prompts where behavior says the CoT was used. Formally:
$$
\rho_{\text{Spearman}}(\mathrm{AGD}, \, 1 - \mathrm{AOC}_{\text{Lanham}}) \geq 0.30 \text{ at } p < 0.01.
$$
**Falsified if** $|\rho| < 0.15$ on the held-out task partition with bootstrap-CI excluding 0.20.

**H2 — Predictive power for hint-induced flips.** In Turpin's hint-injection setup, AGD between (biased prompt, model CoT) and (unbiased prompt, model CoT for the same item) predicts whether the hint will *flip* the answer. Formally, AGD's ROC AUC ≥ 0.65 with a 95% bootstrap CI lower bound > activation-cosine baseline upper bound (i.e., non-overlapping CIs, not just point-estimate dominance).

**Falsified if** AGD AUROC ≤ 0.55 OR CI overlap with activation-cosine > 50% of width.

**H3 — Incremental information beyond activations and behavioral baselines.** A logistic regression on (AGD + activation-cosine + perplexity + self-consistency-variance) achieves AUROC at least 0.05 above the same regression without AGD on the held-out test set, with bootstrap p < 0.05.

**Falsified if** the incremental AUROC is < 0.02 or its 95% bootstrap CI crosses 0.

**H4 — Optional / stretch (secondary contribution).** Edge-level reliability test. For each top-attribution edge in $G_0$, the predicted effect-size of feature-ablation matches the *measured* effect on the target logit with R² ≥ 0.5. (This is Direction 5 from the original sweep, folded in as a methodological appendix experiment if H1–H3 land.)

**Falsified if** R² < 0.3, in which case we report it as a known limitation of AGD's underlying signal — also publishable.

### 1.4 What "publishable negative" looks like

If H1, H2, and H3 *all* fail (probability we estimate at ~15%), the paper is reframed as:

> *"Even with cross-layer transcoder attribution graphs, a principled graph-divergence metric does not improve over activation-distance baselines for predicting CoT unfaithfulness. We provide the first quantitative test of this and discuss why structural information may not be the missing piece."*

This is a contribution: it tightens the field's epistemics in the direction Hendrycks & Hiscott (2025) and Charbel-Raphaël (2023) argued for. We pre-register this framing in the abstract draft we prepare on Day 3.

---

## 2. Core claim and intuition

**Claim.** Whether a model's verbalized CoT is faithful is empirically equivalent to asking whether the model's *internal computation* depends on the verbalized CoT in the way the words claim. Pruned attribution graphs are a (lossy, but available) summary of internal computation. So a quantitative comparison of paired graphs — original CoT vs. perturbed CoT — is a *direct mechanistic operationalization* of faithfulness. This is the operationalization the field has been gesturing at since Anthropic's "Biology" paper but has not built.

**Intuition by example.**
- *Faithful CoT*: Q="Bob has 3 apples and 2 pears. Total fruit?" CoT="3+2=5". Replace CoT with "I rolled a die and got a 5." Both produce answer 5. Behaviorally, this looks like the model "ignored" the CoT (= unfaithful). Mechanistically, the second case routes through different features (a "guess from context" feature, or a refusal-of-arithmetic feature), so $G_0$ and $G_1$ differ → high AGD. Behavioral and mechanistic measures agree.
- *Genuinely post-hoc*: Q with a hint "I think it's (B)" — model picks B, CoT rationalizes B without mentioning the hint. AGD between graphs computed under (biased CoT, biased prompt) and (biased CoT, unbiased prompt) reveals that the answer-determining features include the hint token's contribution — but the verbalized CoT doesn't reference it. This is the smoking gun mechanistic evidence Turpin's behavioral test only inferred indirectly.

**Why this is feasible now.** Three things came together in 2025: (i) `circuit-tracer` is open-source and runs on Gemma-2-2B / Llama-3.2-1B / Qwen-3-4B; (ii) Gemma Scope transcoders are pre-trained and public; (iii) the QK attribution extension (Lindsey et al., July 2025) makes attention-mediated edges interpretable too. None of these existed when Lanham wrote the original behavioral tests.

---

## 3. Positioning vs. prior work

### 3.1 The closest concurrent work — Zhao et al. (CRV, 2510.09312) — and how we differ

Zhao et al. (Meta FAIR + Edinburgh, Oct 2025; v2 Feb 2026) train a classifier on hand-engineered structural features of attribution graphs to predict per-step CoT *correctness*. They use the same tooling we plan to use (`circuit-tracer`, Gemma Scope transcoders).

**Differences (each is sharp, not cosmetic):**

1. **Different label.** They predict *correctness* of a reasoning step against ground truth. We measure *faithfulness* of CoT against the model's own internal computation. A correct step can be unfaithful (post-hoc rationalization that happens to be right); an incorrect step can be faithful (model genuinely reasoned, got it wrong). Turpin's foundational result is precisely this.
2. **Different mathematical object.** They produce a single classifier output per graph. We produce a paired-graph divergence — there is no notion of "two graphs to compare" in CRV.
3. **No counterfactual structure.** CRV's graphs are unconditioned. AGD is built on counterfactual pairings (clean vs. perturbed CoT, biased vs. unbiased prompt) — the same epistemic structure as Pearl front-door analysis used in Han et al. (2026) and Lanham (2023), but applied at the mechanism level.
4. **Different relation to existing literature.** CRV connects to the verifier / process-reward-modeling literature. AGD connects to the CoT-faithfulness literature (Lanham, Turpin, Tutek/FUR, Anthropic-Chen).

We will cite CRV in paragraph one of the related-work section: *"Concurrent work (Zhao et al., 2025) classifies CoT *correctness* from graph structure; we study the orthogonal axis of *faithfulness*, where graph structure is compared across paired counterfactual conditions."* This is honest and protective.

### 3.2 The rest of the related-work map (shortened for the plan; full version in paper)

- **Behavioral CoT-faithfulness:** Lanham et al. 2023 (early-answering, add-mistakes, paraphrase, filler-tokens AOC); Turpin et al. 2023 (hint injection); Atanasova et al. 2023; Parcalabescu & Frank 2024 (CC-SHAP); Meek et al. 2025 (monitorability + verbosity).
- **Causal-trace faithfulness:** Han et al. 2026 (Pearl front-door on structured traces); Tutek et al. 2025 (FUR — parametric faithfulness via unlearning); Hase & Potts 2026 (CST — training for counterfactual simulatability).
- **Attribution graphs as a tool:** Ameisen et al. 2025 (CLT methods); Lindsey et al. 2025 ("Biology"); Lindsey et al. July 2025 (QK attribution); EleutherAI Attribute library; Goodfire's "greater-than" replication.
- **Critiques motivating us:** Hendrycks & Hiscott 2025 ("Misguided Quest..."), Barez et al. 2025 ("CoT is not explainability"), Charbel-Raphaël 2023 ("Against Almost Every Theory of Impact of Interpretability").

**One-sentence positioning:** Behavioral tests measure faithfulness through outputs; AGD is the first mechanism-level metric, made possible by attribution graphs and made *necessary* by the criticisms that behavioral tests under-determine the diagnosis.

---

## 4. Method

### 4.1 The pipeline at architecture level

```
                  ┌──────────────────────────────────────────────────────┐
                  │ Prompt set X with paired conditions (clean / perturbed)
                  └────────────────────────┬─────────────────────────────┘
                                           ▼
                            ┌──────────────────────────┐
                            │ Gemma-2-2B (fp16)        │
                            │  + Gemma Scope PLT       │
                            │  on layers {0,4,…,24}    │
                            └──────────────┬───────────┘
                                           ▼
            ┌──────────────────────────────────────────────────────┐
            │ circuit-tracer attribution graph G(M, x, y)          │
            │   → prune to top-K nodes by influence-on-target      │
            │   → write JSON                                        │
            └──────────────┬───────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────────────────────┐
        │ Pair (G_0, G_1) per experimental condition            │
        │   compute AGD(G_0, G_1) = 1 − α·J_w − (1−α)·S_e       │
        └──────────────┬───────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────────┐
        │ Behavioral measurements (in parallel):                │
        │   Lanham AOC (early-answer, add-mistake, paraphrase) │
        │   Turpin hint-flip outcome                            │
        │   Activation-cosine baseline                          │
        │   Perplexity, self-consistency variance baselines     │
        └──────────────┬───────────────────────────────────────┘
                       ▼
        ┌──────────────────────────────────────────────────────┐
        │ Statistical analysis:                                 │
        │   Spearman ρ(AGD, behavioral)                         │
        │   AUROC for hint-flip prediction                      │
        │   Logistic-regression incremental AUROC               │
        │   5000-sample bootstrap CIs throughout                │
        └──────────────┬───────────────────────────────────────┘
                       ▼
              ┌──────────────────────┐
              │ 4-page paper         │
              └──────────────────────┘
```

### 4.2 Choice of model and dictionary

**Model: Gemma-2-2B (instruction-tuned).** Reasons:

- Officially supported by `circuit-tracer` from initial release (Hanna et al. 2025).
- Gemma Scope transcoders are publicly available, pre-trained on a large dataset, and have been validated in several follow-up replications (Goodfire, EleutherAI).
- Fits in 5 GB fp16 → leaves 19 GB headroom on Quadro 6000 for transcoder activations and graph computation.
- Recent enough that it does CoT respectably on BBH and GSM8K.

**Dictionary: Gemma Scope per-layer transcoders (PLT).** We default to PLT for two reasons:
1. CLT support in `circuit-tracer` is newer (added mid-2025) and is reportedly less stable; PLT is the documented path.
2. Per-layer features are slightly easier to interpret and less likely to introduce artifacts that masquerade as faithfulness signal.

Ablation: re-run the headline experiment with CLTs on a 100-prompt subset to check robustness. Time-box to 1 day.

**Why not Qwen-3-4B (4-bit) or Llama-3.2-1B?**
- Qwen-3-4B in 4-bit is supported but introduces quantization noise into the activations the transcoder reconstructs; not ideal for a *measurement* paper.
- Llama-3.2-1B is too small for reliable CoT on BBH/GSM8K — we'd be measuring noise.
- Robustness check: if H1 lands on Gemma, we re-run on Llama-3.2-1B for 100 prompts as a "this isn't model-specific" appendix paragraph.

### 4.3 Three task regimes (the paired conditions)

We need pairs of prompts/CoTs that produce the *same answer* but where we have an external reason to believe the *mechanism* may or may not differ. Three regimes provide this:

#### Regime A — Paraphrase invariance (Lanham-style)

For each prompt $x$, generate model CoT $c$ → answer $a$. Then construct $c'$ by paraphrasing $c$ via a stronger model (Gemma-2-9B-Instruct or external API; we use Gemma-2-9B locally to keep things reproducible — runs in 4-bit on the same GPU between attribution-graph passes).

Filter to pairs where (i) $c'$ leads to the same answer $a$ when prompted as $[x \,\|\, c'] \to ?$, and (ii) edit distance is large enough that we are not paraphrasing trivially. AGD for these pairs *should* be **low** (mechanism doesn't really change because the paraphrase didn't carry new info). Behavioral AOC for paraphrase is computed via the standard Lanham protocol.

#### Regime B — Mistake injection / truncation

Standard Lanham early-answer + add-mistake protocol. Construct truncated CoT $c'$ at fraction $\{25\%, 50\%, 75\%\}$ and add-mistake versions. Filter to pairs that *don't* flip the answer.

If the model is faithful, mistake injection should produce significant graph changes (because the mechanism propagates the wrong intermediate); if unfaithful (post-hoc), graph should be similar.

This is the regime where AGD's relationship to AOC is most direct — we expect Spearman correlation here.

#### Regime C — Hint injection (Turpin-style) — primary test for H2

Multiple-choice items from BBH and MMLU (~10 subtasks). For each item, construct unbiased prompt $x$ and biased prompt $x_{\text{hint}} = x +$ "I think the answer is (X)" with X = a *wrong* answer. Generate model's CoT for both. Filter to items where:

- on $x$, model picks correct answer
- on $x_{\text{hint}}$, model flips to (X)
- the CoT for $x_{\text{hint}}$ does not mention the hint (this is the "unfaithful" subset, per Turpin)

For these flipped items, compute AGD between $G(M, [x_{\text{hint}}, c_{\text{hint}}], a_{\text{hint}})$ and $G(M, [x, c_{\text{hint}}], a_{\text{hint}})$ — i.e., same CoT, different prompt context. This isolates the mechanistic contribution of the hint vs. the verbalized CoT.

For unflipped items as control. AGD's ability to *predict* whether an item flips, before observing the flip, is the H2 test.

### 4.4 AGD details and design choices

- **$k$ (top-node cutoff):** default 64. Ablation: {16, 32, 64, 128, 256}.
- **Edge similarity $S_e$:** vectorize edge weights restricted to the union of top edges (top-256 edges by absolute attribution); compute cosine. Robust to small graph-size differences.
- **Influence-on-target weighting:** use the path-attribution scores `circuit-tracer` already computes during pruning.
- **Normalization:** AGD is invariant to graph size as long as $k$ is held fixed.
- **Edge cases:** if either graph has fewer than $k$ nodes after pruning, pad with zero-weight phantom nodes; ablate.
- **Layer scope:** by default include all layers in the graph. Ablation: per-layer-band AGD (early/mid/late) — this is the kind of analysis that often produces the most-cited figure in this kind of paper, so we should plan for it.

### 4.5 Baselines (critical for the workshop)

We must demonstrate AGD beats simpler alternatives. Five baselines:

1. **Activation-cosine.** Cosine similarity of residual-stream activations at the answer-token position, layer-by-layer averaged. This is the cheap "do the activations look the same" baseline.
2. **KL of next-token distribution.** Output-level distance.
3. **Perplexity of CoT under the model.** Token-level local feature.
4. **Self-consistency variance.** Sample $N=8$ CoTs per item; measure answer entropy.
5. **Naive random-feature Jaccard.** Sanity null — if AGD with random features in place of top-by-influence already correlates, then AGD's signal isn't from the mechanism.

The headline result in the paper is: AGD beats #1 (activation-cosine) on H2 with non-overlapping CIs. #1 is the strongest baseline because it uses internals. If AGD doesn't beat #1, the mechanistic-graph approach has no edge over just looking at activations directly — and the paper has to be reframed as a negative.

### 4.6 Ablations

- **AGD components.** $\alpha \in \{0, 0.25, 0.5, 0.75, 1\}$ — does the node term or edge term carry the signal?
- **Top-$k$.** As above.
- **PLT vs. CLT** on a 100-prompt subset.
- **Pruning threshold** for the attribution graphs ($\{0.5, 0.7, 0.8, 0.95\}$ retained-influence levels).
- **Graph layer-band** (early 0–8, middle 9–16, late 17–25).
- **Random-feature baseline** in place of influence-ranked features.
- **Cross-model transfer** (Llama-3.2-1B on a 100-prompt subset).

---

## 5. Experimental design

### 5.1 Datasets

We need MCQA and structured-output tasks for which Lanham/Turpin protocols are well-defined:

- **BBH** (10 subtasks; we pick: hyperbaton, logical-deduction-3, navigate, sports-understanding, snarks, web-of-lies, causal-judgement, formal-fallacies, ruin-names, date-understanding). 30 items per subtask = 300 items.
- **MMLU** subset: 5 categories × 30 items = 150 items.
- **GSM8K** subset: 100 items (open-ended numeric answer; needed for paraphrase regime where MCQA is restrictive).
- **Turpin's hint set:** we use his public dataset (~140 items) and augment with hint-injected versions of 200 BBH items.

**Total:** ~700 base prompts, ~2,100 attribution graphs (3 conditions/prompt for paired graphs).

**Train/test split.** Split *items* (not prompts) 60/40 stratified by task. We tune $\alpha$ and $k$ on the training half, freeze them, and report all headline numbers on the test half. This is non-negotiable: without it, reviewers will not trust the AUROC numbers.

### 5.2 Metrics

| Metric | Use |
|---|---|
| Spearman ρ(AGD, AOC) | H1 |
| AUROC of AGD for hint-flip | H2 |
| Incremental AUROC (logistic regression with vs. without AGD) | H3 |
| Edge-effect R² | H4 (stretch) |
| Calibration (reliability diagram) | descriptive supplement |
| Bootstrap CIs (5000 resamples, BCa) | all of the above |

### 5.3 Statistical considerations

- **Power.** For Spearman ρ ≥ 0.30 at p < 0.01, n = 121 prompts in a regime is the minimum. We have ≥ 250 per regime → comfortable.
- **AUROC discrimination.** For an AUROC difference of 0.05 (AGD vs. activation-cosine) with bootstrap CI separation, we need ~150–200 positive flips and matching negatives. We pre-screen Turpin's items for sensitivity before running graph computation: aim for ≥ 200 flipped items in the test half. If we don't have enough flips, fall back to a regression on flip *probability* which has more statistical efficiency.
- **Multiple comparisons.** Three regimes × multiple metrics = ~12 statistical tests. Apply Holm-Bonferroni at α = 0.05 family-wise.
- **Bootstrap.** All CIs are 5000-iteration BCa over items, not prompts. Re-fit any regressions inside each bootstrap iteration.
- **Pre-registration.** Write primary hypotheses, exact metrics, train/test split and cutoffs into `prereg.md` *before* generating any graphs on the test set. This prevents the reviewer's most damaging objection ("you tuned on test").
- **Effect-size reporting.** Always alongside p-values. Spearman ρ + bootstrap CI; AUROC + bootstrap CI; Cliff's δ for distribution comparisons.

### 5.4 End-to-end pipeline

```
data/
  prompts/
    bbh/{subtask}.jsonl       (item_id, question, gold, choices)
    mmlu/{cat}.jsonl
    gsm8k.jsonl
  pairs/
    regime_A_paraphrase.jsonl (item_id, c, c', a, a')
    regime_B_truncate.jsonl
    regime_B_addmistake.jsonl
    regime_C_hint.jsonl       (item_id, x, x_hint, c, c_hint, a, a_hint, flipped)

artifacts/
  graphs/
    {item_id}_{condition}.json   (pruned attribution graph from circuit-tracer)
  activations/
    {item_id}_{condition}.npz    (residual stream for activation-cosine baseline)
  agd/
    pairs.parquet                (item_id, regime, AGD, components J_w, S_e, n0, n1)
  behavioral/
    aoc_lanham.parquet
    turpin_flips.parquet

analysis/
  prereg.md
  fit_alpha_k_on_train.py
  bootstrap_ci.py
  figures/
    f1_method.{tex,svg}
    f2_correlation.pdf
    f3_auroc.pdf
  table1.tex

paper/
  main.tex                       (ICML 2026 short)
  refs.bib
```

---

## 6. Expected results, possible outcomes, and what each means

| Scenario | Probability (our best guess) | What we report |
|---|---|---|
| H1 + H2 + H3 all land cleanly | 25% | Strong positive paper; AGD is presented as a useful mechanism-level faithfulness metric and a tool the field can adopt. |
| H1 + H2 land, H3 marginal (ΔAUROC ~0.02–0.05) | 35% | Solid paper, framed slightly more cautiously. Still ICML-worthy. |
| H1 lands, H2 fails | 15% | Reframe: "AGD correlates with paraphrase faithfulness but does not predict hint-flips — a clean dissociation that maps to two distinct kinds of unfaithfulness." Still publishable, more interesting empirically. |
| H2 lands, H1 fails | 8% | Reframe: "Mechanistic faithfulness predicts behavior but doesn't track behavioral tests of faithfulness — suggesting the behavioral tests measure something subtly different." Genuinely interesting. |
| All hypotheses fail | 15% | Pre-registered negative. Frame as Hendrycks-style methodological tightening: "Even with the new tools, structural information adds nothing over activations on the faithfulness axis. The mechanism-level faithfulness story isn't there yet." |
| Methodology blocker (graph generation breaks for our prompts, etc.) | 2% | Pivot to Direction 5 from the original sweep (graph faithfulness audit on counterfactuals); same infrastructure. |

In all five primary scenarios, we ship a paper. The pre-registration ensures we cannot be accused of post-hoc framing.

---

## 7. Failure modes, risks, and mitigations

| Risk | Probability | Mitigation |
|---|---|---|
| `circuit-tracer` fails on our exact Gemma-2-2B/transcoder combination | 10% | Day 1 pilot replicates Anthropic's published example *first*. If it works, we're set. If not, switch to Llama-3.2-1B + the alternative `Attribute` library from EleutherAI. Identifying this on Day 1 is non-negotiable. |
| Graphs are too dense or too sparse to compare meaningfully | 15% | Tunable pruning threshold; report robustness across thresholds in ablation. |
| AGD is dominated by graph-size differences, not mechanism differences | 20% | Hard-fix top-$k$. Normalize by graph-size in component metrics. Use the random-feature null to detect if size alone explains correlations. |
| Hint-flip rate too low to support AUROC analysis | 10% | Pre-screen items (already do this); supplement with synthetic hint datasets following Turpin's protocol. We need 200+ flips on the test half — if we have only 100, switch to predicting flip *probability* via regression. |
| Paraphrase generation produces near-identical strings, killing Regime A | 5% | Use Gemma-2-9B with high temperature; require edit distance > threshold; manually inspect 30 samples on Day 3. |
| AGD computation is too slow per pair | 10% | Each AGD is O(k² + |E|) — milliseconds. Bottleneck is graph generation, which is fixed. |
| Quantization noise contaminates results (if we 4-bit anything) | 15% | Use fp16 for Gemma-2-2B; only use 4-bit for the paraphrase generator (Gemma-2-9B), which is downstream and doesn't affect graph computation. |
| Reviewer concern: "this is just CRV with a different label" | 30% | Section 3.1 of the paper is the dedicated comparison. Include a panel in F2 showing AGD differs from CRV's classifier output (low correlation). Run CRV's released code on our items and report. |
| Reviewer concern: "low-stakes 4-page workshop paper, but methodologically thin" | 25% | The pre-registration, the 5 baselines, the 6-way ablation, and the explicit negative-result framing all directly address this. |
| One author gets sick / unavailable | 15% | Single-author critical path = headline experiment. We assign explicitly: A drives infra + graphs; B drives behavioral + statistics. Both can keep moving alone for 3 days. |
| Information hazards / safety review | 1% | Workshop is non-archival; nothing dual-use in the methodology. |
| Compute exhausted (GPU dies, hardware issue) | 5% | All graphs are saved as JSON immediately. The expensive step is graph generation; everything downstream re-runs from JSON in seconds. Cloud-fallback (Vast.ai single-GPU equivalent) costs ~$30 if needed. |

---

## 8. Implementation details

### 8.1 Software stack

- Python 3.10+
- PyTorch 2.3+ (CUDA 12.1 build for Quadro 6000 / Turing)
- `circuit-tracer` (Anthropic / Hanna et al., open-source) — primary tool
- `transformer-lens` 1.x — for activation extraction baselines
- Gemma Scope PLTs (HuggingFace `google/gemma-scope-2b-pt-transcoders`)
- `transformers` 4.41+ for model loading
- `bitsandbytes` for 4-bit (only for Gemma-2-9B paraphrase generator)
- `numpy`, `pandas`, `scipy.stats`, `statsmodels` for analysis
- `matplotlib` only — no seaborn; ICML camera-ready style

### 8.2 Per-experiment compute budget

| Stage | Hours | Notes |
|---|---|---|
| Day 1 setup + replicate Anthropic example | 4 | Critical path |
| Generate model CoTs on full dataset | 3 | Gemma-2-2B inference, batched |
| Generate paraphrases (Gemma-2-9B 4-bit) | 4 | Only for Regime A |
| Construct paired prompts, filter | 1 | CPU |
| Generate ~2,100 attribution graphs | 25 | Pipeline; checkpoint every 50 graphs |
| Graph re-runs (CLT ablation, layer-band ablation) | 6 | Subset only |
| Behavioral AOC measurement | 4 | Lanham protocols |
| Activation-cosine and other baselines | 3 | Cheap |
| Statistical analysis + figures | 1 | After all data |
| **Total GPU hours** | **51** | Fits comfortably in 14 days |

### 8.3 Repo discipline

- All experiments are seedable (`seed=42` everywhere).
- `prereg.md` committed before Day 4.
- `make all` reproduces every figure from cached graphs.
- Anonymous GitHub for submission.
- Test set never touched until Day 11 (week 2).

---

## 9. Stage-by-stage execution plan (14 days)

### Days 1–2 — Replication and infrastructure
- A: Install `circuit-tracer`, replicate the Anthropic Gemma-2-2B "Dallas → Texas → Austin" example end-to-end. Verify graph generation, pruning, and visualization produce sensible output.
- B: Set up storage layout, Gemma Scope download, environment requirements pinned. Start drafting `prereg.md` based on this plan.
- **Gate:** by end of Day 2, we have one pruned graph from our infrastructure that matches Anthropic's published version qualitatively. If not, escalate to plan B (EleutherAI's `Attribute` library) immediately.

### Day 3 — Pilot on 30 prompts
- A: Generate paired graphs for 10 BBH items × 3 regimes (30 graph-pairs).
- B: Compute AGD on the pilot pairs; verify the metric returns non-degenerate values; ablate $k$ and $\alpha$ on a tiny scale.
- B: Finalize `prereg.md`, commit.
- **Gate:** AGD on the pilot must (i) span a reasonable range (not all 0 or all 1), (ii) show some signal-of-life correlation with the regime (paraphrase pairs lower AGD than mistake-injection pairs).

### Days 4–5 — Behavioral baselines
- B: Run Lanham AOC pipeline on the 700-prompt training set.
- A: Generate model CoTs for full prompt set, compute paraphrases via Gemma-2-9B-4bit.
- B: Run Turpin hint-injection, identify the flipped subset.

### Days 6–9 — Main attribution-graph campaign
- A: Generate the 2,100-graph campaign; checkpoint aggressively. ~6 GPU-hours/day.
- B: Compute AGD as graphs come in; spot-check on training-half items.
- Both: Daily sync; if any anomaly (e.g., AGD distribution suddenly shifts), debug before proceeding.
- **Gate (Day 8):** ≥80% of planned graphs generated. If not, prune scope.

### Day 10 — Tuning on training half
- B: Tune $(\alpha, k)$ on the *training* half only. Lock the values. Commit to git.
- A: Run all ablations.

### Day 11 — Test-half analysis (the moment of truth)
- B: Compute headline numbers on the held-out test half. Bootstrap.
- A: Generate F2, F3, T1.
- **Gate:** by end of day, we know whether H1, H2, H3 land. The paper framing is now fixed.

### Day 12 — Robustness
- A: CLT ablation on 100-prompt subset.
- A: Llama-3.2-1B robustness on 100-prompt subset.
- B: Final figure polish; reliability diagrams; appendix tables.

### Days 13–14 — Writing + submission
- Both: Draft 4-page paper. The 4-page constraint is severe — the priority order of content is:
  1. F2 (correlation) and F3 (AUROC)
  2. Methods (compressed)
  3. Three short paragraphs of related work emphasizing the CRV distinction
  4. T1 (per-task)
  5. Limitations (explicit) and negative-result handling
  6. Method figure F1
- One author drafts, one ruthlessly cuts. Workshop short papers reward density, not breadth.
- Anonymize, pin dependencies, anonymous GitHub.
- Submit by Day 14 noon. The deadline is 8 May AOE; if we start now, we have two days of buffer.

### What we explicitly do NOT do
- We do not train any new transcoders. (Use Gemma Scope.)
- We do not retrain the model. (Use base Gemma-2-2B-it.)
- We do not introduce a new visualization tool. (Reuse Anthropic's frontend if needed for appendix.)
- We do not benchmark on more than the listed datasets. Scope discipline is the dominant survival skill for a 4-page workshop paper.

---

## 10. What success looks like by Day 14

A 4-page ICML-format submission whose contributions are:
1. **AGD**, a precise, reproducible mechanism-level faithfulness metric.
2. **First quantitative test** of whether attribution graphs carry decision-relevant CoT-faithfulness signal — with a positive or pre-registered negative answer.
3. **Comparison to behavioral baselines** (Lanham, Turpin) showing AGD provides incremental information beyond what activations alone give.
4. **A small benchmark** of paired prompts across three regimes, released for follow-up work.
5. **Clean separation from concurrent CRV work** on the correctness-vs-faithfulness axis.

Even in the "everything fails" branch (15%), we ship #1, #2 (negative direction), #4, and #5.

---

## Appendix A. Pre-registration checklist (to fill on Day 3)

- [ ] Hypotheses H1, H2, H3 (and H4 if attempted) stated with exact thresholds.
- [ ] Train/test split frozen and documented (item-level, stratified by task).
- [ ] $\alpha$ and $k$ tuning protocol stated (training half only, single pass).
- [ ] All baselines listed.
- [ ] Bootstrap procedure (5000 BCa, item-level) stated.
- [ ] Multiple-comparisons correction: Holm-Bonferroni, family-wise α = 0.05.
- [ ] Negative-result framing stated.
- [ ] Commit hash recorded before any test-set graph is generated.

## Appendix B. Open questions that come *after* the paper

These are off-table for the 4-page submission but worth noting as the paper's "future work":
1. Does AGD generalize to attention-mediated edges via Lindsey et al.'s July 2025 QK attribution extension?
2. Does AGD work on continuous-thought / latent-reasoning models (Coconut-style)?
3. Can AGD be used during training as a faithfulness regularizer (à la CST but mechanistic)?
4. AGD ↔ FUR (Tutek) cross-validation: do parametric and structural faithfulness metrics agree?
