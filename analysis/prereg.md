# Pre-Registration: AGD — Attribution-Graph Divergence for CoT Faithfulness
# ICML 2026 Mechanistic Interpretability Workshop

**Registered:** <!-- FILL: ISO date, e.g. 2026-04-29 -->  
**Git commit:** <!-- FILL: `git rev-parse HEAD` BEFORE running script 09 or 10 -->  
**Status:** ☐ Draft → ☐ Committed → ☐ Test-half data generated

---

## 1. Primary Hypotheses

### H1 — Faithfulness signal (Regime B — Mistake/Truncation)

**Statement.** The Spearman rank correlation between AGD and `1 − AOC_composite`,
computed on the held-out test set (Regime B pairs only), satisfies:

$$\rho_{\text{Spearman}}(\text{AGD},\ 1 - \text{AOC\_composite}) \geq 0.30$$
with $p < 0.01$ (two-sided, BCa bootstrap, 5000 resamples, item-level).

*Sign convention note (added 2026-05-02):* Lanham et al. (2023) define AOC such that
**high AOC = more faithful** (i.e. the model *is* sensitive to the CoT, so AOC is
*low* when the model is post-hoc / unfaithful). AGD is designed to be *high* when
behaviour is unfaithful. Therefore we correlate AGD with `1 − AOC` so that both
quantities point in the "unfaithfulness" direction and the pre-registered threshold
`ρ ≥ 0.30` corresponds to a positive correlation. All code in scripts 09 and 10
computes `1 − aoc_composite` before calling `spearman_with_ci`. The falsification
clause below uses `|ρ| < 0.15` which is sign-agnostic, but the test in script 10
enforces the directional threshold `ρ ≥ 0.30` without `abs()`.

**Falsified if:** $\rho < 0.15$ **and** the 95% BCa bootstrap CI for $\rho$ excludes 0.20
on both sides, evaluated on $\geq 100$ Regime B test pairs (one row per base item after
averaging AGD across pair variants).

**Exact metric:** `analysis/results_test.json → H1_spearman.rho`

---

### H2 — Predictive power for hint-induced flips (Regime C)

**Statement.** The AUROC of AGD for predicting unfaithful hint-flips
on the held-out test set satisfies:

$$\text{AUROC}_{\text{AGD}} \geq 0.65$$

**and** the 95% BCa bootstrap CI for $\text{AUROC}_{\text{AGD}}$ has its **lower bound
strictly above** the **upper bound** of the 95% CI for the activation-cosine baseline AUROC
(non-overlapping CIs — stronger than a point-estimate comparison).

**Falsified if:** $\text{AUROC}_{\text{AGD}} \leq 0.55$ **or** CI overlap with activation-cosine
baseline exceeds 50% of combined CI width.

**Exact metric:** `analysis/results_test.json → H2_auroc_agd.auc`, `H2_ci_non_overlapping`

---

### H3 — Incremental information beyond activations (Regime C)

**Statement.** A logistic regression with features
$[\text{AGD}, \text{activation-cosine}, \text{KL-next-token}, \text{CoT-perplexity}, \text{self-consistency-variance}]$
achieves AUROC at least $0.05$ above the same regression **without AGD**:

$$\Delta\text{AUROC} \geq 0.05$$

with bootstrap $p < 0.05$ (one-sided: $P(\Delta\text{AUROC} \leq 0 | H_0)$).

**Falsified if:** $\Delta\text{AUROC} < 0.02$ **or** 95% CI crosses 0.

**Exact metric:** `analysis/results_test.json → H3_incremental_auroc.delta_auc`

---

### H4 — Edge ablation reliability (stretch goal)

**Statement.** For each top-attribution edge in $G_0$, the predicted effect-size of
feature-ablation matches the measured effect on the target logit with $R^2 \geq 0.5$.

**Falsified if:** $R^2 < 0.3$ (reported as a known limitation of AGD's signal, not a failure
of the paper's primary contribution).

**Note:** H4 will be attempted only if H1–H3 complete with time remaining.

---

## 2. Multiple Comparisons

All primary hypothesis tests (H1 Spearman $p$-value, H2 AUROC CI test, H3 bootstrap $p$-value)
are corrected using **Holm-Bonferroni** with family-wise $\alpha = 0.05$ (3 tests).

---

## 3. Train / Test Split

- **Method:** Stratified by task, 60% train / 40% test.
- **Level:** Item-level (not prompt-level, not condition-level).
- **Seed:** 42 (fixed, reproducible).
- **Files:** `data/train_ids.txt`, `data/test_ids.txt`
- **Commit hash of split files:** <!-- FILL: `git log --oneline data/train_ids.txt | head -1` -->

The test set will NOT be examined (no graphs generated for test items, no
behavioral measures run on test items) until Day 11, after `best_hyperparams.json`
has been committed.

---

## 4. Hyperparameter Tuning Protocol

**Parameters:** $\alpha \in \{0, 0.25, 0.5, 0.75, 1.0\}$, $k \in \{16, 32, 64, 128, 256\}$.

**Criterion:** Maximize H2 AUROC on the **training half only** (single pass, no re-evaluation).

**Lock procedure:**
1. Run `scripts/09_tune_on_train.py`.
2. Inspect `analysis/best_hyperparams.json`.
3. Commit the file: `git add analysis/best_hyperparams.json && git commit`.
4. Only then run `scripts/10_test_half_analysis.py`.

**Fallback:** If H2 data is insufficient (< 100 flipped test items), tune on H1 Spearman
and document this deviation here.

---

## 5. Baselines

All baselines are computed and evaluated on the **same test items** as AGD.

| Baseline | Description | Comparison |
|----------|-------------|------------|
| Activation-cosine | Layer-averaged residual-stream cosine at answer-token position | Primary (H2 CI overlap test) |
| KL next-token | $\text{KL}(p_0 \| p_1)$ of last-position logit distribution | H3 regression feature |
| CoT-perplexity | Token-level perplexity of CoT under base model | H3 regression feature |
| Self-consistency variance | Answer entropy across $N=8$ sampled CoTs | H3 regression feature |
| Random-feature Jaccard | Jaccard over randomly sampled features (not top-by-influence) | Sanity null |

---

## 6. Bootstrap Procedure

- **Algorithm:** BCa (bias-corrected accelerated) bootstrap.
- **Resamples:** 5000.
- **Resampling unit:** Items (not prompts, not conditions).
- **For H3:** Logistic regression is re-fit inside each bootstrap iteration.
- **All reported CIs:** 95% two-sided, unless stated otherwise.

---

## 7. Negative Result Framing (pre-committed)

If H1, H2, and H3 **all fail**:

> *"Even with cross-layer transcoder attribution graphs, a principled
> graph-divergence metric does not improve over activation-distance baselines
> for predicting CoT unfaithfulness. We provide the first quantitative test of
> this claim and discuss why structural information may not be the missing piece
> in the CoT faithfulness measurement stack."*

This framing is committed before any test-set data is examined.

---

## 8. Scope Constraints (what we will NOT do)

- We will not generate attribution graphs with CLT transcoders for the primary analysis
  (only for the 100-item ablation in script 11).
- We will not add new datasets beyond BBH, MMLU, GSM8K, and Turpin.
- We will not train a classifier on AGD values (AGD is explicitly a threshold-free metric).
- We will not adjust thresholds in §1 after seeing test-half results.

---

## Checklist (to verify before running script 09)

- [ ] H1, H2, H3 thresholds documented above (exact numbers, not ranges).
- [ ] train_ids.txt and test_ids.txt committed to git.
- [ ] $\alpha$ and $k$ tuning protocol stated (training half only, single pass).
- [ ] All 5 baselines listed.
- [ ] Bootstrap procedure (5000 BCa, item-level) stated.
- [ ] Holm-Bonferroni, family-wise $\alpha = 0.05$ stated.
- [ ] Negative-result framing written.
- [ ] This file committed: `git add analysis/prereg.md && git commit -m "Pre-registration"`
- [ ] Git commit hash recorded in §§ above.
