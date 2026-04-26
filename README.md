# cot-mech-interp — AGD: Attribution-Graph Divergence for CoT Faithfulness

**ICML 2026 Mechanistic Interpretability Workshop** | 4-page short paper | Deadline: 8 May 2026 AOE

---

## What is AGD?

**Attribution-Graph Divergence (AGD)** is a mechanism-level faithfulness metric for chain-of-thought reasoning. Given a base prompt with CoT $c$ and a perturbed variant $c'$, AGD measures how much the model's *internal computation* (captured via pruned attribution graphs over Gemma Scope transcoder features) reorganizes:

$$\mathrm{AGD} = 1 - \alpha \cdot J_w(\mathcal{N}(G_0), \mathcal{N}(G_1)) - (1-\alpha) \cdot S_e(\mathcal{E}(G_0), \mathcal{E}(G_1))$$

where $J_w$ is the influence-weighted Jaccard over top-k feature nodes and $S_e$ is the edge-attribution cosine similarity. AGD ∈ [0, 1]; AGD ≈ 0 means the mechanism is intact; AGD → 1 means it has reorganized.

The central claim: **AGD predicts behavioral faithfulness signals (Lanham-AOC, Turpin-hint-flips) better than activation-distance baselines**, providing the missing causal-mechanistic counterpart to the existing behavioral CoT-faithfulness literature.

---

## Quick Start

```bash
# 1. Install dependencies
make setup

# 2. Download datasets (BBH, MMLU, GSM8K, Turpin)
make download && make split

# Day 1 gate — replicate Anthropic's circuit-tracing example
make replicate

# 3. Generate CoTs, paraphrases, and paired conditions
make cots
make paraphrase
make pairs

# Day 3 gate — pilot on 30 graph-pairs
make pilot

# 4. Full graph campaign (~25 GPU-hours)
make graphs
make activations

# 5. Compute AGD + baselines + behavioral measures
make agd
make baselines
make behavioral

# 6. Tune on training half ONLY (commit prereg.md first!)
make tune

# 7. Test-half analysis (the moment of truth)
make test

# 8. Ablations + figures
make ablations
make figures
```

> **Research integrity:** `analysis/prereg.md` must be committed and pushed before running `make tune` or `make test`. The train/test split is item-level, stratified by task, seeded at 42.

---

## Repository Layout

```
cot-mech-interp/
├── config.yaml                # Central configuration (model, paths, hyperparams)
├── Makefile                   # Reproducible pipeline
├── requirements.txt
│
├── src/                       # Core library
│   ├── config.py              # Config loader → frozen Config dataclass
│   ├── model_utils.py         # Model / transcoder loading
│   ├── graph_utils.py         # Attribution graph generation + I/O
│   ├── agd.py                 # AGD metric (J_w, S_e, compute_agd, batch_agd)
│   ├── baselines.py           # 5 baselines (activation-cosine, KL, PPL, SC, null)
│   ├── behavioral.py          # Lanham AOC + Turpin hint-flip protocols
│   └── stats.py               # Bootstrap CIs, AUROC, Spearman, Holm-Bonferroni
│
├── scripts/                   # Numbered pipeline scripts (run in order)
│   ├── 00_setup_and_replicate.py
│   ├── 01_generate_cots.py
│   ├── 02_generate_paraphrases.py
│   ├── 03_construct_pairs.py
│   ├── 04_generate_graphs.py  # Main graph campaign
│   ├── 05_extract_activations.py
│   ├── 06_compute_agd.py
│   ├── 07_compute_baselines.py
│   ├── 08_behavioral_measures.py
│   ├── 09_tune_on_train.py    # Lock alpha, k on training half
│   ├── 10_test_half_analysis.py
│   ├── 11_ablations.py
│   └── 12_generate_figures.py
│
├── data/
│   ├── download_datasets.py
│   ├── split.py
│   ├── prompts/               # BBH / MMLU / GSM8K / Turpin JSONL (gitignored)
│   └── pairs/                 # Regime A/B/C paired conditions (gitignored)
│
├── artifacts/                 # All generated (gitignored except parquets)
│   ├── graphs/                # {item_id}_{condition}.json  ← EXPENSIVE, back up!
│   ├── activations/           # {item_id}_{condition}.npz
│   ├── agd/pairs.parquet
│   └── behavioral/
│
├── analysis/
│   ├── prereg.md              # Pre-registration (commit BEFORE test-half analysis)
│   ├── best_hyperparams.json  # Frozen alpha + k from training half
│   ├── results_test.json      # H1/H2/H3 results on test half
│   ├── ablations.json
│   └── figures/               # F1 (svg), F2 (pdf), F3 (pdf)
│
├── tests/
│   ├── test_agd.py
│   ├── test_stats.py
│   └── test_graph_utils.py
│
└── paper/
    ├── main.tex               # 4-page ICML 2026 short paper
    └── refs.bib
```

---

## Three Experimental Regimes

| Regime | Description | Expected AGD |
|--------|-------------|--------------|
| **A — Paraphrase** | Semantic-preserving paraphrase of CoT (same answer) | Low (mechanism intact) |
| **B — Mistake/Truncation** | Lanham-style truncation / mistake injection | Higher for faithful models |
| **C — Hint injection** | Turpin-style biased prompt flipping the answer | High for unfaithful items |

## Hypotheses (pre-registered)

| | Hypothesis | Threshold | Test |
|--|------------|-----------|------|
| **H1** | AGD correlates with behavioral AOC | Spearman ρ ≥ 0.30, p < 0.01 | Regime B |
| **H2** | AGD predicts hint-flip with AUROC ≥ 0.65 | 95% CI lower bound > activation-cosine upper bound | Regime C |
| **H3** | AGD adds incremental AUROC ≥ 0.05 over baselines | Bootstrap p < 0.05 | Logistic regression |
| **H4** | Edge ablations have R² ≥ 0.5 | Stretch goal | Appendix |

---

## Hardware

- 1 × Quadro RTX 6000 (24 GB)
- Gemma-2-2B-it in fp16 (~5 GB) + Gemma Scope PLT transcoders
- Paraphrase generator: Gemma-2-9B-it in 4-bit (~8 GB)
- Estimated GPU budget: ~51 hours across 14 days

---

## Citation

If you use AGD or the paired-prompt benchmark in follow-up work, please cite: *(BibTeX forthcoming after submission.)*

---

## License

MIT. The research methodology is open and reproducible.
