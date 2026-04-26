.PHONY: all setup pilot cots paraphrase pairs graphs activations agd \
        baselines behavioral tune analysis ablations figures test clean

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
PYTHON     := python
CONFIG     := config.yaml
SCRIPTS    := scripts
DATA       := data
ARTIFACTS  := artifacts
ANALYSIS   := analysis

# ─────────────────────────────────────────────────────────────────────────────
# Default target: reproduce all figures from cached graphs (no model needed)
# ─────────────────────────────────────────────────────────────────────────────
all: agd baselines behavioral figures
	@echo "✓ All figures and tables reproduced from cached artifacts."

# ─────────────────────────────────────────────────────────────────────────────
# Environment setup
# ─────────────────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	mkdir -p $(ARTIFACTS)/graphs $(ARTIFACTS)/activations \
	         $(ARTIFACTS)/agd $(ARTIFACTS)/behavioral \
	         $(ANALYSIS)/figures $(DATA)/prompts $(DATA)/pairs
	@echo "✓ Dependencies installed and directory tree created."

# Day 1 gate: replication of Anthropic's example
replicate:
	$(PYTHON) $(SCRIPTS)/00_setup_and_replicate.py --config $(CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
download:
	$(PYTHON) $(DATA)/download_datasets.py --config $(CONFIG)

split:
	$(PYTHON) $(DATA)/split.py --config $(CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline stages
# ─────────────────────────────────────────────────────────────────────────────
cots:
	$(PYTHON) $(SCRIPTS)/01_generate_cots.py --config $(CONFIG)

paraphrase:
	$(PYTHON) $(SCRIPTS)/02_generate_paraphrases.py --config $(CONFIG)

pairs:
	$(PYTHON) $(SCRIPTS)/03_construct_pairs.py --config $(CONFIG)

pilot:
	$(PYTHON) $(SCRIPTS)/04_generate_graphs.py --config $(CONFIG) --pilot
	$(PYTHON) $(SCRIPTS)/06_compute_agd.py    --config $(CONFIG) --pilot
	@echo "✓ Pilot complete. Check artifacts/agd/ for range of AGD values."

graphs:
	$(PYTHON) $(SCRIPTS)/04_generate_graphs.py --config $(CONFIG)

activations:
	$(PYTHON) $(SCRIPTS)/05_extract_activations.py --config $(CONFIG)

agd:
	$(PYTHON) $(SCRIPTS)/06_compute_agd.py --config $(CONFIG)

baselines:
	$(PYTHON) $(SCRIPTS)/07_compute_baselines.py --config $(CONFIG)

behavioral:
	$(PYTHON) $(SCRIPTS)/08_behavioral_measures.py --config $(CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Analysis  (ORDER MATTERS — tune before test)
# ─────────────────────────────────────────────────────────────────────────────
tune:
	@echo ">>> Tuning on TRAINING half only. Do NOT run 'make test' until"
	@echo ">>> prereg.md has been committed to git."
	$(PYTHON) $(SCRIPTS)/09_tune_on_train.py --config $(CONFIG)

test:
	$(PYTHON) $(SCRIPTS)/10_test_half_analysis.py --config $(CONFIG)

ablations:
	$(PYTHON) $(SCRIPTS)/11_ablations.py --config $(CONFIG)

# ─────────────────────────────────────────────────────────────────────────────
# Figures + tables
# ─────────────────────────────────────────────────────────────────────────────
figures:
	$(PYTHON) $(SCRIPTS)/12_generate_figures.py --config $(CONFIG)
	@echo "✓ Figures written to $(ANALYSIS)/figures/"

# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────
test-unit:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ --cov=src --cov-report=term-missing

# ─────────────────────────────────────────────────────────────────────────────
# Cleanup (NEVER removes graphs — those are expensive)
# ─────────────────────────────────────────────────────────────────────────────
clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	rm -f $(ARTIFACTS)/agd/pairs.parquet
	rm -f $(ARTIFACTS)/baselines.parquet
	rm -f $(ARTIFACTS)/behavioral/*.parquet
	@echo "✓ Cleaned derived artifacts (graphs preserved)."

clean-all: clean
	@echo "WARNING: This will remove ALL artifacts including graphs."
	@read -p "Are you sure? [y/N] " ans && [ $${ans:-N} = y ]
	rm -rf $(ARTIFACTS)/graphs/ $(ARTIFACTS)/activations/
	@echo "✓ Full clean complete."

# ─────────────────────────────────────────────────────────────────────────────
# Help
# ─────────────────────────────────────────────────────────────────────────────
help:
	@echo "AGD Research Pipeline — Makefile targets:"
	@echo ""
	@echo "  make setup        Install deps + create directory tree"
	@echo "  make replicate    Day-1 gate: replicate Anthropic's example"
	@echo "  make download     Download BBH / MMLU / GSM8K / Turpin datasets"
	@echo "  make split        Create 60/40 train/test item split"
	@echo "  make cots         Generate model CoTs for all prompts"
	@echo "  make paraphrase   Generate paraphrases (Regime A)"
	@echo "  make pairs        Construct all paired conditions (A/B/C)"
	@echo "  make pilot        Run 30-graph pilot (Day 3 gate)"
	@echo "  make graphs       Generate full ~2100-graph campaign"
	@echo "  make activations  Extract residual-stream activations (baselines)"
	@echo "  make agd          Compute AGD for all pairs → parquet"
	@echo "  make baselines    Compute all 5 baselines → parquet"
	@echo "  make behavioral   Lanham AOC + Turpin flip labels → parquet"
	@echo "  make tune         Tune alpha/k on training half ONLY"
	@echo "  make test         Compute headline metrics on test half"
	@echo "  make ablations    Run all 7 ablations"
	@echo "  make figures      Generate F1/F2/F3 + T1"
	@echo "  make all          Reproduce figures from cached artifacts"
	@echo "  make test-unit    Run pytest unit tests"
	@echo "  make clean        Remove derived artifacts (keep graphs)"
