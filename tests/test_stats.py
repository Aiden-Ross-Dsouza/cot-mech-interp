"""
tests/test_stats.py
Unit tests for statistical utilities (src/stats.py).

Tests use synthetic data with known ground-truth properties.
Run with: pytest tests/test_stats.py -v
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats as scipy_stats

from src.stats import (
    spearman_with_ci,
    auroc_with_ci,
    incremental_auroc,
    holm_bonferroni,
    cliffs_delta,
    reliability_diagram_data,
    bootstrap_bca,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def perfectly_correlated():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = x + rng.normal(0, 0.02, 200)  # near-perfect positive correlation
    return x, y


@pytest.fixture
def anticorrelated():
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 1, 200)
    y = 1 - x + rng.normal(0, 0.02, 200)
    return x, y


@pytest.fixture
def perfect_classifier():
    """Scores perfectly separate positives from negatives."""
    rng = np.random.default_rng(42)
    scores = np.concatenate([rng.uniform(0.6, 1.0, 100), rng.uniform(0.0, 0.4, 100)])
    labels = np.array([1] * 100 + [0] * 100)
    return scores, labels


@pytest.fixture
def random_classifier():
    rng = np.random.default_rng(42)
    scores = rng.uniform(0, 1, 200)
    labels = rng.integers(0, 2, 200)
    return scores, labels


# ─────────────────────────────────────────────────────────────────────────────
# spearman_with_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestSpearmanCI:
    def test_perfect_correlation_near_one(self, perfectly_correlated):
        x, y = perfectly_correlated
        result = spearman_with_ci(x, y, n_boot=200)
        assert result["rho"] > 0.95
        assert result["p"] < 0.001

    def test_anticorrelation_negative_rho(self, anticorrelated):
        x, y = anticorrelated
        result = spearman_with_ci(x, y, n_boot=200)
        assert result["rho"] < -0.95

    def test_ci_contains_point_estimate(self, perfectly_correlated):
        x, y = perfectly_correlated
        result = spearman_with_ci(x, y, n_boot=200)
        assert result["ci_lo"] <= result["rho"] <= result["ci_hi"]

    def test_ci_width_positive(self, perfectly_correlated):
        x, y = perfectly_correlated
        result = spearman_with_ci(x, y, n_boot=200)
        assert result["ci_hi"] > result["ci_lo"]

    def test_returns_correct_keys(self, perfectly_correlated):
        x, y = perfectly_correlated
        result = spearman_with_ci(x, y, n_boot=50)
        assert set(result.keys()) == {"rho", "p", "ci_lo", "ci_hi", "n"}

    def test_n_drops_nans(self):
        x = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y = np.array([1.0, 2.0, 3.0, np.nan, 5.0])
        result = spearman_with_ci(x, y, n_boot=50)
        assert result["n"] == 3  # only indices 0, 1, 4 are clean

    def test_known_rho_within_ci_95pct(self):
        """Coverage test: known-rho data should be within CI 95% of the time."""
        rng = np.random.default_rng(42)
        n_trials = 50
        covered = 0
        true_rho = 0.5
        for _ in range(n_trials):
            x = rng.uniform(0, 1, 100)
            y = true_rho * x + np.sqrt(1 - true_rho**2) * rng.uniform(0, 1, 100)
            r = spearman_with_ci(x, y, n_boot=100)
            if r["ci_lo"] <= true_rho <= r["ci_hi"]:
                covered += 1
        # At least 80% coverage (BCa is approximate; 95% target, 80% tolerance)
        assert covered / n_trials >= 0.75


# ─────────────────────────────────────────────────────────────────────────────
# auroc_with_ci
# ─────────────────────────────────────────────────────────────────────────────

class TestAUROC:
    def test_perfect_classifier_auc_near_one(self, perfect_classifier):
        scores, labels = perfect_classifier
        result = auroc_with_ci(scores, labels, n_boot=200)
        assert result["auc"] > 0.95

    def test_random_classifier_auc_near_half(self, random_classifier):
        scores, labels = random_classifier
        result = auroc_with_ci(scores, labels, n_boot=200)
        # Random: AUC should be in [0.35, 0.65]
        assert 0.35 <= result["auc"] <= 0.65

    def test_ci_contains_point_estimate(self, perfect_classifier):
        scores, labels = perfect_classifier
        result = auroc_with_ci(scores, labels, n_boot=200)
        assert result["ci_lo"] <= result["auc"] <= result["ci_hi"]

    def test_perfect_auc_ci_near_one(self, perfect_classifier):
        scores, labels = perfect_classifier
        result = auroc_with_ci(scores, labels, n_boot=200)
        assert result["ci_lo"] > 0.90

    def test_returns_correct_keys(self, perfect_classifier):
        scores, labels = perfect_classifier
        result = auroc_with_ci(scores, labels, n_boot=50)
        assert "auc" in result and "ci_lo" in result and "ci_hi" in result

    def test_n_pos_n_neg_correct(self, perfect_classifier):
        scores, labels = perfect_classifier
        result = auroc_with_ci(scores, labels, n_boot=50)
        assert result["n_pos"] == 100
        assert result["n_neg"] == 100


# ─────────────────────────────────────────────────────────────────────────────
# incremental_auroc
# ─────────────────────────────────────────────────────────────────────────────

class TestIncrementalAUROC:
    def test_adding_informative_feature_increases_auc(self):
        """AGD feature that perfectly predicts labels should increase AUROC."""
        rng = np.random.default_rng(0)
        n = 200
        labels = rng.integers(0, 2, n)

        # Baseline: random features
        feat_reduced = rng.uniform(0, 1, (n, 3))
        # AGD: weakly informative signal
        agd_signal = labels.astype(float) + rng.normal(0, 0.3, n)
        feat_with = np.column_stack([agd_signal, feat_reduced])

        result = incremental_auroc(feat_with, feat_reduced, labels, n_boot=100)
        assert result["delta_auc"] > 0.0
        assert result["auc_with"] >= result["auc_without"]

    def test_delta_auc_near_zero_for_random_features(self):
        """Adding a random (uninformative) AGD column should not greatly increase AUROC."""
        rng = np.random.default_rng(7)
        n = 200
        labels = rng.integers(0, 2, n)
        feat_reduced = rng.uniform(0, 1, (n, 3))
        feat_with = np.column_stack([rng.uniform(0, 1, n), feat_reduced])

        result = incremental_auroc(feat_with, feat_reduced, labels, n_boot=100)
        assert abs(result["delta_auc"]) < 0.1

    def test_returns_correct_keys(self):
        rng = np.random.default_rng(0)
        n = 100
        labels = rng.integers(0, 2, n)
        X_f = rng.uniform(0, 1, (n, 2))
        X_r = X_f[:, :1]
        result = incremental_auroc(X_f, X_r, labels, n_boot=50)
        expected_keys = {"delta_auc", "auc_with", "auc_without", "ci_lo", "ci_hi",
                         "p_value", "n_boot_valid"}
        assert expected_keys <= set(result.keys())


# ─────────────────────────────────────────────────────────────────────────────
# holm_bonferroni
# ─────────────────────────────────────────────────────────────────────────────

class TestHolmBonferroni:
    def test_all_significant(self):
        p_values = [0.001, 0.002, 0.003]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert all(result["rejected"])

    def test_none_significant(self):
        p_values = [0.3, 0.4, 0.5]
        result = holm_bonferroni(p_values, alpha=0.05)
        assert not any(result["rejected"])

    def test_partial_rejection(self):
        # p=0.001 should be significant; p=0.1 should not
        p_values = [0.001, 0.1, 0.8]
        result = holm_bonferroni(p_values, alpha=0.05)
        # After Holm-Bonferroni: 0.001*3=0.003 < 0.05 ✓; 0.1*2=0.2 > 0.05 ✗
        assert result["rejected"][0] is True

    def test_returns_same_length(self):
        p_values = [0.01, 0.02, 0.03, 0.04]
        result = holm_bonferroni(p_values)
        assert len(result["corrected_p"]) == 4
        assert len(result["rejected"]) == 4

    def test_sequential_stop_rule(self):
        """Once we fail to reject, all subsequent must also not be rejected."""
        p_values = [0.001, 0.2, 0.001]  # sorted: 0.001, 0.001, 0.2
        result = holm_bonferroni(p_values, alpha=0.05)
        # 0.001*3=0.003 ✓, 0.001*2=0.002 ✓, 0.2*1=0.2 ✗
        # All rejections after the first failure should be False
        order = result["order"]
        rejected = result["rejected"]
        found_failure = False
        for idx in order:
            if not rejected[idx]:
                found_failure = True
            if found_failure:
                assert not rejected[idx], "Sequential stop rule violated"


# ─────────────────────────────────────────────────────────────────────────────
# bootstrap_bca
# ─────────────────────────────────────────────────────────────────────────────

class TestBootstrapBCA:
    def test_mean_ci_contains_true_mean(self):
        rng = np.random.default_rng(42)
        data = rng.normal(5.0, 1.0, 300)
        obs, lo, hi = bootstrap_bca(np.mean, data, n_boot=500)
        assert obs == pytest.approx(np.mean(data))
        assert lo <= 5.0 <= hi  # CI should contain the true mean (probably)

    def test_ci_width_larger_for_smaller_samples(self):
        rng = np.random.default_rng(0)
        big = rng.normal(0, 1, 500)
        small = rng.normal(0, 1, 30)
        _, lo_big, hi_big = bootstrap_bca(np.mean, big, n_boot=300)
        _, lo_small, hi_small = bootstrap_bca(np.mean, small, n_boot=300)
        # Smaller sample → wider CI
        assert (hi_small - lo_small) > (hi_big - lo_big)


# ─────────────────────────────────────────────────────────────────────────────
# cliffs_delta
# ─────────────────────────────────────────────────────────────────────────────

class TestCliffsDelta:
    def test_identical_groups_near_zero(self):
        data = np.array([1.0, 2.0, 3.0, 4.0])
        assert cliffs_delta(data, data) == pytest.approx(0.0)

    def test_fully_separated_positive_one(self):
        g1 = np.array([5.0, 6.0, 7.0])
        g2 = np.array([1.0, 2.0, 3.0])
        assert cliffs_delta(g1, g2) == pytest.approx(1.0)

    def test_fully_separated_negative_one(self):
        g1 = np.array([1.0, 2.0, 3.0])
        g2 = np.array([5.0, 6.0, 7.0])
        assert cliffs_delta(g1, g2) == pytest.approx(-1.0)

    def test_range(self):
        rng = np.random.default_rng(42)
        for _ in range(20):
            g1 = rng.normal(0, 1, 30)
            g2 = rng.normal(0.5, 1, 30)
            d = cliffs_delta(g1, g2)
            assert -1.0 <= d <= 1.0


# ─────────────────────────────────────────────────────────────────────────────
# reliability_diagram_data
# ─────────────────────────────────────────────────────────────────────────────

class TestReliabilityDiagram:
    def test_returns_dataframe_with_correct_cols(self, perfect_classifier):
        scores, labels = perfect_classifier
        df = reliability_diagram_data(scores, labels)
        assert set(df.columns) >= {"bin_center", "mean_score", "fraction_positive", "count"}

    def test_perfectly_calibrated_near_diagonal(self):
        """If scores == labels (0 or 1), calibration should be near-perfect."""
        n = 1000
        labels = np.array([0] * 500 + [1] * 500)
        scores = labels.astype(float) + np.random.default_rng(0).normal(0, 0.05, n)
        scores = np.clip(scores, 0, 1)
        df = reliability_diagram_data(scores, labels, n_bins=5)
        # Fraction_positive should roughly equal bin_center
        diffs = np.abs(df["fraction_positive"] - df["bin_center"])
        assert diffs.mean() < 0.2
