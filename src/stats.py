"""
src/stats.py
Statistical analysis utilities for the AGD paper.

All CIs use BCa (bias-corrected accelerated) bootstrap with n_boot=5000.
All p-values are reported alongside effect sizes.
Multiple-comparisons correction: Holm-Bonferroni.

Public API:
  spearman_with_ci     → (rho, p, ci_lo, ci_hi)     [H1]
  auroc_with_ci        → (auc, ci_lo, ci_hi)          [H2]
  incremental_auroc    → (delta, ci_lo, ci_hi, p)     [H3]
  holm_bonferroni      → corrected p-values
  bootstrap_bca        → generic BCa bootstrap
  reliability_diagram  → calibration data for plotting
"""
from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Generic BCa Bootstrap
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_bca(
    statistic: Callable[[np.ndarray], float],
    data: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """BCa (bias-corrected accelerated) bootstrap confidence interval.

    Parameters
    ----------
    statistic:
        Function that takes a 1-D array and returns a scalar.
    data:
        1-D numpy array of observations.
    n_boot:
        Number of bootstrap samples.
    alpha:
        Significance level (two-sided CI: [alpha/2, 1-alpha/2]).
    seed:
        Random seed.

    Returns
    -------
    (observed_statistic, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    observed = statistic(data)

    # Bootstrap distribution
    boot_stats = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias-correction factor z0
    z0 = scipy_stats.norm.ppf(np.mean(boot_stats < observed) + 1e-12)

    # Acceleration factor a (jackknife)
    jack_stats = np.array([
        statistic(np.delete(data, i))
        for i in range(n)
    ])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    # Adjusted percentiles
    z_alpha_lo = scipy_stats.norm.ppf(alpha / 2)
    z_alpha_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def adj_pct(z_alpha: float) -> float:
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo_pct = adj_pct(z_alpha_lo)
    ci_hi_pct = adj_pct(z_alpha_hi)

    ci_lo = float(np.percentile(boot_stats, ci_lo_pct))
    ci_hi = float(np.percentile(boot_stats, ci_hi_pct))

    return observed, ci_lo, ci_hi


def bootstrap_bca_2sample(
    statistic: Callable[[np.ndarray, np.ndarray], float],
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """BCa bootstrap for a two-sample statistic.

    Parameters
    ----------
    statistic:
        Function(x, y) → scalar.

    Returns
    -------
    (observed, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    n_x, n_y = len(x), len(y)
    observed = statistic(x, y)

    boot_stats = np.array([
        statistic(
            rng.choice(x, size=n_x, replace=True),
            rng.choice(y, size=n_y, replace=True),
        )
        for _ in range(n_boot)
    ])

    # R2 fix: use BCa (bias-corrected accelerated) bootstrap, not plain percentile.
    # Matches the pre-registration commitment (prereg.md §6).
    observed = statistic(x, y)
    z0 = scipy_stats.norm.ppf(np.mean(boot_stats < observed) + 1e-12)
    # Jackknife acceleration over the pooled paired re-evaluation
    n_paired = min(n_x, n_y)
    jack_stats = np.array([
        statistic(np.delete(x[:n_paired], i), np.delete(y[:n_paired], i))
        for i in range(n_paired)
    ])
    jack_mean = np.mean(jack_stats)
    num = np.sum((jack_mean - jack_stats) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_stats) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo = float(np.percentile(boot_stats, _adj(z_lo)))
    ci_hi = float(np.percentile(boot_stats, _adj(z_hi)))
    return observed, ci_lo, ci_hi


# ─────────────────────────────────────────────────────────────────────────────
# H1 — Spearman correlation with BCa CI
# ─────────────────────────────────────────────────────────────────────────────

def spearman_with_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Spearman ρ between x and y with BCa bootstrap CI.

    Parameters
    ----------
    x, y:
        Arrays of equal length (NaN rows are dropped pairwise).

    Returns
    -------
    dict with keys: rho, p, ci_lo, ci_hi, n
    """
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = x[mask]
    y_clean = y[mask]
    n = len(x_clean)

    rho, p = scipy_stats.spearmanr(x_clean, y_clean)

    # Bootstrap CI
    rng = np.random.default_rng(seed)
    boot_rhos = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        xs, ys = x_clean[idx], y_clean[idx]
        if len(np.unique(xs)) < 2 or len(np.unique(ys)) < 2:
            continue
        r, _ = scipy_stats.spearmanr(xs, ys)
        if not np.isnan(r):
            boot_rhos.append(r)
    boot_rhos = np.array(boot_rhos)

    # R2 fix: BCa bootstrap (bias-corrected accelerated), matching prereg.md §6.
    # Plain percentile produces CIs ~2x too narrow at n<200 for Spearman.
    z0 = scipy_stats.norm.ppf(np.mean(boot_rhos < rho) + 1e-12)
    jack_rhos = np.array([
        scipy_stats.spearmanr(np.delete(x_clean, i), np.delete(y_clean, i))[0]
        for i in range(n)
    ])
    jack_mean = np.mean(jack_rhos)
    num = np.sum((jack_mean - jack_rhos) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_rhos) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo = float(np.percentile(boot_rhos, _adj(z_lo)))
    ci_hi = float(np.percentile(boot_rhos, _adj(z_hi)))

    return {
        "rho": float(rho),
        "p": float(p),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n": n,
    }


# ─────────────────────────────────────────────────────────────────────────────
# H2 — AUROC with BCa CI
# ─────────────────────────────────────────────────────────────────────────────

def auroc_with_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """AUROC for binary classification with BCa bootstrap CI.

    Parameters
    ----------
    scores:
        Predicted scores (higher = more likely positive).
    labels:
        Binary labels (0/1).

    Returns
    -------
    dict with keys: auc, ci_lo, ci_hi, n_pos, n_neg
    """
    mask = ~np.isnan(scores)
    scores_clean = scores[mask]
    labels_clean = labels[mask]

    auc = roc_auc_score(labels_clean, scores_clean)

    rng = np.random.default_rng(seed)
    n = len(scores_clean)
    boot_aucs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        try:
            b = roc_auc_score(labels_clean[idx], scores_clean[idx])
            boot_aucs.append(b)
        except ValueError:
            # Only one class in bootstrap sample → skip
            pass

    boot_aucs = np.array(boot_aucs)

    # R2 fix: BCa bootstrap, matching prereg.md §6.
    z0 = scipy_stats.norm.ppf(np.mean(boot_aucs < auc) + 1e-12)
    jack_aucs = np.array([
        roc_auc_score(np.delete(labels_clean, i), np.delete(scores_clean, i))
        for i in range(len(scores_clean))
    ])
    jack_mean = np.mean(jack_aucs)
    num = np.sum((jack_mean - jack_aucs) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_aucs) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo = float(np.percentile(boot_aucs, _adj(z_lo)))
    ci_hi = float(np.percentile(boot_aucs, _adj(z_hi)))

    return {
        "auc": float(auc),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "n_pos": int(labels_clean.sum()),
        "n_neg": int((1 - labels_clean).sum()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# H3 — Incremental AUROC (logistic regression with vs. without AGD)
# ─────────────────────────────────────────────────────────────────────────────

def incremental_auroc(
    features_with_agd: np.ndarray,    # shape [n, p]
    features_without_agd: np.ndarray, # shape [n, p-1]
    labels: np.ndarray,                # shape [n]
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute ΔAUROC = AUROC(with AGD) - AUROC(without AGD).

    B1 fix: The point estimate (delta_auc) is now the **bootstrap median** of the
    honest out-of-sample ΔAUROC (each iteration: resample → fit on 80% → eval on 20%).
    The previously used in-sample delta was inflated by ~0.02-0.05 for small n,
    which could push the estimate past the H3 threshold of 0.05 spuriously.

    The in-sample values are retained as `auc_with_insample` and
    `auc_without_insample` for diagnostic / sanity-check purposes.

    CI uses BCa bootstrap, matching prereg.md §6.

    Returns
    -------
    dict with keys: delta_auc, auc_with_insample, auc_without_insample,
                    ci_lo, ci_hi, p_value, n_boot_valid
    """
    def _bootstrap_delta_auroc(feat_full, feat_reduced, y, rng_seed):
        rng = np.random.default_rng(rng_seed)
        n = len(y)
        idx = rng.choice(n, size=n, replace=True)
        X_full = feat_full[idx]
        X_red = feat_reduced[idx]
        y_boot = y[idx]

        # Need at least 2 classes
        if len(np.unique(y_boot)) < 2:
            return float("nan")

        # Hold out 20% for evaluation
        n_train = int(0.8 * n)
        X_full_tr, X_full_te = X_full[:n_train], X_full[n_train:]
        X_red_tr, X_red_te = X_red[:n_train], X_red[n_train:]
        y_tr, y_te = y_boot[:n_train], y_boot[n_train:]

        if len(np.unique(y_te)) < 2:
            return float("nan")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lr_full = LogisticRegression(max_iter=500, random_state=42)
            lr_full.fit(X_full_tr, y_tr)
            lr_red = LogisticRegression(max_iter=500, random_state=42)
            lr_red.fit(X_red_tr, y_tr)

        auc_full = roc_auc_score(y_te, lr_full.predict_proba(X_full_te)[:, 1])
        auc_red = roc_auc_score(y_te, lr_red.predict_proba(X_red_te)[:, 1])
        return auc_full - auc_red

    # In-sample values retained for diagnostics only — NOT used as the headline estimate.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_full_obs = LogisticRegression(max_iter=500, random_state=42)
        lr_full_obs.fit(features_with_agd, labels)
        lr_red_obs = LogisticRegression(max_iter=500, random_state=42)
        lr_red_obs.fit(features_without_agd, labels)

    auc_with_insample = roc_auc_score(labels, lr_full_obs.predict_proba(features_with_agd)[:, 1])
    auc_without_insample = roc_auc_score(labels, lr_red_obs.predict_proba(features_without_agd)[:, 1])

    # Bootstrap
    boot_deltas = [
        _bootstrap_delta_auroc(features_with_agd, features_without_agd, labels, seed + i)
        for i in range(n_boot)
    ]
    boot_deltas = np.array([d for d in boot_deltas if not np.isnan(d)])

    # B1 fix: use bootstrap median as the honest, out-of-sample point estimate.
    # This avoids the ~0.02-0.05 in-sample inflation that could falsely pass H3.
    delta_obs = float(np.median(boot_deltas))

    p_value = float(np.mean(boot_deltas <= 0))  # one-sided: P(delta ≤ 0 | H0)

    # BCa CI — matching prereg.md §6 (R2 fix carried forward to H3)
    z0 = scipy_stats.norm.ppf(np.mean(boot_deltas < delta_obs) + 1e-12)
    # Jackknife acceleration: re-run one-sample bootstrap without each observation
    jack_deltas = np.array([
        _bootstrap_delta_auroc(
            np.delete(features_with_agd, i, axis=0),
            np.delete(features_without_agd, i, axis=0),
            np.delete(labels, i),
            rng_seed=seed,
        )
        for i in range(len(labels))
    ])
    jack_deltas = jack_deltas[~np.isnan(jack_deltas)]
    jack_mean = np.mean(jack_deltas)
    num = np.sum((jack_mean - jack_deltas) ** 3)
    denom = 6 * (np.sum((jack_mean - jack_deltas) ** 2) ** 1.5)
    a = num / denom if denom != 0 else 0.0

    z_lo = scipy_stats.norm.ppf(alpha / 2)
    z_hi = scipy_stats.norm.ppf(1 - alpha / 2)

    def _adj(z_alpha):
        z_adj = z0 + (z0 + z_alpha) / (1 - a * (z0 + z_alpha))
        return float(scipy_stats.norm.cdf(z_adj) * 100)

    ci_lo = float(np.percentile(boot_deltas, _adj(z_lo)))
    ci_hi = float(np.percentile(boot_deltas, _adj(z_hi)))

    return {
        "delta_auc": delta_obs,                       # B1 fix: bootstrap median (honest)
        "auc_with_insample": float(auc_with_insample),   # diagnostic
        "auc_without_insample": float(auc_without_insample),  # diagnostic
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "n_boot_valid": len(boot_deltas),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Holm-Bonferroni correction
# ─────────────────────────────────────────────────────────────────────────────

def holm_bonferroni(
    p_values: List[float],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Holm-Bonferroni procedure for family-wise error rate control.

    Parameters
    ----------
    p_values:
        List of uncorrected p-values (one per hypothesis).
    alpha:
        Family-wise alpha level (default 0.05).

    Returns
    -------
    dict with:
        corrected_p:   List of corrected p-values (same order as input).
        rejected:      List of bool (True = reject H0 at family-wise alpha).
        order:         Original indices sorted by p-value (ascending).
    """
    n = len(p_values)
    p_arr = np.array(p_values, dtype=float)
    order = np.argsort(p_arr)
    sorted_p = p_arr[order]

    corrected = np.zeros(n)
    reject = np.zeros(n, dtype=bool)

    for k in range(n):
        adjusted = sorted_p[k] * (n - k)
        corrected[order[k]] = min(adjusted, 1.0)
        reject[order[k]] = adjusted <= alpha

    # Once we fail to reject, all subsequent are also not rejected
    # (sequential stop rule)
    found_first_failure = False
    for k in range(n):
        if not reject[order[k]]:
            found_first_failure = True
        if found_first_failure:
            reject[order[k]] = False

    return {
        "corrected_p": corrected.tolist(),
        "rejected": reject.tolist(),
        "order": order.tolist(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Cliff's delta (distribution comparison)
# ─────────────────────────────────────────────────────────────────────────────

def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cliff's delta effect size for two samples.

    Returns float in [-1, 1]. Positive = group1 tends to be larger than group2.
    |d| < 0.11 = negligible, 0.11–0.28 = small, 0.28–0.43 = medium, >0.43 = large.
    """
    n1, n2 = len(group1), len(group2)
    more = sum(1 for a in group1 for b in group2 if a > b)
    less = sum(1 for a in group1 for b in group2 if a < b)
    return (more - less) / (n1 * n2)


# ─────────────────────────────────────────────────────────────────────────────
# Reliability diagram data (calibration)
# ─────────────────────────────────────────────────────────────────────────────

def reliability_diagram_data(
    scores: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Compute binned mean prediction vs. actual positive rate for calibration.

    Returns DataFrame with columns: bin_center, mean_score, fraction_positive, count.
    """
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    bins = np.linspace(0, 1, n_bins + 1)
    records = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (scores >= lo) & (scores < hi)
        if mask.sum() == 0:
            continue
        records.append({
            "bin_center": (lo + hi) / 2,
            "mean_score": float(scores[mask].mean()),
            "fraction_positive": float(labels[mask].mean()),
            "count": int(mask.sum()),
        })
    return pd.DataFrame(records)
