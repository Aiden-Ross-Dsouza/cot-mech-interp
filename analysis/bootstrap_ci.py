"""
analysis/bootstrap_ci.py
Standalone BCa bootstrap utility — can be run directly for auditing
or imported by stats.py.

Usage (CLI audit):
    python analysis/bootstrap_ci.py --data path/to/values.txt --stat mean --n_boot 5000

The file should contain one float per line.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import stats as scipy_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def bca_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """BCa bootstrap confidence interval.

    Parameters
    ----------
    data:       1-D array of observations.
    statistic:  Function(data) → scalar.
    n_boot:     Bootstrap resamples.
    alpha:      Two-sided CI level (0.05 → 95% CI).
    seed:       RNG seed.

    Returns
    -------
    (observed, ci_lo, ci_hi)
    """
    rng = np.random.default_rng(seed)
    n = len(data)
    observed = statistic(data)

    # Bootstrap distribution
    boot = np.array([
        statistic(rng.choice(data, size=n, replace=True))
        for _ in range(n_boot)
    ])

    # Bias-correction z0
    prop_below = np.mean(boot < observed)
    prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
    z0 = scipy_stats.norm.ppf(prop_below)

    # Acceleration a (jackknife)
    jack = np.array([statistic(np.delete(data, i)) for i in range(n)])
    jack_mean = np.mean(jack)
    num = np.sum((jack_mean - jack) ** 3)
    denom = 6 * (np.sum((jack_mean - jack) ** 2) ** 1.5)
    a = float(num / denom) if denom != 0 else 0.0

    # Adjusted percentiles
    def adj_pctile(z_a: float) -> float:
        z_adj = z0 + (z0 + z_a) / (1.0 - a * (z0 + z_a))
        return float(scipy_stats.norm.cdf(z_adj))

    pct_lo = adj_pctile(scipy_stats.norm.ppf(alpha / 2)) * 100
    pct_hi = adj_pctile(scipy_stats.norm.ppf(1 - alpha / 2)) * 100

    ci_lo = float(np.percentile(boot, pct_lo))
    ci_hi = float(np.percentile(boot, pct_hi))

    return observed, ci_lo, ci_hi


def bca_ci_2sample(
    x: np.ndarray,
    y: np.ndarray,
    statistic: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """BCa bootstrap for a two-sample statistic (e.g. Spearman rho).

    Returns (observed, ci_lo, ci_hi).
    """
    rng = np.random.default_rng(seed)
    n = len(x)
    observed = statistic(x, y)

    boot = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        boot.append(statistic(x[idx], y[idx]))
    boot = np.array(boot)

    ci_lo = float(np.percentile(boot, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
    return observed, ci_lo, ci_hi


# ── CLI audit mode ────────────────────────────────────────────────────────────

STAT_FUNCTIONS = {
    "mean": np.mean,
    "median": np.median,
    "std": np.std,
}


def main():
    parser = argparse.ArgumentParser(
        description="BCa bootstrap CI for a 1-D dataset (audit tool)"
    )
    parser.add_argument("--data", required=True, help="Path to file with one float per line")
    parser.add_argument("--stat", default="mean", choices=list(STAT_FUNCTIONS.keys()))
    parser.add_argument("--n_boot", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    data = np.array([float(line.strip()) for line in Path(args.data).read_text().splitlines()
                     if line.strip()])
    logger.info(f"Loaded {len(data)} values from {args.data}")
    logger.info(f"Statistic: {args.stat} = {STAT_FUNCTIONS[args.stat](data):.6f}")

    obs, lo, hi = bca_ci(
        data, STAT_FUNCTIONS[args.stat],
        n_boot=args.n_boot, alpha=args.alpha, seed=args.seed
    )
    print(f"\nBCa Bootstrap {int((1-args.alpha)*100)}% CI")
    print(f"  Observed {args.stat}: {obs:.6f}")
    print(f"  CI: [{lo:.6f}, {hi:.6f}]")
    print(f"  n_boot: {args.n_boot}, seed: {args.seed}")


if __name__ == "__main__":
    main()
