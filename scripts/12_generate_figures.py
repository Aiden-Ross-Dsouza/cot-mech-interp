"""
scripts/12_generate_figures.py
Generate all paper figures and Table 1 from cached artifacts.

Outputs:
  analysis/figures/f1_method.svg        — Pipeline schematic (programmatic)
  analysis/figures/f2_correlation.pdf   — AGD vs. AOC scatter + CI band (H1)
  analysis/figures/f3_auroc.pdf         — ROC comparison (H2): AGD vs. baselines
  analysis/table1.tex                   — Per-task results table (LaTeX)

This script is intentionally model-free: it only reads parquet / JSON
artifacts, so it can be run quickly on any machine once artifacts exist.

Usage:
    python scripts/12_generate_figures.py --config config.yaml
    python scripts/12_generate_figures.py --config config.yaml --fig f2 f3
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import load_config

# Matplotlib ICML-compatible style
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── ICML-compatible style constants ──────────────────────────────────────────
FONT_FAMILY = "DejaVu Serif"   # closest serif to Computer Modern available without LaTeX
FONT_SIZE_BASE = 9
FONT_SIZE_TITLE = 10
LINE_WIDTH = 1.2
MARKER_SIZE = 4
COLOR_AGD  = "#2563EB"   # blue
COLOR_ACTCOS = "#DC2626" # red
COLOR_PPL  = "#16A34A"   # green
COLOR_SC   = "#9333EA"   # purple
COLOR_NULL = "#6B7280"   # gray
ALPHA_FILL = 0.15

plt.rcParams.update({
    "font.family": FONT_FAMILY,
    "font.size": FONT_SIZE_BASE,
    "axes.titlesize": FONT_SIZE_TITLE,
    "axes.labelsize": FONT_SIZE_BASE,
    "xtick.labelsize": FONT_SIZE_BASE - 1,
    "ytick.labelsize": FONT_SIZE_BASE - 1,
    "legend.fontsize": FONT_SIZE_BASE - 1,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# F1 — Method diagram (programmatic matplotlib)
# ─────────────────────────────────────────────────────────────────────────────

def draw_f1_method(cfg, out_path: Path):
    """Draw a clean pipeline schematic as SVG."""
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")

    # Box helper
    def box(x, y, w, h, color, label, sublabel=""):
        rect = mpatches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.08",
            facecolor=color, edgecolor="#374151", linewidth=0.8,
        )
        ax.add_patch(rect)
        cx, cy = x + w / 2, y + h / 2
        ax.text(cx, cy + (0.12 if sublabel else 0), label,
                ha="center", va="center", fontsize=7.5, fontweight="bold", color="#1F2937")
        if sublabel:
            ax.text(cx, cy - 0.18, sublabel,
                    ha="center", va="center", fontsize=6.5, color="#4B5563")

    # Arrow helper
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#374151", lw=0.9))

    # Boxes
    box(0.1, 1.5, 2.1, 0.9, "#DBEAFE", "Prompt x", "Question + CoT c")
    box(0.1, 0.3, 2.1, 0.9, "#FEF3C7", "Perturbed", "CoT c\u2019")

    box(2.6, 0.9, 2.1, 1.1, "#D1FAE5", "Gemma-2-2B\n+ Gemma Scope", "PLT transcoders")

    box(5.1, 1.5, 2.1, 0.9, "#E0E7FF", "Graph G\u2080", "clean")
    box(5.1, 0.3, 2.1, 0.9, "#FCE7F3", "Graph G\u2081", "perturbed")

    box(7.6, 0.9, 2.2, 1.1, "#FEE2E2", "AGD(G\u2080, G\u2081)", "1 \u2212 \u03b1\u00b7J_w \u2212 (1\u2212\u03b1)\u00b7S_e")

    # Arrows
    arrow(2.2, 1.95, 2.6, 1.45)  # prompt → model
    arrow(2.2, 0.75, 2.6, 1.10)  # perturbed → model
    arrow(4.7, 1.45, 5.1, 1.95)  # model → G0
    arrow(4.7, 1.10, 5.1, 0.75)  # model → G1
    arrow(7.2, 1.95, 7.6, 1.45)  # G0 → AGD
    arrow(7.2, 0.75, 7.6, 1.10)  # G1 → AGD

    # Labels above arrows
    ax.text(2.42, 1.70, "forward\npass", fontsize=5.5, ha="center", color="#6B7280")
    ax.text(7.42, 1.70, "compare", fontsize=5.5, ha="center", color="#6B7280")

    # Title
    ax.text(5.0, 3.7, "AGD Pipeline: Paired Attribution Graph Divergence",
            ha="center", va="top", fontsize=9, fontweight="bold", color="#1F2937")

    fig.tight_layout()
    fig.savefig(out_path, format="svg")
    plt.close(fig)
    logger.info(f"F1 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# F2 — AGD vs. AOC scatter (H1)
# ─────────────────────────────────────────────────────────────────────────────

def draw_f2_correlation(cfg, agd_df: pd.DataFrame, aoc_df: pd.DataFrame,
                         results: dict, out_path: Path):
    """Scatter: AGD (x) vs. AOC (y) for Regime B, coloured by task."""
    # Merge
    regime_col = "regime_label" if "regime_label" in agd_df.columns else "regime"
    b_agd = agd_df[agd_df[regime_col].str.contains("B", na=False)].copy()
    
    # BCa fix: merge on base_item_id (e.g. bbh_0001_trunc25) -> aoc item_id (e.g. bbh_0001)
    merge_left = "base_item_id" if "base_item_id" in b_agd.columns else "item_id"
    merged = b_agd.merge(aoc_df, left_on=merge_left, right_on="item_id", how="inner")
    merged = merged.dropna(subset=["agd", "aoc_composite"])
    if merged.empty:
        logger.warning("F2: no data to plot.")
        return

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    # Scatter by regime_label
    for label, color in [("B_trunc", "#2563EB"), ("B_mistake", "#DC2626")]:
        sub = merged[merged.get("fname", merged.get("regime_label", "")) == label] \
              if "regime_label" in merged.columns else merged
        ax.scatter(sub["agd"], sub["aoc_composite"],
                   c=color, s=12, alpha=0.55, linewidths=0, label=label.replace("_", " "))

    # Regression line
    x = merged["agd"].values
    y = merged["aoc_composite"].values
    m, b_coef = np.polyfit(x, y, 1)
    xs = np.linspace(x.min(), x.max(), 100)
    ax.plot(xs, m * xs + b_coef, color="#374151", lw=1.0, ls="--", label="OLS fit")

    # Annotate rho
    h1 = results.get("H1_spearman", {})
    rho = h1.get("rho", None)
    ci_lo = h1.get("ci_lo", None)
    ci_hi = h1.get("ci_hi", None)
    if rho is not None:
        label_str = (
            rf"$\rho={rho:.2f}$ [{ci_lo:.2f}, {ci_hi:.2f}]"
            if ci_lo is not None else rf"$\rho={rho:.2f}$"
        )
        ax.text(0.97, 0.05, label_str, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7.5,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))

    ax.set_xlabel("AGD", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("AOC (Lanham, composite)", fontsize=FONT_SIZE_BASE)
    ax.set_title("F2: AGD vs. Behavioral Faithfulness (Regime B)", fontsize=FONT_SIZE_TITLE)
    ax.legend(frameon=False, fontsize=FONT_SIZE_BASE - 1)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    logger.info(f"F2 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# F3 — ROC comparison for hint-flip prediction (H2)
# ─────────────────────────────────────────────────────────────────────────────

def draw_f3_auroc(cfg, agd_df: pd.DataFrame, flip_df: pd.DataFrame,
                   base_df: Optional[pd.DataFrame], results: dict, out_path: Path):
    """ROC curves for AGD vs. baselines on hint-flip prediction."""
    from sklearn.metrics import roc_curve

    regime_col = "regime_label" if "regime_label" in agd_df.columns else "regime"
    c_agd = agd_df[agd_df[regime_col].str.contains("C", na=False)].copy()
    # BCa fix: drop pre-existing unfaithful_flip to avoid _x/_y suffix issue
    if "unfaithful_flip" in c_agd.columns:
        c_agd = c_agd.drop(columns=["unfaithful_flip"])
    merged = c_agd.merge(flip_df[["item_id", "unfaithful_flip"]], on="item_id", how="inner")
    merged = merged.dropna(subset=["agd", "unfaithful_flip"])
    if merged.empty or merged["unfaithful_flip"].sum() < 5:
        logger.warning("F3: insufficient Regime C flip data to plot.")
        return

    if base_df is not None:
        merged = merged.merge(base_df, on="item_id", how="left")

    labels = merged["unfaithful_flip"].astype(int).values

    fig, ax = plt.subplots(figsize=(3.3, 3.0))

    def plot_roc(scores, label, color, ls="-"):
        fpr, tpr, _ = roc_curve(labels, scores)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels, scores)
        ax.plot(fpr, tpr, color=color, lw=LINE_WIDTH, ls=ls,
                label=f"{label} (AUC={auc:.3f})")

    plot_roc(merged["agd"].fillna(0).values, "AGD", COLOR_AGD)

    if base_df is not None and "activation_cosine" in merged.columns:
        # Distance = 1 - cosine for the baseline
        plot_roc((1 - merged["activation_cosine"].fillna(0.5)).values,
                 "Act-cosine", COLOR_ACTCOS, ls="--")

    if base_df is not None and "kl_next_token" in merged.columns:
        plot_roc(merged["kl_next_token"].fillna(0).values, "KL", COLOR_PPL, ls="-.")

    if base_df is not None and "random_jaccard" in merged.columns:
        plot_roc((1 - merged["random_jaccard"].fillna(0.5)).values,
                 "Random-Jaccard (null)", COLOR_NULL, ls=":")

    # Diagonal
    ax.plot([0, 1], [0, 1], color="#9CA3AF", lw=0.8, ls=":", label="Random")

    # H2 CI annotation
    h2 = results.get("H2_auroc_agd", {})
    if h2:
        ax.text(0.97, 0.05,
                f"AGD CI: [{h2.get('ci_lo', 0):.3f}, {h2.get('ci_hi', 0):.3f}]",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB"))

    ax.set_xlabel("False positive rate", fontsize=FONT_SIZE_BASE)
    ax.set_ylabel("True positive rate", fontsize=FONT_SIZE_BASE)
    ax.set_title("F3: ROC for Hint-Flip Prediction (Regime C)", fontsize=FONT_SIZE_TITLE)
    ax.legend(frameon=False, fontsize=FONT_SIZE_BASE - 2, loc="lower right")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, format="pdf")
    plt.close(fig)
    logger.info(f"F3 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# T1 — Per-task results table
# ─────────────────────────────────────────────────────────────────────────────

def generate_table1(cfg, agd_df: pd.DataFrame, aoc_df: Optional[pd.DataFrame],
                     results: dict, out_path: Path):
    """Generate LaTeX table with per-task AGD mean ± std and AOC."""
    rows = []

    task_col = "task" if "task" in agd_df.columns else ("regime_label" if "regime_label" in agd_df.columns else "regime")
    for task, group in agd_df.groupby(task_col):
        valid = group.dropna(subset=["agd"])
        if valid.empty:
            continue

        agd_mean = valid["agd"].mean()
        agd_std = valid["agd"].std()

        # AOC for this task (if available)
        aoc_mean = None
        if aoc_df is not None and "task" in aoc_df.columns:
            task_aoc = aoc_df[aoc_df["task"] == task]
            if not task_aoc.empty:
                aoc_mean = task_aoc["aoc_composite"].mean()

        rows.append({
            "Task": str(task).replace("_", r"\_"),
            "Regime": valid.get("regime", valid.get("fname", pd.Series([""]))).iloc[0][:1],
            r"AGD ($\bar{x} \pm \sigma$)": f"{agd_mean:.3f} $\\pm$ {agd_std:.3f}",
            "AOC": f"{aoc_mean:.3f}" if aoc_mean is not None else "—",
            "N": len(valid),
        })

    if not rows:
        logger.warning("T1: no per-task data to tabulate.")
        return

    df_table = pd.DataFrame(rows)

    # Build LaTeX
    header = r"""
\begin{table}[t]
\centering
\small
\caption{Per-task AGD and behavioral faithfulness (AOC) on the held-out test set.
         Higher AOC = more post-hoc reasoning. $N$ = number of test pairs.}
\label{tab:per_task}
\begin{tabular}{llccc}
\toprule
Task & Regime & AGD ($\bar{x} \pm \sigma$) & AOC & $N$ \\
\midrule
"""
    footer = r"""
\bottomrule
\end{tabular}
\end{table}
"""

    lines = []
    # BCa fix: move LaTeX key access outside f-string to avoid backslash SyntaxError
    agd_col = r"AGD ($\bar{x} \pm \sigma$)"
    for _, row in df_table.iterrows():
        agd_val = row[agd_col]
        lines.append(
            f"{row['Task']} & "
            f"{row['Regime']} & "
            f"{agd_val} & "
            f"{row['AOC']} & "
            f"{row['N']} \\\\"
        )

    tex = header + "\n".join(lines) + footer
    out_path.write_text(tex, encoding="utf-8")
    logger.info(f"T1 saved → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--fig", nargs="*", default=["f1", "f2", "f3", "t1"],
                        choices=["f1", "f2", "f3", "t1"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    fig_dir = Path(cfg.paths.figures)
    fig_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = Path(cfg.paths.analysis)

    # Load artifacts
    agd_path = Path(cfg.paths.agd) / "pairs.parquet"
    agd_df = pd.read_parquet(agd_path) if agd_path.exists() else pd.DataFrame()

    behavior_dir = Path(cfg.paths.behavioral)
    aoc_df = pd.read_parquet(behavior_dir / "aoc_lanham.parquet") \
        if (behavior_dir / "aoc_lanham.parquet").exists() else None
    flip_df = pd.read_parquet(behavior_dir / "turpin_flips.parquet") \
        if (behavior_dir / "turpin_flips.parquet").exists() else None
    base_df = pd.read_parquet(Path(cfg.paths.agd).parent / "baselines.parquet") \
        if (Path(cfg.paths.agd).parent / "baselines.parquet").exists() else None

    results_path = analysis_dir / "results_test.json"
    results = json.loads(results_path.read_text()) if results_path.exists() else {}

    if "f1" in args.fig:
        draw_f1_method(cfg, fig_dir / "f1_method.svg")

    if "f2" in args.fig:
        if not agd_df.empty and aoc_df is not None:
            draw_f2_correlation(cfg, agd_df, aoc_df, results, fig_dir / "f2_correlation.pdf")
        else:
            logger.warning("F2: missing AGD or AOC data.")

    if "f3" in args.fig:
        if not agd_df.empty and flip_df is not None:
            draw_f3_auroc(cfg, agd_df, flip_df, base_df, results, fig_dir / "f3_auroc.pdf")
        else:
            logger.warning("F3: missing AGD or flip data.")

    if "t1" in args.fig:
        generate_table1(cfg, agd_df, aoc_df, results, analysis_dir / "table1.tex")

    logger.info("\n✓ Figure generation complete.")


if __name__ == "__main__":
    main()
