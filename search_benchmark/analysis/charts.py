"""
analysis/charts.py
------------------
All 7 publication-quality charts for the benchmarking report.

Each function accepts a pandas DataFrame (output of BenchmarkRunner.run_suite)
and saves a PNG to results/charts/.  All charts use dark background style.

Charts
------
C1  nodes_expanded_vs_size      — line + CI bands, log y-axis
C2  runtime_vs_size             — line + CI bands
C3  peak_memory_vs_size         — grouped bar
C4  path_cost_optimality_gap    — horizontal bar, normalised to UCS=1
C5  heuristic_error_boxplot     — box-plot per heuristic type
C6  frontier_growth             — step-plot of frontier size over expansions
C7  success_rate_heatmap        — algo × experiment category heatmap
"""

from __future__ import annotations
import os
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

# ── Style constants ──────────────────────────────────────────────────────────
DARK_BG    = "#0F172A"
PANEL_BG   = "#1E293B"
BORDER_CLR = "#334155"
TEXT_CLR   = "#F1F5F9"
MUTED_CLR  = "#94A3B8"

ALGO_COLORS = {
    "BFS":    "#3B82F6",   # blue
    "DFS":    "#A78BFA",   # purple
    "UCS":    "#22C55E",   # green
    "Greedy": "#F59E0B",   # amber
    "A*":     "#EF4444",   # red
}
ALGO_ORDER = ["BFS", "DFS", "UCS", "Greedy", "A*"]

CHART_DIR = "results/charts"
DPI       = 150


def _setup_fig(w=10, h=6):
    fig, ax = plt.subplots(figsize=(w, h), facecolor=DARK_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors=MUTED_CLR, labelsize=9)
    ax.xaxis.label.set_color(TEXT_CLR)
    ax.yaxis.label.set_color(TEXT_CLR)
    ax.title.set_color(TEXT_CLR)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER_CLR)
    ax.grid(True, color=BORDER_CLR, linewidth=0.5, alpha=0.6)
    return fig, ax


def _save(fig, name: str, chart_dir: str = CHART_DIR):
    os.makedirs(chart_dir, exist_ok=True)
    path = os.path.join(chart_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor=DARK_BG)
    plt.close(fig)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# C1 — Nodes Expanded vs Problem Size
# ─────────────────────────────────────────────────────────────────────────────

def chart_c1_nodes_expanded(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """
    Line chart with shaded ±1 std bands.
    X = problem size (nodes or grid dimension).
    Y = nodes_expanded (log scale).
    One coloured line per algorithm.
    """
    fig, ax = _setup_fig(10, 6)

    size_col = "size"
    if size_col not in df.columns:
        ax.text(0.5, 0.5, "No 'size' column in DataFrame",
                ha="center", va="center", color=TEXT_CLR, transform=ax.transAxes)
        return _save(fig, "c1_nodes_expanded", chart_dir)

    for algo in ALGO_ORDER:
        sub = df[df["algo"] == algo].copy()
        if sub.empty:
            continue
        grp  = sub.groupby(size_col)["nodes_expanded"]
        mean = grp.mean()
        std  = grp.std().fillna(0)
        x    = mean.index.values
        color = ALGO_COLORS.get(algo, "#888")
        ax.plot(x, mean.values, marker="o", markersize=5,
                color=color, linewidth=2, label=algo)
        ax.fill_between(x,
                        np.maximum(mean - std, 1),
                        mean + std,
                        alpha=0.15, color=color)

    ax.set_yscale("log")
    ax.set_xlabel("Problem size (nodes / grid side)", fontsize=11)
    ax.set_ylabel("Nodes expanded  (log scale)", fontsize=11)
    ax.set_title("C1  —  Nodes Expanded vs Problem Size", fontsize=13, pad=12)
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "c1_nodes_expanded", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C2 — Runtime vs Problem Size
# ─────────────────────────────────────────────────────────────────────────────

def chart_c2_runtime(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """
    Line chart with shaded ±1 std bands.
    Y = runtime_ms (linear scale).
    """
    fig, ax = _setup_fig(10, 6)

    size_col = "size"
    if size_col not in df.columns or "runtime_ms" not in df.columns:
        return _save(fig, "c2_runtime", chart_dir)

    for algo in ALGO_ORDER:
        sub  = df[df["algo"] == algo].copy()
        if sub.empty:
            continue
        grp  = sub.groupby(size_col)["runtime_ms"]
        mean = grp.mean()
        std  = grp.std().fillna(0)
        x    = mean.index.values
        color = ALGO_COLORS.get(algo, "#888")
        ax.plot(x, mean.values, marker="s", markersize=5,
                color=color, linewidth=2, label=algo)
        ax.fill_between(x,
                        np.maximum(mean - std, 0),
                        mean + std,
                        alpha=0.15, color=color)

    ax.set_xlabel("Problem size", fontsize=11)
    ax.set_ylabel("Runtime  (ms)", fontsize=11)
    ax.set_title("C2  —  Runtime vs Problem Size", fontsize=13, pad=12)
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "c2_runtime", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C3 — Peak Memory vs Problem Size
# ─────────────────────────────────────────────────────────────────────────────

def chart_c3_memory(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """Grouped bar chart — one cluster per size, one bar per algorithm."""
    fig, ax = _setup_fig(12, 6)

    size_col = "size"
    if size_col not in df.columns or "peak_memory_kb" not in df.columns:
        return _save(fig, "c3_memory", chart_dir)

    sizes = sorted(df[size_col].unique())
    algos = [a for a in ALGO_ORDER if a in df["algo"].unique()]
    n_algos = len(algos)
    x_base  = np.arange(len(sizes))
    bar_w   = 0.8 / n_algos

    for i, algo in enumerate(algos):
        sub    = df[df["algo"] == algo]
        means  = [sub[sub[size_col] == s]["peak_memory_kb"].mean() for s in sizes]
        offset = (i - n_algos / 2 + 0.5) * bar_w
        ax.bar(x_base + offset, means, width=bar_w * 0.9,
               color=ALGO_COLORS.get(algo, "#888"),
               label=algo, alpha=0.85, edgecolor=DARK_BG, linewidth=0.5)

    ax.set_xticks(x_base)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.set_xlabel("Problem size", fontsize=11)
    ax.set_ylabel("Peak memory  (KB)", fontsize=11)
    ax.set_title("C3  —  Peak Memory vs Problem Size", fontsize=13, pad=12)
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "c3_memory", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C4 — Path Cost Optimality Gap
# ─────────────────────────────────────────────────────────────────────────────

def chart_c4_optimality(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """
    Horizontal bar chart.
    Each bar = mean path_cost / mean UCS path_cost  (normalised to UCS = 1.0).
    Lower = better; UCS and A* should be at 1.0.
    """
    fig, ax = _setup_fig(9, 5)

    if "path_cost" not in df.columns:
        return _save(fig, "c4_optimality", chart_dir)

    ucs_mean = df[df["algo"] == "UCS"]["path_cost"].mean()
    if ucs_mean is None or ucs_mean == 0:
        ucs_mean = 1.0

    means = {}
    for algo in ALGO_ORDER:
        sub = df[df["algo"] == algo]["path_cost"].dropna()
        if not sub.empty:
            means[algo] = sub.mean() / ucs_mean

    algos  = list(means.keys())
    values = [means[a] for a in algos]
    colors = [ALGO_COLORS.get(a, "#888") for a in algos]
    y_pos  = np.arange(len(algos))

    bars = ax.barh(y_pos, values, color=colors, alpha=0.85,
                   edgecolor=DARK_BG, linewidth=0.5, height=0.55)

    # Reference line at 1.0 (UCS optimal)
    ax.axvline(1.0, color=TEXT_CLR, linewidth=1.2, linestyle="--", alpha=0.5)
    ax.text(1.02, len(algos) - 0.3, "UCS optimal", color=MUTED_CLR, fontsize=8)

    # Value labels
    for bar, val in zip(bars, values):
        ax.text(val + 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}×", va="center", color=TEXT_CLR, fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(algos, color=TEXT_CLR)
    ax.set_xlabel("Path cost  (normalised to UCS = 1.0)", fontsize=11)
    ax.set_title("C4  —  Path Cost Optimality Gap", fontsize=13, pad=12)
    ax.set_xlim(0, max(values) * 1.18)
    fig.tight_layout()
    return _save(fig, "c4_optimality", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C5 — Heuristic Error Distribution
# ─────────────────────────────────────────────────────────────────────────────

def chart_c5_heuristic_error(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """
    Box-plot of |h(n) - h*(n)| for A* and Greedy, grouped by heuristic name.
    """
    fig, ax = _setup_fig(10, 6)

    if "heuristic_error" not in df.columns or "heuristic_name" not in df.columns:
        ax.text(0.5, 0.5, "No heuristic_error / heuristic_name columns",
                ha="center", va="center", color=TEXT_CLR, transform=ax.transAxes)
        return _save(fig, "c5_heuristic_error", chart_dir)

    algos_h   = ["A*", "Greedy"]
    h_names   = sorted(df["heuristic_name"].dropna().unique())
    positions  = []
    data_sets  = []
    labels_    = []
    colors_    = []
    n_h = len(h_names)

    for i, h in enumerate(h_names):
        for j, algo in enumerate(algos_h):
            sub = df[
                (df["algo"] == algo) &
                (df["heuristic_name"] == h) &
                (df["heuristic_error"].notna())
            ]["heuristic_error"]
            if not sub.empty:
                positions.append(i * (len(algos_h) + 1) + j)
                data_sets.append(sub.values)
                labels_.append(f"{h}\n{algo}")
                colors_.append(ALGO_COLORS.get(algo, "#888"))

    if not data_sets:
        ax.text(0.5, 0.5, "No heuristic_error data available",
                ha="center", va="center", color=TEXT_CLR, transform=ax.transAxes)
        return _save(fig, "c5_heuristic_error", chart_dir)

    bp = ax.boxplot(data_sets, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=True,
                    flierprops=dict(marker=".", color=MUTED_CLR, markersize=3),
                    medianprops=dict(color=TEXT_CLR, linewidth=1.5))

    for patch, color in zip(bp["boxes"], colors_):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    for whisker in bp["whiskers"]:
        whisker.set_color(MUTED_CLR)
    for cap in bp["caps"]:
        cap.set_color(MUTED_CLR)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels_, fontsize=8, color=TEXT_CLR)
    ax.set_ylabel("|h(n) − h*(n)|  (heuristic error)", fontsize=11)
    ax.set_title("C5  —  Heuristic Error Distribution", fontsize=13, pad=12)

    legend_elements = [
        Line2D([0], [0], color=ALGO_COLORS["A*"],     linewidth=4, label="A*"),
        Line2D([0], [0], color=ALGO_COLORS["Greedy"], linewidth=4, label="Greedy"),
    ]
    ax.legend(handles=legend_elements, facecolor=PANEL_BG,
              edgecolor=BORDER_CLR, labelcolor=TEXT_CLR, fontsize=9)
    fig.tight_layout()
    return _save(fig, "c5_heuristic_error", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C6 — Frontier Growth Over Time
# ─────────────────────────────────────────────────────────────────────────────

def chart_c6_frontier_growth(
    expansion_logs: dict,
    chart_dir: str = CHART_DIR,
) -> str:
    """
    Step-plot of frontier size at each expansion step.

    Parameters
    ----------
    expansion_logs : dict mapping algo_name → list of frontier-size ints
                     (i.e. [len(frontier) at step 0, step 1, …])
    """
    fig, ax = _setup_fig(10, 6)

    if not expansion_logs:
        ax.text(0.5, 0.5, "No expansion log data provided",
                ha="center", va="center", color=TEXT_CLR, transform=ax.transAxes)
        return _save(fig, "c6_frontier_growth", chart_dir)

    for algo, sizes in expansion_logs.items():
        if not sizes:
            continue
        color = ALGO_COLORS.get(algo, "#888")
        ax.step(range(len(sizes)), sizes, where="post",
                color=color, linewidth=1.8, label=algo, alpha=0.9)

    ax.set_xlabel("Expansion step", fontsize=11)
    ax.set_ylabel("Frontier size", fontsize=11)
    ax.set_title("C6  —  Frontier Size Growth Over Time", fontsize=13, pad=12)
    _legend(ax)
    fig.tight_layout()
    return _save(fig, "c6_frontier_growth", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# C7 — Algorithm Success Rate Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def chart_c7_success_heatmap(df: pd.DataFrame, chart_dir: str = CHART_DIR) -> str:
    """
    Heatmap: rows = algorithm, columns = experiment category.
    Cell value = success_rate (0.0 – 1.0).
    """
    fig, ax = _setup_fig(10, 5)

    if "experiment" not in df.columns:
        ax.text(0.5, 0.5, "No 'experiment' column in DataFrame",
                ha="center", va="center", color=TEXT_CLR, transform=ax.transAxes)
        return _save(fig, "c7_success_heatmap", chart_dir)

    # Filter to standard 5 algos for the heatmap
    df_std = df[df["algo"].isin(ALGO_ORDER)]
    pivot  = (
        df_std.groupby(["algo", "experiment"])["solution_found"]
        .mean()
        .unstack(fill_value=0.0)
    )
    pivot = pivot.reindex(index=[a for a in ALGO_ORDER if a in pivot.index])

    im = ax.imshow(pivot.values, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"Exp {c}" for c in pivot.columns],
                       color=TEXT_CLR, fontsize=10)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.tolist(), color=TEXT_CLR, fontsize=10)

    # Annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val  = pivot.values[i, j]
            text = f"{val:.0%}"
            tc   = "black" if val > 0.55 else TEXT_CLR
            ax.text(j, i, text, ha="center", va="center",
                    color=tc, fontsize=10, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    cbar.ax.tick_params(colors=MUTED_CLR, labelsize=8)
    cbar.set_label("Success rate", color=TEXT_CLR, fontsize=9)

    ax.set_title("C7  —  Algorithm Success Rate by Experiment", fontsize=13, pad=12)
    fig.tight_layout()
    return _save(fig, "c7_success_heatmap", chart_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: generate all charts from a combined DataFrame
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_charts(
    df:             pd.DataFrame,
    expansion_logs: Optional[dict] = None,
    chart_dir:      str = CHART_DIR,
    verbose:        bool = True,
) -> dict:
    """
    Generate all 7 charts from a combined experiment DataFrame.

    Parameters
    ----------
    df             : Combined DataFrame from run_all_experiments().
    expansion_logs : Optional dict for C6 (algo → frontier size list).
                     If None, C6 is skipped / shows placeholder.
    chart_dir      : Output directory for PNGs.
    verbose        : Print saved paths.

    Returns
    -------
    Dict mapping chart_id → file path.
    """
    saved = {}
    charts = [
        ("C1", lambda: chart_c1_nodes_expanded(df, chart_dir)),
        ("C2", lambda: chart_c2_runtime(df, chart_dir)),
        ("C3", lambda: chart_c3_memory(df, chart_dir)),
        ("C4", lambda: chart_c4_optimality(df, chart_dir)),
        ("C5", lambda: chart_c5_heuristic_error(df, chart_dir)),
        ("C6", lambda: chart_c6_frontier_growth(expansion_logs or {}, chart_dir)),
        ("C7", lambda: chart_c7_success_heatmap(df, chart_dir)),
    ]
    for cid, fn in charts:
        try:
            path = fn()
            saved[cid] = path
            if verbose:
                print(f"  {cid} → {path}")
        except Exception as e:
            if verbose:
                print(f"  {cid} ERROR: {e}")
    return saved


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _legend(ax):
    leg = ax.legend(
        facecolor=PANEL_BG, edgecolor=BORDER_CLR,
        labelcolor=TEXT_CLR, fontsize=9,
        framealpha=0.9,
    )
