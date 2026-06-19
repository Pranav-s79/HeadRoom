"""Shared visual layer for the HeadRoom dashboard.

This module is intentionally free of any Streamlit / thermalguard_cal imports so
that it can be reused both by the live dashboard (``dashboard/app.py``) and by the
static asset generator (``dashboard/generate_assets.py``).

Everything here is pure matplotlib + numpy + pandas and obeys the locked HeadRoom
visual identity (dark charcoal, amber accent, monospace numeric readouts).
"""

from __future__ import annotations

from typing import Sequence

import matplotlib

matplotlib.use("Agg")  # headless-safe; Streamlit renders the returned figures

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

# --------------------------------------------------------------------------- #
# Locked visual identity
# --------------------------------------------------------------------------- #
PALETTE = {
    "bg": "#1a1a1a",        # dark charcoal background
    "panel": "#242424",     # panel / card background
    "amber": "#f59e0b",     # amber accent (primary)
    "amber_hi": "#d97706",  # amber hover / highlight
    "text": "#f5f5f5",      # white primary text
    "muted": "#9ca3af",     # muted secondary text
    "red": "#ef4444",       # danger / violation
    "green": "#22c55e",     # safe
    "yellow": "#eab308",    # warning
    "border": "#374151",    # border / divider
}

SANS_STACK = ["Inter", "Helvetica Neue", "Arial", "DejaVu Sans", "sans-serif"]
MONO_STACK = ["JetBrains Mono", "Fira Code", "DejaVu Sans Mono", "monospace"]

# Fixed temperature scale used everywhere so heatmaps are honestly comparable.
TEMP_VMIN = 35.0
TEMP_VMAX = 90.0
THERMAL_LIMIT = 85.0

# Custom dark -> amber -> red gradient (perceptually monotone, on-brand).
THERMAL_CMAP = LinearSegmentedColormap.from_list(
    "headroom_thermal",
    [
        (0.00, "#16233b"),  # 35C  cool, dark blue-charcoal
        (0.30, "#1f5673"),  # ~52C teal
        (0.50, "#f59e0b"),  # ~63C amber accent
        (0.74, "#f97316"),  # ~76C orange
        (0.90, "#ef4444"),  # ~85C thermal limit -> red
        (1.00, "#7f1d1d"),  # 90C  deep red
    ],
)
THERMAL_CMAP.set_bad(PALETTE["border"])


def apply_mpl_theme() -> None:
    """Force every matplotlib figure onto the HeadRoom dark identity."""
    plt.rcParams.update(
        {
            "figure.facecolor": PALETTE["bg"],
            "savefig.facecolor": PALETTE["bg"],
            "axes.facecolor": PALETTE["panel"],
            "axes.edgecolor": PALETTE["border"],
            "axes.labelcolor": PALETTE["text"],
            "axes.titlecolor": PALETTE["text"],
            "text.color": PALETTE["text"],
            "xtick.color": PALETTE["muted"],
            "ytick.color": PALETTE["muted"],
            "grid.color": PALETTE["border"],
            "font.family": SANS_STACK,
            "font.size": 10,
            "axes.titlesize": 12,
            "figure.autolayout": False,
        }
    )


# --------------------------------------------------------------------------- #
# Heatmap rendering
# --------------------------------------------------------------------------- #
def _draw_heatmap(
    ax: plt.Axes,
    values: Sequence[float],
    *,
    grid_size: int,
    title: str,
    subtitle: str,
    selected_core: int | None,
    sensor_indices: Sequence[int],
    sensor_mask: Sequence[bool] | None = None,
    thermal_limit: float = THERMAL_LIMIT,
    show_colorbar: bool = True,
) -> None:
    grid = np.asarray(values, dtype=float).reshape(grid_size, grid_size)
    image = ax.imshow(grid, cmap=THERMAL_CMAP, vmin=TEMP_VMIN, vmax=TEMP_VMAX)

    ax.set_title(title, color=PALETTE["amber"], fontsize=12, fontweight="bold", pad=14)
    ax.text(
        0.5,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        color=PALETTE["muted"],
        fontsize=8.5,
    )
    ax.set_xticks(range(grid_size))
    ax.set_yticks(range(grid_size))
    ax.set_xticklabels(range(grid_size), color=PALETTE["muted"], fontsize=7)
    ax.set_yticklabels(range(grid_size), color=PALETTE["muted"], fontsize=7)
    for spine in ax.spines.values():
        spine.set_color(PALETTE["border"])

    sensor_set = set(int(s) for s in sensor_indices)
    mask = None if sensor_mask is None else np.asarray(sensor_mask, dtype=bool).reshape(-1)

    for row in range(grid_size):
        for col in range(grid_size):
            core = row * grid_size + col
            value = grid[row, col]
            label = "--" if not np.isfinite(value) else f"{value:.0f}"
            ax.text(
                col,
                row - 0.18,
                f"{core}",
                ha="center",
                va="center",
                color="#0b0b0b",
                fontsize=6.5,
                alpha=0.65,
            )
            ax.text(
                col,
                row + 0.16,
                label,
                ha="center",
                va="center",
                color="#0b0b0b" if (np.isfinite(value) and value > 62) else PALETTE["text"],
                fontsize=9,
                fontweight="bold",
                family=MONO_STACK,
            )
            # Over-thermal-limit cores: red border overlay.
            if np.isfinite(value) and value >= thermal_limit:
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.5, row - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor=PALETTE["red"],
                        linewidth=3,
                        zorder=5,
                    )
                )

    # Sensor cores: small amber circle (open) for the four fixed sensors.
    for core in sensor_set:
        row, col = divmod(core, grid_size)
        observed = True if mask is None else bool(mask[core])
        ax.scatter(
            col,
            row,
            marker="o",
            s=140,
            facecolors="none",
            edgecolors=PALETTE["amber"] if observed else PALETTE["muted"],
            linewidths=1.8 if observed else 1.2,
            linestyle="solid" if observed else "dotted",
            zorder=6,
        )

    # Selected core: larger amber diamond.
    if selected_core is not None:
        row, col = divmod(int(selected_core), grid_size)
        ax.scatter(
            col,
            row,
            marker="D",
            s=260,
            facecolors="none",
            edgecolors=PALETTE["amber"],
            linewidths=2.6,
            zorder=7,
        )

    if show_colorbar:
        cbar = ax.figure.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=PALETTE["muted"], labelsize=7)
        cbar.outline.set_edgecolor(PALETTE["border"])
        cbar.ax.axhline(
            (thermal_limit - TEMP_VMIN) / (TEMP_VMAX - TEMP_VMIN),
            color=PALETTE["red"],
            linewidth=1.5,
        )


def dual_heatmap_figure(
    left_values: Sequence[float],
    right_values: Sequence[float],
    *,
    grid_size: int,
    left_title: str,
    right_title: str,
    left_subtitle: str,
    right_subtitle: str,
    left_selected: int | None,
    right_selected: int | None,
    sensor_indices: Sequence[int],
    left_mask: Sequence[bool] | None = None,
    right_mask: Sequence[bool] | None = None,
    thermal_limit: float = THERMAL_LIMIT,
    figsize: tuple[float, float] = (10.4, 4.6),
    suptitle: str | None = None,
) -> plt.Figure:
    """Side-by-side synchronized heatmaps (the hero visual)."""
    apply_mpl_theme()
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    _draw_heatmap(
        axes[0],
        left_values,
        grid_size=grid_size,
        title=left_title,
        subtitle=left_subtitle,
        selected_core=left_selected,
        sensor_indices=sensor_indices,
        sensor_mask=left_mask,
        thermal_limit=thermal_limit,
    )
    _draw_heatmap(
        axes[1],
        right_values,
        grid_size=grid_size,
        title=right_title,
        subtitle=right_subtitle,
        selected_core=right_selected,
        sensor_indices=sensor_indices,
        sensor_mask=right_mask,
        thermal_limit=thermal_limit,
    )
    if suptitle:
        fig.suptitle(suptitle, color=PALETTE["text"], fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94 if suptitle else 1.0))
    return fig


# --------------------------------------------------------------------------- #
# Max-temperature-over-time chart
# --------------------------------------------------------------------------- #
def max_temp_chart_figure(
    timesteps: Sequence[int],
    conformal_series: Sequence[float],
    coolest_series: Sequence[float],
    *,
    conformal_label: str = "Conformal upper-bound",
    coolest_label: str = "Coolest-core (observed)",
    thermal_limit: float = THERMAL_LIMIT,
    current_t: int | None = None,
    figsize: tuple[float, float] = (10.4, 3.4),
) -> plt.Figure:
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=figsize)
    t = np.asarray(timesteps, dtype=float)
    ax.plot(t, np.asarray(conformal_series, dtype=float), color=PALETTE["amber"], linewidth=2.2, label=conformal_label)
    ax.plot(
        t,
        np.asarray(coolest_series, dtype=float),
        color=PALETTE["muted"],
        linewidth=1.8,
        linestyle="--",
        label=coolest_label,
    )
    ax.axhline(thermal_limit, color=PALETTE["red"], linewidth=1.4, linestyle=(0, (6, 4)))
    ax.text(
        t[0] if len(t) else 0,
        thermal_limit + 1.2,
        f"Thermal limit {thermal_limit:g}C",
        color=PALETTE["red"],
        fontsize=8,
        va="bottom",
    )
    if current_t is not None:
        ax.axvline(current_t, color=PALETTE["amber_hi"], linewidth=1.0, alpha=0.6)
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max chip temp (C)")
    ax.grid(True, color=PALETTE["border"], linewidth=0.5, alpha=0.5)
    ax.set_ylim(TEMP_VMIN, max(TEMP_VMAX, float(np.nanmax(coolest_series)) + 5 if len(coolest_series) else TEMP_VMAX))
    legend = ax.legend(loc="upper left", fontsize=8, facecolor=PALETTE["panel"], edgecolor=PALETTE["border"])
    for txt in legend.get_texts():
        txt.set_color(PALETTE["text"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Horizontal comparison bars (scheduler tab)
# --------------------------------------------------------------------------- #
def hbar_figure(
    labels: Sequence[str],
    values: Sequence[float],
    *,
    title: str,
    xlabel: str,
    color: str,
    reference: float | None = None,
    reference_label: str | None = None,
    figsize: tuple[float, float] = (5.2, 4.2),
    value_fmt: str = "{:.1f}",
    highlight: str | None = None,
) -> plt.Figure:
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    bar_colors = [
        PALETTE["amber"] if (highlight is not None and lab == highlight) else color
        for lab in labels
    ]
    ax.barh(y, values, color=bar_colors, edgecolor=PALETTE["border"], height=0.66)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8, color=PALETTE["text"])
    ax.invert_yaxis()
    ax.set_title(title, color=PALETTE["amber"], fontsize=11, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.grid(True, axis="x", color=PALETTE["border"], linewidth=0.5, alpha=0.5)
    span = max(values) - min(min(values), 0) if len(values) else 1
    for yi, val in zip(y, values):
        ax.text(
            val + span * 0.01,
            yi,
            value_fmt.format(val),
            va="center",
            ha="left",
            fontsize=8,
            color=PALETTE["text"],
            family=MONO_STACK,
        )
    if reference is not None:
        ax.axvline(reference, color=PALETTE["red"], linewidth=1.4, linestyle=(0, (6, 4)))
        if reference_label:
            ax.text(reference, -0.6, reference_label, color=PALETTE["red"], fontsize=8, ha="center")
    fig.tight_layout()
    return fig


def safety_throughput_figure(
    rows: Sequence[dict],
    *,
    thermal_limit: float = THERMAL_LIMIT,
    figsize: tuple[float, float] = (8.0, 5.0),
) -> plt.Figure:
    """Scatter: x = completed tasks, y = peak temperature, one point per scheduler."""
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=figsize)
    for row in rows:
        is_conformal = row.get("is_conformal", False)
        is_oracle = row.get("is_oracle", False)
        color = PALETTE["amber"] if is_conformal else (PALETTE["muted"] if is_oracle else PALETTE["text"])
        ax.scatter(
            row["completed"],
            row["peak"],
            s=200 if is_conformal else 110,
            color=color,
            edgecolors=PALETTE["bg"],
            linewidths=1.2,
            zorder=4,
            marker="*" if is_conformal else ("s" if is_oracle else "o"),
        )
        ax.annotate(
            row["label"],
            (row["completed"], row["peak"]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
            color=color,
        )
    ax.axhline(thermal_limit, color=PALETTE["red"], linewidth=1.3, linestyle=(0, (6, 4)))
    ax.text(ax.get_xlim()[0], thermal_limit + 1, f"Thermal limit {thermal_limit:g}C", color=PALETTE["red"], fontsize=8)
    # "better" arrow toward high throughput + low temperature (bottom-right).
    ax.annotate(
        "better",
        xy=(0.93, 0.08),
        xytext=(0.72, 0.30),
        xycoords="axes fraction",
        textcoords="axes fraction",
        color=PALETTE["green"],
        fontsize=11,
        fontweight="bold",
        ha="center",
        arrowprops=dict(arrowstyle="-|>", color=PALETTE["green"], linewidth=2.0),
    )
    ax.set_xlabel("Completed tasks")
    ax.set_ylabel("Peak temperature (C)")
    ax.set_title("Safety vs throughput", color=PALETTE["amber"], fontsize=11, fontweight="bold")
    ax.grid(True, color=PALETTE["border"], linewidth=0.5, alpha=0.5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Architecture diagram (Section 9)
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, text, *, facecolor, edgecolor, textcolor, fontsize=9, weight="normal"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=1.6,
            mutation_aspect=1.0,
            zorder=3,
        )
    )
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        color=textcolor,
        fontsize=fontsize,
        fontweight=weight,
        zorder=4,
        wrap=True,
    )


def _arrow(ax, x0, y0, x1, y1, color):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y0),
            (x1, y1),
            arrowstyle="-|>",
            mutation_scale=16,
            color=color,
            linewidth=1.8,
            zorder=2,
        )
    )


def architecture_figure(figsize: tuple[float, float] = (12.0, 5.0), dpi: int = 100) -> plt.Figure:
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)

    amber = PALETTE["amber"]
    panel = PALETTE["panel"]
    text = PALETTE["text"]
    border = PALETTE["border"]

    fig.text(0.5, 0.965, "HeadRoom  /  thermalguard_cal pipeline", ha="center",
             color=amber, fontsize=13, fontweight="bold")

    # Stage 1: simulator
    _box(ax, 0.2, 2.05, 1.85, 0.95, "Stochastic\n4x4 Simulator", facecolor=panel, edgecolor=amber, textcolor=text, weight="bold")
    # Stage 2: sensors
    _box(ax, 2.35, 2.05, 1.85, 0.95, "Sparse / Noisy\nSensors\n(4 of 16 cores)", facecolor=panel, edgecolor=border, textcolor=text)
    # Stage 3: feature engineering
    _box(ax, 4.5, 2.05, 1.95, 0.95, "Feature\nEngineering\n(128 feats,\nsensor-only)", facecolor=panel, edgecolor=border, textcolor=text)

    # Stage 4: three parallel model boxes
    _box(ax, 6.85, 3.35, 2.05, 0.9, "Point Predictor\n(RF)", facecolor=panel, edgecolor=border, textcolor=text, fontsize=8.5)
    _box(ax, 6.85, 2.05, 2.05, 0.9, "Upper-Quantile\n(GBR q=0.90)", facecolor=panel, edgecolor=border, textcolor=text, fontsize=8.5)
    _box(ax, 6.85, 0.75, 2.05, 0.9, "Conformal\nCalibrator (CQR)", facecolor=amber, edgecolor=amber, textcolor="#1a1a1a", fontsize=8.5, weight="bold")

    # Stage 5: scheduler decision
    _box(ax, 9.25, 2.05, 1.55, 0.95, "Scheduler\nDecision\n(argmin\nupper bound)", facecolor=panel, edgecolor=amber, textcolor=text, fontsize=8, weight="bold")
    # Stage 6: evaluation
    _box(ax, 9.25, 0.35, 2.55, 1.25,
         "Evaluation\npeak temp | violations\nmarginal & selected\ncoverage | drift",
         facecolor=panel, edgecolor=border, textcolor=text, fontsize=8)

    # Arrows
    _arrow(ax, 2.05, 2.52, 2.35, 2.52, amber)
    _arrow(ax, 4.2, 2.52, 4.5, 2.52, amber)
    _arrow(ax, 6.45, 2.7, 6.85, 3.8, amber)
    _arrow(ax, 6.45, 2.52, 6.85, 2.5, amber)
    _arrow(ax, 6.45, 2.34, 6.85, 1.2, amber)
    _arrow(ax, 8.9, 3.8, 9.25, 2.75, amber)
    _arrow(ax, 8.9, 2.5, 9.25, 2.52, amber)
    _arrow(ax, 8.9, 1.2, 9.25, 2.3, amber)
    _arrow(ax, 10.0, 2.05, 10.4, 1.6, amber)

    ax.text(7.87, 0.2, "+ load-balance fallback among near-tied safe cores",
            ha="center", color=PALETTE["muted"], fontsize=7.5)
    return fig


# --------------------------------------------------------------------------- #
# Research-summary composite (Section 7.2): coverage cards + before/after panel
# --------------------------------------------------------------------------- #
def _card(ax, x, y, w, h, title, value, *, value_color, sub=None, edge=None):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            facecolor=PALETTE["panel"],
            edgecolor=edge or PALETTE["border"],
            linewidth=1.6,
            zorder=3,
        )
    )
    ax.text(x + w / 2, y + h - 0.16, title, ha="center", va="top", color=PALETTE["muted"], fontsize=8.5, zorder=4)
    ax.text(x + w / 2, y + h * 0.42, value, ha="center", va="center", color=value_color,
            fontsize=20, fontweight="bold", family=MONO_STACK, zorder=4)
    if sub:
        ax.text(x + w / 2, y + 0.14, sub, ha="center", va="bottom", color=PALETTE["muted"], fontsize=7.5, zorder=4)


def research_summary_figure(
    *,
    nominal: float,
    id_marginal: float,
    id_selected: float,
    ood_marginal: float,
    ood_selected: float,
    before_cov: float,
    after_cov: float,
    correction_c: float,
    correction_note: str,
    figsize: tuple[float, float] = (12.0, 7.2),
    dpi: int = 100,
) -> plt.Figure:
    apply_mpl_theme()
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.2)
    ax.axis("off")

    def pct(v):
        return f"{v * 100:.1f}%"

    def cov_color(v):
        return PALETTE["green"] if v >= nominal - 0.005 else PALETTE["red"]

    fig.text(0.5, 0.965, "HeadRoom  /  Conformal coverage & calibration", ha="center",
             color=PALETTE["amber"], fontsize=14, fontweight="bold")

    # Row 1: in-distribution coverage cards
    fig.text(0.06, 0.875, "IN-DISTRIBUTION", color=PALETTE["text"], fontsize=10, fontweight="bold")
    _card(ax, 0.4, 5.25, 3.4, 1.15, "Nominal target", pct(nominal), value_color=PALETTE["muted"])
    _card(ax, 4.1, 5.25, 3.4, 1.15, "Marginal coverage", pct(id_marginal), value_color=cov_color(id_marginal))
    _card(ax, 7.8, 5.25, 3.8, 1.15, "Selected-core coverage", pct(id_selected), value_color=cov_color(id_selected))

    # Row 2: OOD coverage cards
    fig.text(0.06, 0.62, "OUT-OF-DISTRIBUTION", color=PALETTE["text"], fontsize=10, fontweight="bold")
    _card(ax, 0.4, 3.55, 3.4, 1.15, "Nominal target", pct(nominal), value_color=PALETTE["muted"])
    _card(ax, 4.1, 3.55, 3.4, 1.15, "Marginal coverage", pct(ood_marginal), value_color=cov_color(ood_marginal))
    _card(ax, 7.8, 3.55, 3.8, 1.15, "Selected-core coverage", pct(ood_selected), value_color=cov_color(ood_selected))

    # Row 3: before / after conformal correction
    fig.text(0.06, 0.37, "CONFORMAL CORRECTION (calibration set)", color=PALETTE["text"], fontsize=10, fontweight="bold")
    _card(ax, 0.4, 0.7, 4.4, 1.7, "Before conformal correction", pct(before_cov),
          value_color=cov_color(before_cov), sub="base upper-quantile model", edge=PALETTE["red"] if before_cov < nominal - 0.005 else PALETTE["border"])
    _arrow(ax, 5.0, 1.55, 6.7, 1.55, PALETTE["amber"])
    ax.text(5.85, 1.95, f"+{correction_c:.1f}C", ha="center", color=PALETTE["amber"], fontsize=12, fontweight="bold")
    _card(ax, 6.9, 0.7, 4.7, 1.7, "After conformal correction", pct(after_cov),
          value_color=cov_color(after_cov), sub="calibrated upper bound", edge=PALETTE["green"] if after_cov >= nominal - 0.005 else PALETTE["border"])

    fig.text(0.5, 0.045, correction_note, ha="center", color=PALETTE["muted"], fontsize=8.5, wrap=True)
    return fig
