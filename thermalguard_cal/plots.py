from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import ThermalGuardConfig


def make_all_plots(
    cfg: ThermalGuardConfig,
    metrics: pd.DataFrame,
    traces: pd.DataFrame,
    coverage: pd.DataFrame,
    drift: pd.DataFrame,
    representative_heatmap: np.ndarray | None = None,
    heatmap_snapshots: dict[str, np.ndarray] | None = None,
) -> None:
    figures = cfg.output_path / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    _plot_temperature_traces(cfg, traces, figures)
    _plot_peak_temperature_split(cfg, metrics, "id", figures / "peak_temperature_by_scheduler_id.png")
    _plot_peak_temperature_split(cfg, metrics, "ood", figures / "peak_temperature_by_scheduler_ood.png")
    _plot_hotspots_split(metrics, "id", figures / "hotspot_violations_id.png")
    _plot_hotspots_split(metrics, "ood", figures / "hotspot_violations_ood.png")
    _plot_coverage_id_vs_ood(cfg, coverage, figures / "coverage_id_vs_ood.png")
    _plot_selected_coverage_gap(metrics, figures / "selected_core_coverage_gap.png")
    _plot_policy_drift(metrics, figures / "policy_drift_id_vs_ood.png")
    _plot_safety_vs_throughput(cfg, metrics, "id", figures / "safety_vs_throughput_id.png")
    _plot_safety_vs_throughput(cfg, metrics, "ood", figures / "safety_vs_throughput_ood.png")
    _plot_heatmap_snapshots(cfg, heatmap_snapshots or {}, figures)

    # Backward-compatible figure names used by the initial MVP report/tests.
    _plot_peak_temperature_split(cfg, metrics, "id", figures / "peak_temperature_bar.png")
    _plot_hotspots_split(metrics, "id", figures / "hotspot_violations_bar.png")
    _plot_throughput(metrics, "id", figures / "throughput_bar.png")
    _plot_coverage_id_vs_ood(cfg, coverage, figures / "coverage_comparison.png")
    _plot_selected_coverage_gap(metrics, figures / "selected_core_coverage.png")
    _plot_policy_drift(metrics, figures / "policy_drift_comparison.png")
    _plot_safety_vs_throughput(cfg, metrics, "id", figures / "throughput_vs_thermal_safety.png")

    if representative_heatmap is not None:
        _plot_single_heatmap(
            cfg,
            representative_heatmap,
            "Representative conformal scheduler heatmap",
            figures / "representative_heatmap.png",
        )


def make_plots_from_outputs(cfg: ThermalGuardConfig, update_report: bool = True) -> None:
    reports = cfg.output_path / "reports"
    metrics = _load_metrics(reports)
    traces = _read_csv_if_exists(reports / "temperature_traces.csv")
    coverage = _read_csv_if_exists(reports / "coverage_metrics.csv")
    drift = _read_csv_if_exists(reports / "policy_drift_metrics.csv")
    heatmaps = load_heatmap_snapshots(reports / "heatmap_snapshots.npz")
    representative = heatmaps.get("conformal_upper_bound_id")
    make_all_plots(cfg, metrics, traces, coverage, drift, representative, heatmaps)
    if update_report:
        update_report_visual_results(cfg, metrics, coverage)


def load_heatmap_snapshots(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def visual_results_markdown(
    cfg: ThermalGuardConfig,
    metrics: pd.DataFrame | None = None,
    coverage: pd.DataFrame | None = None,
) -> str:
    id_cov, ood_cov = _coverage_summary(coverage if coverage is not None else pd.DataFrame())
    id_note = "near nominal" if id_cov is not None and id_cov >= cfg.conformal_target_coverage - 0.03 else "below nominal"
    ood_note = "collapses under OOD shift" if ood_cov is not None and ood_cov < cfg.conformal_target_coverage - 0.15 else "changes under OOD shift"

    return f"""## Visual Results

The figures below turn the raw CSV metrics into the main result story for the
current quick run.

![ID peak temperature by scheduler](../figures/peak_temperature_by_scheduler_id.png)

![OOD peak temperature by scheduler](../figures/peak_temperature_by_scheduler_ood.png)

Peak-temperature comparisons show the thermal safety margin against the 85 C
limit. In this quick run, several model-based schedulers avoid hotspots while
some sparse-observed baselines overheat badly under OOD conditions.

![ID hotspot violations](../figures/hotspot_violations_id.png)

![OOD hotspot violations](../figures/hotspot_violations_ood.png)

Hotspot bars make the safety failures more direct: ID is thermally easy here,
but OOD separates robust policies from brittle observed-sensor heuristics.

![Coverage ID vs OOD](../figures/coverage_id_vs_ood.png)

Conformal coverage is reported three ways: nominal target, marginal candidate
coverage, and selected-core coverage. ID selected coverage is {id_note}; OOD
coverage {ood_note}. This is why OOD calibration performance should not be
claimed as a formal safety guarantee.

![Selected-core coverage gap](../figures/selected_core_coverage_gap.png)

Selected-core coverage is measured separately because the scheduler chooses one
candidate out of 16. That selection step can create a gap from nominal coverage
even before considering policy-induced state drift.

![Policy drift ID vs OOD](../figures/policy_drift_id_vs_ood.png)

Policy drift compares rollout feature distributions against the calibration
feature distribution. It is a distinct effect from candidate selection bias and
from the deliberate OOD workload shift.

![ID safety vs throughput](../figures/safety_vs_throughput_id.png)

![OOD safety vs throughput](../figures/safety_vs_throughput_ood.png)

The safety/throughput scatter shows whether lower temperature is being bought
with lost completed work. In the current quick run, model-based schedulers avoid
hotspots without losing completed tasks, but harder presets are still needed if
ID remains too thermally easy.

![Conformal ID heatmap](../figures/heatmap_conformal_id.png)

![Conformal OOD heatmap](../figures/heatmap_conformal_ood.png)

Representative 4x4 heatmaps show the final recorded chip-temperature frame for
the conformal scheduler. The optional OOD comparison heatmap contrasts the
conformal scheduler with a bad observed-sensor baseline when available.
"""


def update_report_visual_results(cfg: ThermalGuardConfig, metrics: pd.DataFrame, coverage: pd.DataFrame) -> None:
    report_path = cfg.output_path / "reports" / "final_report.md"
    section = visual_results_markdown(cfg, metrics, coverage)
    if not report_path.exists():
        report_path.write_text(f"# ThermalGuard-Cal Final Report\n\n{section}\n", encoding="utf-8")
        return

    text = report_path.read_text(encoding="utf-8")
    start = text.find("## Visual Results")
    if start != -1:
        next_start = text.find("\n## ", start + 1)
        if next_start == -1:
            text = text[:start].rstrip() + "\n\n" + section.rstrip() + "\n"
        else:
            text = text[:start].rstrip() + "\n\n" + section.rstrip() + "\n\n" + text[next_start + 1 :].lstrip()
    else:
        marker = "## Limitations"
        marker_at = text.find(marker)
        if marker_at == -1:
            text = text.rstrip() + "\n\n" + section.rstrip() + "\n"
        else:
            text = text[:marker_at].rstrip() + "\n\n" + section.rstrip() + "\n\n" + text[marker_at:].lstrip()
    report_path.write_text(text, encoding="utf-8")


def _load_metrics(reports: Path) -> pd.DataFrame:
    frames = []
    for name in ("metrics_id.csv", "metrics_ood.csv"):
        path = reports / name
        if path.exists():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No scheduler metrics found in {reports}")
    return pd.concat(frames, ignore_index=True)


def _read_csv_if_exists(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def _plot_temperature_traces(cfg: ThermalGuardConfig, traces: pd.DataFrame, figures: Path) -> None:
    import matplotlib.pyplot as plt

    if traces.empty:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    id_traces = traces[traces["split"] == "id"]
    for scheduler, group in id_traces.groupby("scheduler"):
        by_t = group.groupby("timestep")["max_temperature"].mean()
        ax.plot(by_t.index, by_t.values, label=scheduler, linewidth=1.8)
    ax.axhline(cfg.thermal_limit, color="black", linestyle="--", linewidth=1, label="thermal limit")
    ax.set_xlabel("Timestep")
    ax.set_ylabel("Max chip temperature (C)")
    ax.set_title("ID max temperature over time by scheduler")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(figures / "max_temperature_by_scheduler.png", dpi=160)
    plt.close(fig)


def _plot_peak_temperature_split(cfg: ThermalGuardConfig, metrics: pd.DataFrame, split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = metrics[metrics["split"] == split].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    colors = _scheduler_colors(df)
    ax.bar(df["scheduler"], df["peak_temperature"], color=colors)
    ax.axhline(cfg.thermal_limit, color="black", linestyle="--", linewidth=1.2, label=f"{cfg.thermal_limit:g} C thermal limit")
    ax.set_ylabel("Peak temperature (C)")
    ax.set_title(f"{split.upper()} scheduler peak temperature")
    ax.tick_params(axis="x", rotation=35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_hotspots_split(metrics: pd.DataFrame, split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = metrics[metrics["split"] == split].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(df["scheduler"], df["hotspot_violations"], color=_scheduler_colors(df))
    ax.set_ylabel("Hotspot timestep count")
    ax.set_title(f"{split.upper()} hotspot violations")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_coverage_id_vs_ood(cfg: ThermalGuardConfig, coverage: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if coverage.empty:
        return
    rows = []
    for split in ("id", "ood"):
        split_cov = coverage[coverage["split"] == split]
        marginal = _coverage_value(split_cov, "marginal")
        selected = _coverage_value(split_cov, "selected")
        rows.append({"split": split.upper(), "coverage": "Nominal", "value": cfg.conformal_target_coverage})
        rows.append({"split": split.upper(), "coverage": "Marginal", "value": marginal})
        rows.append({"split": split.upper(), "coverage": "Selected-core", "value": selected})
    df = pd.DataFrame(rows).dropna()
    if df.empty:
        return

    labels = ["Nominal", "Marginal", "Selected-core"]
    x = np.arange(2)
    width = 0.24
    colors = {"Nominal": "#444444", "Marginal": "#2b7a78", "Selected-core": "#f2a541"}
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, label in enumerate(labels):
        values = [
            float(df[(df["split"] == split) & (df["coverage"] == label)]["value"].iloc[0])
            if not df[(df["split"] == split) & (df["coverage"] == label)].empty
            else np.nan
            for split in ("ID", "OOD")
        ]
        ax.bar(x + (idx - 1) * width, values, width=width, label=label, color=colors[label])
    ax.set_xticks(x)
    ax.set_xticklabels(["ID", "OOD"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Coverage: ID holds, OOD breaks")
    ax.legend()
    ax.text(1, 0.08, "OOD shift", ha="center", va="bottom", fontsize=9, color="#7a1f1f")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_selected_coverage_gap(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = metrics[metrics["scheduler"] == "conformal_upper_bound"].copy()
    df = df.dropna(subset=["selected_coverage_gap"])
    if df.empty:
        return
    colors = ["#2b7a78" if value >= 0 else "#b23a48" for value in df["selected_coverage_gap"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    ax.bar(df["split"].str.upper(), df["selected_coverage_gap"], color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_ylabel("Selected coverage - nominal")
    ax.set_title("Selected-core coverage gap")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_policy_drift(metrics: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    if metrics.empty or "drift_mean_abs_z" not in metrics:
        return
    schedulers = list(metrics["scheduler"].drop_duplicates())
    x = np.arange(len(schedulers))
    width = 0.38
    fig, ax = plt.subplots(figsize=(12, 5))
    for offset, split, color in ((-width / 2, "id", "#476c9b"), (width / 2, "ood", "#b56576")):
        values = []
        for scheduler in schedulers:
            row = metrics[(metrics["split"] == split) & (metrics["scheduler"] == scheduler)]
            values.append(float(row["drift_mean_abs_z"].iloc[0]) if not row.empty else np.nan)
        ax.bar(x + offset, values, width=width, label=split.upper(), color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=35, ha="right")
    ax.set_ylabel("Mean abs z-shift vs calibration")
    ax.set_title("Policy-induced drift by scheduler")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_safety_vs_throughput(cfg: ThermalGuardConfig, metrics: pd.DataFrame, split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = metrics[metrics["split"] == split].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    ax.scatter(df["completed_tasks"], df["peak_temperature"], s=70, color=_scheduler_colors(df))
    for _, row in df.iterrows():
        ax.annotate(row["scheduler"], (row["completed_tasks"], row["peak_temperature"]), fontsize=7)
    ax.axhline(cfg.thermal_limit, color="black", linestyle="--", linewidth=1, label="thermal limit")
    ax.set_xlabel("Completed tasks")
    ax.set_ylabel("Peak temperature (C)")
    ax.set_title(f"{split.upper()} safety vs throughput")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_throughput(metrics: pd.DataFrame, split: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    df = metrics[metrics["split"] == split].copy()
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(df["scheduler"], df["completed_tasks"], color=_scheduler_colors(df))
    ax.set_ylabel("Completed tasks")
    ax.set_title(f"{split.upper()} completed tasks")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_heatmap_snapshots(
    cfg: ThermalGuardConfig,
    heatmaps: dict[str, np.ndarray],
    figures: Path,
) -> None:
    if "conformal_upper_bound_id" in heatmaps:
        _plot_single_heatmap(
            cfg,
            heatmaps["conformal_upper_bound_id"],
            "Conformal scheduler ID heatmap",
            figures / "heatmap_conformal_id.png",
        )
    if "conformal_upper_bound_ood" in heatmaps:
        _plot_single_heatmap(
            cfg,
            heatmaps["conformal_upper_bound_ood"],
            "Conformal scheduler OOD heatmap",
            figures / "heatmap_conformal_ood.png",
        )
    if "coolest_core_observed_ood" in heatmaps and "conformal_upper_bound_ood" in heatmaps:
        _plot_heatmap_pair(
            cfg,
            heatmaps["coolest_core_observed_ood"],
            heatmaps["conformal_upper_bound_ood"],
            figures / "heatmap_comparison_ood.png",
        )


def _plot_single_heatmap(cfg: ThermalGuardConfig, grid: np.ndarray, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(4.8, 4))
    vmax = max(cfg.thermal_limit, float(np.nanmax(grid)))
    image = ax.imshow(grid, cmap="inferno", vmin=cfg.ambient_temp, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(cfg.grid_size))
    ax.set_yticks(range(cfg.grid_size))
    fig.colorbar(image, ax=ax, label="Temperature (C)")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_heatmap_pair(
    cfg: ThermalGuardConfig,
    baseline_grid: np.ndarray,
    conformal_grid: np.ndarray,
    path: Path,
) -> None:
    import matplotlib.pyplot as plt

    vmax = max(cfg.thermal_limit, float(np.nanmax(baseline_grid)), float(np.nanmax(conformal_grid)))
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 4))
    for ax, grid, title in (
        (axes[0], baseline_grid, "Observed coolest-core OOD"),
        (axes[1], conformal_grid, "Conformal OOD"),
    ):
        image = ax.imshow(grid, cmap="inferno", vmin=cfg.ambient_temp, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks(range(cfg.grid_size))
        ax.set_yticks(range(cfg.grid_size))
    fig.colorbar(image, ax=axes.ravel().tolist(), label="Temperature (C)")
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _scheduler_colors(df: pd.DataFrame) -> list[str]:
    colors = []
    for _, row in df.iterrows():
        if bool(row.get("uses_oracle", False)) or row.get("baseline_type") == "oracle_privileged_true_temperature":
            colors.append("#777777")
        elif row["scheduler"] == "conformal_upper_bound":
            colors.append("#2b7a78")
        elif row["scheduler"] in {"point_prediction_rf", "uncalibrated_quantile"}:
            colors.append("#476c9b")
        else:
            colors.append("#b56576")
    return colors


def _coverage_value(coverage: pd.DataFrame, token: str) -> float | None:
    if coverage.empty:
        return None
    row = coverage[coverage["coverage_type"].str.contains(token, case=False, na=False)]
    if row.empty:
        return None
    return float(row["empirical_coverage"].iloc[0])


def _coverage_summary(coverage: pd.DataFrame) -> tuple[float | None, float | None]:
    id_cov = _coverage_value(coverage[coverage["split"] == "id"], "selected") if not coverage.empty else None
    ood_cov = _coverage_value(coverage[coverage["split"] == "ood"], "selected") if not coverage.empty else None
    return id_cov, ood_cov
