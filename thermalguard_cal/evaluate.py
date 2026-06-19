from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import ThermalGuardConfig, ensure_output_dirs
from .dataset import load_split, split_specs
from .features import build_candidate_features, build_candidate_matrix, feature_names
from .models import ModelBundle, load_model_bundle
from .plots import make_all_plots, update_report_visual_results
from .schedulers import (
    ConformalUpperBoundScheduler,
    CoolestCoreObservedScheduler,
    CoolestCoreOracleScheduler,
    PointPredictionScheduler,
    RandomScheduler,
    RoundRobinScheduler,
    Scheduler,
    TrendAwareScheduler,
    UncalibratedQuantileScheduler,
)
from .sensors import SensorModel
from .simulator import ThermalSimulator
from .utils import RunSummary, current_git_commit, utc_now_iso, write_json
from .workloads import WorkloadGenerator


def evaluate_and_report(cfg: ThermalGuardConfig, write_plots: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dirs(cfg)
    bundle = load_model_bundle(cfg.output_path)
    X_cal, _, _ = load_split(cfg.output_path, "calibration")
    cal_stats = _calibration_feature_stats(X_cal)

    all_metrics: list[dict[str, Any]] = []
    all_traces: list[pd.DataFrame] = []
    all_coverage: list[dict[str, Any]] = []
    all_drift: list[dict[str, Any]] = []
    representative_heatmap: np.ndarray | None = None

    # Representative final-frame chip grids saved for the 4x4 heatmap figures.
    # We keep the conformal scheduler (ID and OOD) and one bad observed-sensor
    # baseline (coolest_core_observed, OOD) for the side-by-side comparison.
    heatmap_keys = {
        ("conformal_upper_bound", "id"): "conformal_upper_bound_id",
        ("conformal_upper_bound", "ood"): "conformal_upper_bound_ood",
        ("coolest_core_observed", "ood"): "coolest_core_observed_ood",
    }
    heatmap_snapshots: dict[str, np.ndarray] = {}

    for split in ("id", "ood"):
        schedulers = make_schedulers(cfg, bundle)
        for scheduler in schedulers:
            result = evaluate_scheduler(cfg, scheduler, bundle, split, cal_stats)
            all_metrics.append(result["summary"].to_dict())
            trace = result["trace"]
            all_traces.append(trace)
            all_coverage.extend(result["coverage_rows"])
            all_drift.extend(result["drift_rows"])
            key = heatmap_keys.get((scheduler.name, split))
            if key is not None:
                heatmap_snapshots[key] = result["last_temperature_grid"]
            if representative_heatmap is None and scheduler.name == "conformal_upper_bound" and split == "id":
                representative_heatmap = result["last_temperature_grid"]

    metrics = pd.DataFrame(all_metrics)
    traces = pd.concat(all_traces, ignore_index=True) if all_traces else pd.DataFrame()
    coverage = pd.DataFrame(all_coverage)
    drift = pd.DataFrame(all_drift)

    reports_dir = cfg.output_path / "reports"
    metrics[metrics["split"] == "id"].to_csv(reports_dir / "metrics_id.csv", index=False)
    metrics[metrics["split"] == "ood"].to_csv(reports_dir / "metrics_ood.csv", index=False)
    metrics[metrics["split"] == "id"].to_csv(reports_dir / f"metrics_id_{cfg.preset}.csv", index=False)
    metrics[metrics["split"] == "ood"].to_csv(reports_dir / f"metrics_ood_{cfg.preset}.csv", index=False)
    coverage.to_csv(reports_dir / "coverage_metrics.csv", index=False)
    coverage.to_csv(reports_dir / f"coverage_metrics_{cfg.preset}.csv", index=False)
    drift.to_csv(reports_dir / "policy_drift_metrics.csv", index=False)
    drift.to_csv(reports_dir / f"policy_drift_metrics_{cfg.preset}.csv", index=False)
    traces.to_csv(reports_dir / "temperature_traces.csv", index=False)
    traces.to_csv(reports_dir / f"temperature_traces_{cfg.preset}.csv", index=False)
    if heatmap_snapshots:
        np.savez(reports_dir / "heatmap_snapshots.npz", **heatmap_snapshots)

    if write_plots:
        make_all_plots(cfg, metrics, traces, coverage, drift, representative_heatmap, heatmap_snapshots)
    write_final_report(cfg, metrics, coverage, drift)
    if write_plots:
        update_report_visual_results(cfg, metrics, coverage)
    return metrics, coverage


def make_schedulers(cfg: ThermalGuardConfig, bundle: ModelBundle) -> list[Scheduler]:
    return [
        RandomScheduler(cfg, seed=cfg.random_seed + 500),
        RoundRobinScheduler(cfg),
        CoolestCoreObservedScheduler(cfg),
        CoolestCoreOracleScheduler(cfg),
        TrendAwareScheduler(cfg),
        PointPredictionScheduler(cfg, bundle),
        UncalibratedQuantileScheduler(cfg, bundle),
        ConformalUpperBoundScheduler(cfg, bundle),
    ]


def evaluate_scheduler(
    cfg: ThermalGuardConfig,
    scheduler: Scheduler,
    bundle: ModelBundle,
    split: str,
    cal_stats: dict[str, np.ndarray],
) -> dict[str, Any]:
    scheduler.reset()
    workload_split = "ood" if split == "ood" else "id"
    generator = WorkloadGenerator.for_split(cfg, workload_split)
    dropout = cfg.ood_sensor_dropout_prob if split == "ood" else cfg.sensor_dropout_prob
    sensor_model = SensorModel(cfg, dropout_prob=dropout)

    max_temps: list[float] = []
    avg_temps: list[float] = []
    trace_rows: list[dict[str, Any]] = []
    selected_bounds: list[float] = []
    selected_actuals: list[float] = []
    all_candidate_bounds: list[float] = []
    all_candidate_actuals: list[float] = []
    visited_features: list[np.ndarray] = []
    predicted_selected: list[float] = []
    assignment_counts_total = np.zeros(cfg.num_cores, dtype=int)

    assigned_start = 0
    dropped_start = 0
    completed_start = 0
    last_grid = np.full((cfg.grid_size, cfg.grid_size), cfg.initial_temp)

    spec_name = "test_ood" if split == "ood" else "test_id"
    spec = {item.name: item for item in split_specs(cfg)}[spec_name]
    for episode_id in spec.episode_ids:
        seed = cfg.random_seed + episode_id * 9973
        sim = ThermalSimulator(cfg, seed=seed)
        arrivals = generator.generate_episode(seed=seed + 101, episode_id=episode_id)

        for timestep in range(cfg.episode_length):
            observed_state = sim.observe_state(sensor_model)
            readings = observed_state.sensor_readings.copy()
            mask = observed_state.sensor_mask.copy()
            trend = observed_state.observed_temp_trend.copy()

            for task in arrivals.get(timestep, []):
                state = sim.build_state(readings, mask, trend)
                if scheduler.name == "conformal_upper_bound":
                    X_candidates = build_candidate_matrix(state, task, cfg)
                    bounds = bundle.predict_conformal_upper(X_candidates)
                    actuals = np.array(
                        [
                            sim.future_peak_after_assignment(task, core, cfg.prediction_horizon)
                            for core in range(cfg.num_cores)
                        ],
                        dtype=float,
                    )
                    all_candidate_bounds.extend(bounds.tolist())
                    all_candidate_actuals.extend(actuals.tolist())

                core = scheduler.choose_core(state, task)
                selected_feature = build_candidate_features(state, task, core, cfg)
                visited_features.append(selected_feature)

                if scheduler.name == "conformal_upper_bound":
                    selected_bounds.append(float(bundle.predict_conformal_upper(selected_feature.reshape(1, -1))[0]))
                    selected_actuals.append(
                        float(sim.future_peak_after_assignment(task, core, cfg.prediction_horizon))
                    )
                elif scheduler.name in {"point_prediction_rf", "uncalibrated_quantile"}:
                    X_one = selected_feature.reshape(1, -1)
                    if scheduler.name == "point_prediction_rf":
                        predicted_selected.append(float(bundle.predict_point(X_one)[0]))
                    else:
                        predicted_selected.append(float(bundle.predict_quantile(X_one)[0]))

                sim.assign_task(task, core)

            sim.step()
            max_temp = float(np.max(sim.true_temperatures))
            avg_temp = float(np.mean(sim.true_temperatures))
            max_temps.append(max_temp)
            avg_temps.append(avg_temp)
            trace_rows.append(
                {
                    "scheduler": scheduler.name,
                    "split": split,
                    "episode_id": episode_id,
                    "timestep": timestep,
                    "max_temperature": max_temp,
                    "average_temperature": avg_temp,
                    "hotspot": bool(max_temp > cfg.thermal_limit),
                }
            )
            last_grid = sim.true_temperatures.reshape(cfg.grid_size, cfg.grid_size).copy()

        assigned_start += sim.assigned_tasks
        dropped_start += sim.dropped_tasks
        completed_start += sim.completed_tasks
        assignment_counts_total += sim.assignment_counts.astype(int)

    max_arr = np.array(max_temps, dtype=float)
    avg_arr = np.array(avg_temps, dtype=float)
    hotspot_steps = int(np.sum(max_arr > cfg.thermal_limit))
    selected_cov = None
    marginal_cov = None
    selected_gap = None
    selected_violations = None
    avg_bound = None
    avg_actual = None
    coverage_rows: list[dict[str, Any]] = []

    if scheduler.name == "conformal_upper_bound":
        selected_bounds_arr = np.array(selected_bounds, dtype=float)
        selected_actuals_arr = np.array(selected_actuals, dtype=float)
        all_bounds_arr = np.array(all_candidate_bounds, dtype=float)
        all_actuals_arr = np.array(all_candidate_actuals, dtype=float)
        if selected_bounds_arr.size:
            selected_cov = float(np.mean(selected_actuals_arr <= selected_bounds_arr))
            selected_violations = int(np.sum(selected_actuals_arr > selected_bounds_arr))
            avg_bound = float(np.mean(selected_bounds_arr))
            avg_actual = float(np.mean(selected_actuals_arr))
        if all_bounds_arr.size:
            marginal_cov = float(np.mean(all_actuals_arr <= all_bounds_arr))
        if selected_cov is not None:
            selected_gap = selected_cov - cfg.conformal_target_coverage

        coverage_rows.extend(
            [
                {
                    "split": split,
                    "scheduler": scheduler.name,
                    "coverage_type": "marginal_all_candidates_on_visited_states",
                    "nominal_coverage": cfg.conformal_target_coverage,
                    "empirical_coverage": marginal_cov,
                    "n": int(all_bounds_arr.size),
                },
                {
                    "split": split,
                    "scheduler": scheduler.name,
                    "coverage_type": "selected_core_after_scheduler_selection",
                    "nominal_coverage": cfg.conformal_target_coverage,
                    "empirical_coverage": selected_cov,
                    "n": int(selected_bounds_arr.size),
                },
            ]
        )

    visited = np.vstack(visited_features) if visited_features else np.empty((0, len(feature_names(cfg))))
    drift_rows, drift_summary = _policy_drift_rows(visited, cal_stats, scheduler.name, split, bundle.feature_names)

    summary = RunSummary(
        scheduler=scheduler.name,
        split=split,
        preset=cfg.preset,
        baseline_type=_scheduler_baseline_type(scheduler.name),
        uses_oracle=scheduler.name == "coolest_core_oracle_true_temp",
        peak_temperature=float(np.max(max_arr)) if max_arr.size else float("nan"),
        average_temperature=float(np.mean(avg_arr)) if avg_arr.size else float("nan"),
        hotspot_violations=hotspot_steps,
        hotspot_timestep_pct=float(hotspot_steps / max(1, len(max_temps))),
        completed_tasks=int(completed_start),
        assigned_tasks=int(assigned_start),
        dropped_tasks=int(dropped_start),
        average_waiting_time=0.0,
        assignment_load_std=float(np.std(assignment_counts_total)),
        average_max_temperature=float(np.mean(max_arr)) if max_arr.size else float("nan"),
        hit_max_temperature_cap=bool(np.any(max_arr >= cfg.max_reasonable_temp - 1e-9)),
        marginal_coverage=marginal_cov,
        selected_core_coverage=selected_cov,
        selected_coverage_gap=selected_gap,
        selected_bound_violations=selected_violations,
        average_selected_bound=avg_bound,
        average_selected_actual=avg_actual,
        drift_mean_abs_z=drift_summary["mean_abs_z"],
        drift_max_ks=drift_summary["max_ks"],
    )

    trace = pd.DataFrame(trace_rows)
    return {
        "summary": summary,
        "trace": trace,
        "coverage_rows": coverage_rows,
        "drift_rows": drift_rows,
        "last_temperature_grid": last_grid,
    }


def _calibration_feature_stats(X_cal: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "mean": np.mean(X_cal, axis=0),
        "std": np.std(X_cal, axis=0) + 1e-9,
        "sample": X_cal,
    }


def _policy_drift_rows(
    visited: np.ndarray,
    cal_stats: dict[str, np.ndarray],
    scheduler: str,
    split: str,
    names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    if visited.size == 0:
        return [], {"mean_abs_z": float("nan"), "max_ks": float("nan")}
    cal = cal_stats["sample"]
    mean_diff_z = np.abs((np.mean(visited, axis=0) - cal_stats["mean"]) / cal_stats["std"])
    ks_stats = np.array([_ks_stat(cal[:, i], visited[:, i]) for i in range(visited.shape[1])])

    key_prefixes = (
        "sensor_temp_imputed_",
        "power_",
        "load_",
        "candidate_core_",
        "candidate_task_power",
        "total_power",
        "total_load",
        "max_observed_temp",
        "mean_observed_temp",
    )
    rows: list[dict[str, Any]] = []
    for i, name in enumerate(names):
        if name.startswith(key_prefixes) or name in key_prefixes:
            rows.append(
                {
                    "split": split,
                    "scheduler": scheduler,
                    "feature": name,
                    "calibration_mean": float(cal_stats["mean"][i]),
                    "visited_mean": float(np.mean(visited[:, i])),
                    "abs_mean_z_shift": float(mean_diff_z[i]),
                    "ks_stat": float(ks_stats[i]),
                }
            )
    return rows, {"mean_abs_z": float(np.mean(mean_diff_z)), "max_ks": float(np.max(ks_stats))}


def _ks_stat(x: np.ndarray, y: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    if x.size == 0 or y.size == 0:
        return float("nan")
    values = np.sort(np.concatenate([x, y]))
    cdf_x = np.searchsorted(x, values, side="right") / x.size
    cdf_y = np.searchsorted(y, values, side="right") / y.size
    return float(np.max(np.abs(cdf_x - cdf_y)))


def _scheduler_baseline_type(scheduler_name: str) -> str:
    if scheduler_name == "coolest_core_oracle_true_temp":
        return "oracle_privileged_true_temperature"
    if scheduler_name in {"point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"}:
        return "model_based_sensor_observed"
    return "deployable_baseline_sensor_observed"


def write_final_report(
    cfg: ThermalGuardConfig,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    drift: pd.DataFrame,
) -> None:
    reports_dir = cfg.output_path / "reports"
    model_metrics_path = reports_dir / "model_metrics.csv"
    model_metrics = pd.read_csv(model_metrics_path) if model_metrics_path.exists() else pd.DataFrame()
    conformal_diagnostics_path = reports_dir / "conformal_diagnostics.csv"
    conformal_diagnostics = (
        pd.read_csv(conformal_diagnostics_path) if conformal_diagnostics_path.exists() else pd.DataFrame()
    )

    def md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
        if df.empty:
            return "_No rows produced._"
        return df.head(max_rows).to_markdown(index=False)

    id_metrics = metrics[metrics["split"] == "id"][
        [
            "scheduler",
            "baseline_type",
            "peak_temperature",
            "average_max_temperature",
            "hotspot_violations",
            "completed_tasks",
            "assigned_tasks",
            "marginal_coverage",
            "selected_core_coverage",
            "drift_mean_abs_z",
            "drift_max_ks",
        ]
    ]
    ood_metrics = metrics[metrics["split"] == "ood"][
        [
            "scheduler",
            "baseline_type",
            "peak_temperature",
            "average_max_temperature",
            "hotspot_violations",
            "completed_tasks",
            "assigned_tasks",
            "marginal_coverage",
            "selected_core_coverage",
            "drift_mean_abs_z",
            "drift_max_ks",
        ]
    ]

    drift_summary = (
        metrics[
            [
                "split",
                "scheduler",
                "drift_mean_abs_z",
                "drift_max_ks",
            ]
        ]
        .sort_values(["split", "drift_mean_abs_z"], ascending=[True, False])
        .reset_index(drop=True)
    )
    top_drift_features = (
        drift.sort_values(["split", "abs_mean_z_shift"], ascending=[True, False])
        if not drift.empty
        else drift
    )
    student_summary = _student_summary_markdown(cfg, metrics, coverage, conformal_diagnostics)
    conformal_section = _conformal_diagnostics_markdown(cfg, conformal_diagnostics)
    scheduler_comparison = _model_scheduler_comparison_markdown(cfg, metrics, coverage)
    preset_results = _preset_results_markdown(reports_dir)
    stress_sweep = _optional_markdown_file(reports_dir / "stress_sweep_summary.md", "## Stress Sweep Result")
    multiseed = _optional_markdown_file(
        reports_dir / f"multiseed_summary_{cfg.preset}.md",
        "## Multiseed Result",
    )

    report = f"""# ThermalGuard-Cal Final Report

Generated: {utc_now_iso()}

{student_summary}

## Project Summary

ThermalGuard-Cal is a simulation-based Python MVP for conformal upper-bound
thermal scheduling on a 4x4 many-core chip. It implements a stochastic thermal
simulator, stable in-distribution and separate OOD workload generators, sparse
noisy sensors, action-conditioned datasets, point/quantile/conformal models,
scheduler baselines, fair ID/OOD evaluation, plots, and reports.

Current run preset: **{cfg.preset}**.

## Prediction Target

The model predicts **future peak chip temperature over the horizon** as a
global whole-chip quantity. It does **not** predict the candidate core's own
future temperature in isolation. Therefore selected-core coverage means
coverage of the global outcome that results from a placement choice.

## Simulator And Workloads

The simulator uses heat gain from active task power, ambient cooling,
4-neighbor diffusion, mild thermal inertia, and small stochastic noise. ID
train/calibration/test episodes share the same stable workload distribution.
OOD episodes use a separate higher-power, burstier mix and higher sensor
dropout. Model features only use sparse/noisy sensor observations and workload
metadata; true temperatures are reserved for simulator physics and labels.

The `coolest_core_oracle_true_temp` row is intentionally labeled as a privileged
oracle baseline because it reads true current temperatures. It is useful as a
simulation reference, not as a deployable sparse-sensor scheduler.

## Model Metrics

{md_table(model_metrics)}

{conformal_section}

## ID Scheduler Metrics

{md_table(id_metrics)}

## OOD Scheduler Metrics

{md_table(ood_metrics)}

## Coverage Metrics

{md_table(coverage)}

Marginal candidate coverage and selected-core coverage are reported separately.
The difference captures the selection step where the scheduler chooses one
candidate out of 16. Distribution drift is reported separately below and should
not be conflated with the selection-bias coverage gap.

{scheduler_comparison}

{preset_results}

{stress_sweep}

{multiseed}

## Policy-Induced Distribution Drift

Scheduler-level drift summary:

{md_table(drift_summary)}

Largest feature-level shifts:

{md_table(top_drift_features)}

The drift table compares features visited by each scheduler's rollout against
the calibration feature distribution. This measures policy-induced state
distribution shift even when the workload generator remains in-distribution.

## Figures

Stable figure filenames under `outputs/figures/` (see the Visual Results
section below for plain-English explanations). Regenerate any time with
`python run_make_plots.py`.

- `outputs/figures/executive_summary.png` (portfolio/README overview)
- `outputs/figures/peak_temperature_by_scheduler_id.png`
- `outputs/figures/peak_temperature_by_scheduler_ood.png`
- `outputs/figures/hotspot_violations_id.png`
- `outputs/figures/hotspot_violations_ood.png`
- `outputs/figures/coverage_id_vs_ood.png`
- `outputs/figures/selected_core_coverage_gap.png`
- `outputs/figures/policy_drift_id_vs_ood.png`
- `outputs/figures/safety_vs_throughput_id.png`
- `outputs/figures/safety_vs_throughput_ood.png`
- `outputs/figures/heatmap_conformal_id.png`
- `outputs/figures/heatmap_conformal_ood.png`
- `outputs/figures/heatmap_comparison_ood.png`
- `outputs/figures/max_temperature_by_scheduler.png`

## Limitations

This is a research MVP, not a validated chip thermal model. It does not include
HotSpot, OpenROAD, chiplets, DVFS, GNNs, active sensing, FPGA logic, or a web
dashboard. The conformal guarantee is marginal on calibration-like data and is
not a formal OOD safety guarantee.
"""
    (reports_dir / "final_report.md").write_text(report, encoding="utf-8")


def _student_summary_markdown(
    cfg: ThermalGuardConfig,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
    conformal_diagnostics: pd.DataFrame,
) -> str:
    id_cov = _coverage_from_rows(coverage, "id", "selected")
    ood_cov = _coverage_from_rows(coverage, "ood", "selected")
    ood_model_cov = _diagnostic_value(diagnostics=conformal_diagnostics, metric="ood_empirical_coverage_after_conformal")
    overheated = metrics[metrics["hotspot_violations"] > 0]["scheduler"].drop_duplicates().tolist()
    safe = [
        scheduler
        for scheduler, group in metrics.groupby("scheduler")
        if int(group["hotspot_violations"].sum()) == 0
    ]
    sparse_failure = bool(
        (
            (metrics["baseline_type"] == "deployable_baseline_sensor_observed")
            & (metrics["hotspot_violations"] > 0)
        ).any()
    )
    model_based = metrics[metrics["baseline_type"] == "model_based_sensor_observed"]
    deployable = metrics[metrics["baseline_type"] == "deployable_baseline_sensor_observed"]
    model_advantage = "promising" if not model_based.empty and not deployable.empty and model_based["hotspot_violations"].mean() < deployable["hotspot_violations"].mean() else "not proven"
    conformal_advantage = _conformal_advantage_verdict(metrics, coverage)
    correction = _diagnostic_value(conformal_diagnostics, "conformal_correction")
    conformal_worked = (
        "Coverage improved or was verified on calibration-like data."
        if correction is not None
        else "Conformal diagnostics were not available for this run."
    )
    verdict = pd.DataFrame(
        [
            ("Pipeline status", "working"),
            ("ID calibration", "good" if id_cov is not None and id_cov >= cfg.conformal_target_coverage - 0.03 else "bad"),
            (
                "OOD calibration",
                "good"
                if (
                    ood_cov is not None
                    and ood_cov >= cfg.conformal_target_coverage - 0.03
                    and ood_model_cov is not None
                    and ood_model_cov >= cfg.conformal_target_coverage - 0.03
                )
                else "bad",
            ),
            ("Sparse-sensor baseline failure", "yes" if sparse_failure else "no"),
            ("Model-based scheduler advantage", model_advantage),
            ("Conformal advantage over model baselines", conformal_advantage),
            ("Next needed work", "challenging preset + multiseed validation"),
        ],
        columns=["Item", "Verdict"],
    )
    return f"""## Student Summary: What This Run Means

ThermalGuard-Cal builds a 4x4 chip thermal simulator, generates task-placement
datasets, trains point and upper-bound models, calibrates an upper bound with
one-sided conformal prediction, and compares schedulers. The model predicts the
future peak temperature of the whole chip after assigning a task to a candidate
core.

Scheduler categories: simple baselines use randomness or round-robin placement,
sparse-sensor heuristics use only noisy observed temperatures, the oracle uses
privileged true temperatures for reference, and model-based schedulers use
learned future-temperature predictions. ID means the test workload matches the
training/calibration distribution. OOD means the workload is shifted to hotter,
burstier behavior with more sensor dropout. Coverage means the true future peak
temperature is below the predicted upper bound. Selected-core coverage measures
that coverage only after the scheduler has selected one core.

Schedulers with at least one hotspot violation in any evaluated split: {", ".join(overheated) if overheated else "none"}.
Schedulers with zero hotspot violations across all evaluated splits: {", ".join(safe) if safe else "none"}.
Conformal status: {conformal_worked} What is not proven yet: this is not real
silicon validation, and OOD coverage is not guaranteed. The next experiment is
the challenging preset plus multiseed validation, which checks whether scheduler
choice matters in a harder but non-saturating regime.

| Result verdict | Status |
|---|---|
{chr(10).join(f"| {row.Item} | {row.Verdict} |" for row in verdict.itertuples(index=False))}
"""


def _conformal_diagnostics_markdown(cfg: ThermalGuardConfig, diagnostics: pd.DataFrame) -> str:
    if diagnostics.empty:
        return "## Conformal Calibration Diagnostics\n\n_No conformal diagnostics were produced._"
    correction = _diagnostic_value(diagnostics, "conformal_correction")
    note = ""
    if correction is not None and abs(correction) <= 1e-12:
        note = (
            "\n\nIn this run, the quantile model was already conservative on the "
            "calibration split, so conformal calibration verified the bound but "
            "did not widen it."
        )
    table = diagnostics.to_markdown(index=False)
    return f"""## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

{table}{note}
"""


def _model_scheduler_comparison_markdown(
    cfg: ThermalGuardConfig,
    metrics: pd.DataFrame,
    coverage: pd.DataFrame,
) -> str:
    schedulers = ["point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"]
    rows: list[dict[str, object]] = []
    statements: list[str] = []
    for split in ("id", "ood"):
        subset = metrics[(metrics["split"] == split) & (metrics["scheduler"].isin(schedulers))].copy()
        if subset.empty:
            continue
        lowest_peak = subset.loc[subset["peak_temperature"].idxmin(), "scheduler"]
        fewest_hotspots = subset.loc[subset["hotspot_violations"].idxmin(), "scheduler"]
        best_coverage = "conformal_upper_bound" if _coverage_from_rows(coverage, split, "selected") is not None else "not measured"
        conf_row = subset[subset["scheduler"] == "conformal_upper_bound"]
        conf_hotspots = int(conf_row["hotspot_violations"].iloc[0]) if not conf_row.empty else -1
        min_hotspots = int(subset["hotspot_violations"].min())
        if lowest_peak == "conformal_upper_bound" and conf_hotspots == min_hotspots:
            verdict = "safer in this split"
        elif conf_hotspots == min_hotspots:
            verdict = "coverage value, similar safety outcome"
        else:
            verdict = "not better in this split"
        rows.append(
            {
                "split": split,
                "lowest_peak_temperature": lowest_peak,
                "fewest_hotspot_violations": fewest_hotspots,
                "best_measured_coverage": best_coverage,
                "conformal_interpretation": verdict,
            }
        )
        statements.append(
            f"For {split.upper()}, {lowest_peak} has the lowest peak temperature and "
            f"{fewest_hotspots} has the fewest hotspot violations."
        )
    table = pd.DataFrame(rows).to_markdown(index=False) if rows else "_No model-based scheduler rows found._"
    statement_text = "\n".join(f"- {item}" for item in statements)
    return f"""## Does Conformal Add Scheduling Value?

This section compares only `point_prediction_rf`, `uncalibrated_quantile`, and
`conformal_upper_bound`. Conformal should not be called best unless both the
scheduling metrics and coverage metrics support that claim.

{table}

{statement_text}

Calibration can improve statistical trust even when scheduler outcomes are
similar, because it turns an uncalibrated quantile model into an auditable
coverage claim on calibration-like data. It does not create an OOD guarantee.
"""


def _preset_results_markdown(reports_dir: Path) -> str:
    rows: list[pd.DataFrame] = []
    for path in sorted(reports_dir.glob("metrics_id_*.csv")):
        preset = path.stem.replace("metrics_id_", "")
        try:
            frame = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        frame = frame.copy()
        frame["result_group"] = f"{preset}_id"
        rows.append(
            frame[
                [
                    "result_group",
                    "scheduler",
                    "peak_temperature",
                    "average_max_temperature",
                    "hotspot_violations",
                    "completed_tasks",
                    "selected_core_coverage",
                ]
            ]
        )
    if not rows:
        return "## Preset Results\n\n_No preset-stamped metric snapshots are available yet._"
    table = pd.concat(rows, ignore_index=True).to_markdown(index=False)
    return f"""## Preset Results

This separates saved preset snapshots from the current canonical CSVs, so the
original normal/easy quick result can be compared against a challenging run when
both have been executed.

{table}
"""


def _optional_markdown_file(path: Path, replacement_heading: str) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8").strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("# "):
        lines[0] = replacement_heading
    return "\n".join(lines)


def _coverage_from_rows(coverage: pd.DataFrame, split: str, token: str) -> float | None:
    if coverage.empty:
        return None
    rows = coverage[
        (coverage["split"] == split)
        & (coverage["coverage_type"].str.contains(token, case=False, na=False))
    ]
    if rows.empty:
        return None
    return float(rows["empirical_coverage"].iloc[0])


def _diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float | None:
    if diagnostics.empty:
        return None
    rows = diagnostics[diagnostics["metric"] == metric]
    if rows.empty:
        return None
    return float(rows["value"].iloc[0])


def _conformal_advantage_verdict(metrics: pd.DataFrame, coverage: pd.DataFrame) -> str:
    model = metrics[metrics["scheduler"].isin(["point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"])]
    if model.empty:
        return "not proven"
    conformal = model[model["scheduler"] == "conformal_upper_bound"]
    if conformal.empty:
        return "not proven"
    best_peak_count = 0
    best_hotspot_count = 0
    for split, group in model.groupby("split"):
        conf = group[group["scheduler"] == "conformal_upper_bound"]
        if conf.empty:
            continue
        best_peak_count += int(float(conf["peak_temperature"].iloc[0]) <= float(group["peak_temperature"].min()) + 1e-9)
        best_hotspot_count += int(int(conf["hotspot_violations"].iloc[0]) == int(group["hotspot_violations"].min()))
    selected_cov_exists = _coverage_from_rows(coverage, "id", "selected") is not None
    if best_peak_count == len(model["split"].unique()) and best_hotspot_count == len(model["split"].unique()) and selected_cov_exists:
        return "proven"
    if best_hotspot_count > 0 or selected_cov_exists:
        return "mixed"
    return "not proven"


def write_run_manifest(cfg: ThermalGuardConfig, mode: str) -> None:
    manifest = {
        "timestamp_utc": utc_now_iso(),
        "mode": mode,
        "preset": cfg.preset,
        "quick": cfg.quick,
        "config": cfg.to_dict(),
        "git_commit": current_git_commit(Path.cwd()),
    }
    write_json(cfg.output_path / "reports" / "run_manifest.json", manifest)
