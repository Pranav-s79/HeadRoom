from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from thermalguard_cal.config import PRESET_OVERRIDES, ThermalGuardConfig, make_config
from thermalguard_cal.features import build_candidate_matrix
from thermalguard_cal.models import ModelBundle, load_model_bundle
from thermalguard_cal.schedulers import (
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
from thermalguard_cal.sensors import SensorModel
from thermalguard_cal.simulator import ThermalSimulator
from thermalguard_cal.workloads import WorkloadGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_SCHEDULERS = {
    "point_prediction_rf",
    "uncalibrated_quantile",
    "conformal_upper_bound",
}
SCHEDULER_ORDER = [
    "random",
    "round_robin",
    "coolest_core_observed",
    "coolest_core_oracle_true_temp",
    "trend_aware_observed",
    "point_prediction_rf",
    "uncalibrated_quantile",
    "conformal_upper_bound",
]


def main() -> None:
    st.set_page_config(page_title="ThermalGuard-Cal", layout="wide")
    st.title("ThermalGuard-Cal")

    output_dir = Path(
        st.sidebar.text_input(
            "Output directory",
            value=str(DEFAULT_OUTPUT_DIR),
        )
    )
    reports_dir = output_dir / "reports"
    figures_dir = output_dir / "figures"

    page = st.sidebar.radio(
        "View",
        [
            "Simulation Replay",
            "Results Explorer",
            "Calibration View",
            "Scheduler Comparison",
            "Heatmap Comparison",
            "Stress and Multiseed",
        ],
    )

    if page == "Simulation Replay":
        render_simulation_replay(output_dir)
    elif page == "Results Explorer":
        render_results_explorer(reports_dir, figures_dir)
    elif page == "Calibration View":
        render_calibration_view(reports_dir)
    elif page == "Scheduler Comparison":
        render_scheduler_comparison(reports_dir)
    elif page == "Heatmap Comparison":
        render_heatmap_comparison(reports_dir, figures_dir)
    else:
        render_stress_multiseed(reports_dir, figures_dir)


def render_simulation_replay(output_dir: Path) -> None:
    bundle = load_bundle_or_none(output_dir)
    has_model = bundle is not None
    available_schedulers = [
        scheduler for scheduler in SCHEDULER_ORDER if has_model or scheduler not in MODEL_SCHEDULERS
    ]

    controls = st.sidebar.container()
    preset = controls.selectbox("Preset", sorted(PRESET_OVERRIDES), index=sorted(PRESET_OVERRIDES).index("challenging"))
    split = controls.selectbox("Workload", ["id", "ood"], index=0)
    scheduler_name = controls.selectbox(
        "Scheduler",
        available_schedulers,
        index=available_schedulers.index("conformal_upper_bound") if "conformal_upper_bound" in available_schedulers else 0,
    )
    seed = controls.number_input("Seed", min_value=0, value=17, step=1)
    episode_length = controls.slider("Replay timesteps", min_value=30, max_value=180, value=90, step=10)

    if not has_model:
        st.warning(
            f"No model bundle found at `{output_dir / 'models' / 'model_bundle.joblib'}`. "
            "Model-based schedulers and risk heatmaps are disabled."
        )

    replay = build_replay(
        output_dir=str(output_dir),
        preset=preset,
        split=split,
        scheduler_name=scheduler_name,
        seed=int(seed),
        episode_length=int(episode_length),
    )
    if not replay:
        st.error("Replay produced no frames.")
        return

    frames = pd.DataFrame(replay)
    timestep = st.slider(
        "Timestep",
        min_value=int(frames["timestep"].min()),
        max_value=int(frames["timestep"].max()),
        value=int(frames["timestep"].min()),
        step=1,
    )
    frame = dict(frames[frames["timestep"] == timestep].iloc[0])
    cfg = make_config("quick", preset=preset, random_seed=int(seed), episode_length=int(episode_length))

    metric_cols = st.columns(5)
    metric_cols[0].metric("Max temp", f"{frame['max_temperature']:.1f} C")
    metric_cols[1].metric("Avg temp", f"{frame['average_temperature']:.1f} C")
    metric_cols[2].metric("Active tasks", int(frame["active_tasks"]))
    metric_cols[3].metric("Assigned", int(frame["assigned_tasks"]))
    selected = "none" if pd.isna(frame["selected_core"]) else int(frame["selected_core"])
    metric_cols[4].metric("Selected core", selected)

    top_cols = st.columns([1.1, 1.1, 1.2])
    with top_cols[0]:
        st.subheader("Chip Temperature")
        st.pyplot(
            plot_grid(
                np.array(frame["temperatures"], dtype=float),
                cfg,
                title="True temperature",
                selected_core=None if pd.isna(frame["selected_core"]) else int(frame["selected_core"]),
                sensor_mask=np.array(frame["sensor_mask"], dtype=bool),
                cmap="inferno",
                vmin=cfg.ambient_temp,
                vmax=max(cfg.thermal_limit, float(frames["max_temperature"].max())),
            )
        )
    with top_cols[1]:
        st.subheader("Sensor Readings")
        st.pyplot(
            plot_grid(
                np.array(frame["sensor_readings"], dtype=float),
                cfg,
                title="Sparse observed sensors",
                selected_core=None if pd.isna(frame["selected_core"]) else int(frame["selected_core"]),
                sensor_mask=np.array(frame["sensor_mask"], dtype=bool),
                cmap="viridis",
                vmin=cfg.ambient_temp,
                vmax=max(cfg.thermal_limit, float(frames["max_temperature"].max())),
                missing_value=cfg.ambient_temp,
            )
        )
    with top_cols[2]:
        st.subheader("Decision")
        if pd.isna(frame["selected_core"]):
            st.info("No task arrival at this timestep.")
        else:
            st.write(
                {
                    "scheduler": scheduler_name,
                    "task_power": round(float(frame["task_power"]), 3),
                    "task_duration": int(frame["task_duration"]),
                    "selected_core": int(frame["selected_core"]),
                    "candidate_count": cfg.num_cores,
                }
            )

    risk_cols = st.columns(3)
    for column, label, key, cmap in (
        (risk_cols[0], "Point Prediction", "point_scores", "magma"),
        (risk_cols[1], "Quantile Upper", "quantile_scores", "plasma"),
        (risk_cols[2], "Conformal Upper", "conformal_scores", "cividis"),
    ):
        with column:
            st.subheader(label)
            scores = frame[key]
            if scores is None or len(scores) == 0:
                st.info("No model score for this frame.")
            else:
                score_arr = np.array(scores, dtype=float)
                st.pyplot(
                    plot_grid(
                        score_arr,
                        cfg,
                        title=label,
                        selected_core=None if pd.isna(frame["selected_core"]) else int(frame["selected_core"]),
                        cmap=cmap,
                        vmin=float(np.nanmin(score_arr)),
                        vmax=float(np.nanmax(score_arr)),
                    )
                )

    peak = frames[["timestep", "max_temperature"]].set_index("timestep")
    st.subheader("Peak Temperature Over Time")
    st.line_chart(peak)

    if st.button("Save replay log"):
        path = write_replay_log(output_dir, frames, preset, split, scheduler_name, int(seed))
        st.success(f"Replay log written to `{path}`")


@st.cache_data(show_spinner=False)
def build_replay(
    output_dir: str,
    preset: str,
    split: str,
    scheduler_name: str,
    seed: int,
    episode_length: int,
) -> list[dict[str, Any]]:
    cfg = make_config("quick", preset=preset, random_seed=seed, episode_length=episode_length)
    bundle = load_bundle_or_none(Path(output_dir))
    scheduler = make_scheduler(scheduler_name, cfg, bundle)
    generator = WorkloadGenerator.for_split(cfg, "ood" if split == "ood" else "id")
    dropout = cfg.ood_sensor_dropout_prob if split == "ood" else cfg.sensor_dropout_prob
    sensor_model = SensorModel(cfg, dropout_prob=dropout)
    sim = ThermalSimulator(cfg, seed=seed)
    arrivals = generator.generate_episode(seed=seed + 101, episode_id=0)

    frames: list[dict[str, Any]] = []
    for timestep in range(cfg.episode_length):
        observed_state = sim.observe_state(sensor_model)
        readings = observed_state.sensor_readings.copy()
        mask = observed_state.sensor_mask.copy()
        trend = observed_state.observed_temp_trend.copy()

        selected_core: int | None = None
        task_power: float | None = None
        task_duration: int | None = None
        point_scores: list[float] | None = None
        quantile_scores: list[float] | None = None
        conformal_scores: list[float] | None = None

        for task in arrivals.get(timestep, []):
            state = sim.build_state(readings, mask, trend)
            if bundle is not None:
                candidates = build_candidate_matrix(state, task, cfg)
                point_scores = bundle.predict_point(candidates).astype(float).tolist()
                quantile_scores = bundle.predict_quantile(candidates).astype(float).tolist()
                conformal_scores = bundle.predict_conformal_upper(candidates).astype(float).tolist()
            selected_core = scheduler.choose_core(state, task)
            task_power = float(task.power)
            task_duration = int(task.duration)
            sim.assign_task(task, selected_core)

        sim.step()
        frames.append(
            {
                "timestep": timestep,
                "temperatures": sim.true_temperatures.astype(float).tolist(),
                "sensor_readings": readings.astype(float).tolist(),
                "sensor_mask": mask.astype(bool).tolist(),
                "selected_core": np.nan if selected_core is None else int(selected_core),
                "task_power": np.nan if task_power is None else float(task_power),
                "task_duration": np.nan if task_duration is None else int(task_duration),
                "point_scores": point_scores,
                "quantile_scores": quantile_scores,
                "conformal_scores": conformal_scores,
                "max_temperature": float(np.max(sim.true_temperatures)),
                "average_temperature": float(np.mean(sim.true_temperatures)),
                "active_tasks": int(np.sum(sim.task_counts)),
                "assigned_tasks": int(sim.assigned_tasks),
                "completed_tasks": int(sim.completed_tasks),
            }
        )
    return frames


@st.cache_resource(show_spinner=False)
def load_bundle_or_none(output_dir: Path) -> ModelBundle | None:
    bundle_path = output_dir / "models" / "model_bundle.joblib"
    if not bundle_path.exists():
        return None
    return load_model_bundle(output_dir)


def make_scheduler(name: str, cfg: ThermalGuardConfig, bundle: ModelBundle | None) -> Scheduler:
    if name == "random":
        return RandomScheduler(cfg, seed=cfg.random_seed + 500)
    if name == "round_robin":
        return RoundRobinScheduler(cfg)
    if name == "coolest_core_observed":
        return CoolestCoreObservedScheduler(cfg)
    if name == "coolest_core_oracle_true_temp":
        return CoolestCoreOracleScheduler(cfg)
    if name == "trend_aware_observed":
        return TrendAwareScheduler(cfg)
    if bundle is None:
        raise ValueError(f"Scheduler {name!r} requires a trained model bundle")
    if name == "point_prediction_rf":
        return PointPredictionScheduler(cfg, bundle)
    if name == "uncalibrated_quantile":
        return UncalibratedQuantileScheduler(cfg, bundle)
    if name == "conformal_upper_bound":
        return ConformalUpperBoundScheduler(cfg, bundle)
    raise ValueError(f"Unknown scheduler {name!r}")


def render_results_explorer(reports_dir: Path, figures_dir: Path) -> None:
    st.header("Results Explorer")
    metrics = load_scheduler_metrics(reports_dir)
    coverage = read_csv(reports_dir / "coverage_metrics.csv")
    model_metrics = read_csv(reports_dir / "model_metrics.csv")

    if metrics.empty:
        missing("scheduler metrics", reports_dir)
        return

    split_filter = st.multiselect("Split", sorted(metrics["split"].dropna().unique()), default=sorted(metrics["split"].dropna().unique()))
    sched_filter = st.multiselect(
        "Scheduler",
        sorted(metrics["scheduler"].dropna().unique()),
        default=sorted(metrics["scheduler"].dropna().unique()),
    )
    view = metrics[metrics["split"].isin(split_filter) & metrics["scheduler"].isin(sched_filter)]
    st.dataframe(view, use_container_width=True)

    chart_cols = st.columns(2)
    with chart_cols[0]:
        st.subheader("Peak Temperature")
        st.bar_chart(view.pivot(index="scheduler", columns="split", values="peak_temperature"))
    with chart_cols[1]:
        st.subheader("Hotspot Violations")
        st.bar_chart(view.pivot(index="scheduler", columns="split", values="hotspot_violations"))

    st.subheader("Coverage Metrics")
    show_table_or_missing(coverage, "coverage metrics")
    st.subheader("Model Metrics")
    show_table_or_missing(model_metrics, "model metrics")
    show_existing_figure(figures_dir / "executive_summary.png")


def render_calibration_view(reports_dir: Path) -> None:
    st.header("Calibration View")
    diagnostics = read_csv(reports_dir / "conformal_diagnostics.csv")
    coverage = read_csv(reports_dir / "coverage_metrics.csv")
    model_metrics = read_csv(reports_dir / "model_metrics.csv")

    if diagnostics.empty:
        missing("conformal diagnostics", reports_dir)
    else:
        st.dataframe(diagnostics, use_container_width=True)
        values = diagnostics.set_index("metric")["value"].to_dict()
        cols = st.columns(4)
        cols[0].metric("Target", f"{float(values.get('target_coverage', np.nan)):.3f}")
        cols[1].metric("Base alpha", f"{float(values.get('quantile_model_alpha', np.nan)):.3f}")
        cols[2].metric("Correction", f"{float(values.get('conformal_correction', np.nan)):.3f} C")
        cols[3].metric("Calibration n", f"{int(float(values.get('calibration_samples', 0)))}")

    if not model_metrics.empty:
        upper = model_metrics[
            (model_metrics["metric"] == "empirical_coverage")
            & (model_metrics["model"].isin(["quantile_upper", "conformal_upper"]))
        ]
        if not upper.empty:
            st.subheader("Before vs After Empirical Coverage")
            st.bar_chart(upper.pivot(index="split", columns="model", values="value"))

    st.subheader("Scheduler Rollout Coverage")
    show_table_or_missing(coverage, "coverage metrics")


def render_scheduler_comparison(reports_dir: Path) -> None:
    st.header("Scheduler Comparison")
    metrics = load_scheduler_metrics(reports_dir)
    if metrics.empty:
        missing("scheduler metrics", reports_dir)
        return

    focus = metrics[
        metrics["scheduler"].isin(
            ["point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"]
        )
    ].copy()
    st.subheader("Model-Based Schedulers")
    st.dataframe(focus, use_container_width=True)
    cols = st.columns(2)
    with cols[0]:
        st.bar_chart(focus.pivot(index="scheduler", columns="split", values="peak_temperature"))
    with cols[1]:
        st.bar_chart(focus.pivot(index="scheduler", columns="split", values="hotspot_violations"))

    st.subheader("All Scheduler Safety/Throughput")
    cols = [
        "split",
        "scheduler",
        "baseline_type",
        "peak_temperature",
        "average_max_temperature",
        "hotspot_violations",
        "completed_tasks",
        "selected_core_coverage",
    ]
    st.dataframe(metrics[[col for col in cols if col in metrics.columns]], use_container_width=True)


def render_heatmap_comparison(reports_dir: Path, figures_dir: Path) -> None:
    st.header("Heatmap Comparison")
    heatmap_files = sorted(figures_dir.glob("heatmap_*.png")) + sorted(figures_dir.glob("*heatmap*.png"))
    unique_files = []
    seen: set[Path] = set()
    for path in heatmap_files:
        if path not in seen:
            seen.add(path)
            unique_files.append(path)

    if unique_files:
        cols = st.columns(2)
        for idx, path in enumerate(unique_files):
            with cols[idx % 2]:
                st.caption(path.name)
                st.image(str(path), use_container_width=True)
    else:
        missing("heatmap figures", figures_dir)

    snapshots = reports_dir / "heatmap_snapshots.npz"
    if snapshots.exists():
        st.subheader("Raw Heatmap Snapshots")
        with np.load(snapshots) as data:
            keys = list(data.files)
            selected = st.selectbox("Snapshot", keys)
            cfg = make_config("quick")
            st.pyplot(plot_grid(data[selected].reshape(-1), cfg, title=selected, cmap="inferno"))


def render_stress_multiseed(reports_dir: Path, figures_dir: Path) -> None:
    st.header("Stress and Multiseed")
    recommendations = read_csv(reports_dir / "stress_sweep_recommendations.csv")
    stress = read_csv(reports_dir / "stress_sweep_results.csv")
    multiseed = read_csv(reports_dir / "multiseed_metrics_challenging.csv")

    st.subheader("Stress Sweep Recommendations")
    show_table_or_missing(recommendations, "stress sweep recommendations")
    st.subheader("Stress Sweep Results")
    show_table_or_missing(stress, "stress sweep results")

    st.subheader("Multiseed Metrics")
    show_table_or_missing(multiseed, "multiseed metrics")
    for name in (
        "multiseed_hotspots_challenging.png",
        "multiseed_peak_temp_challenging.png",
        "multiseed_coverage_challenging.png",
    ):
        show_existing_figure(figures_dir / name)


def load_scheduler_metrics(reports_dir: Path) -> pd.DataFrame:
    frames = []
    for name in ("metrics_id.csv", "metrics_ood.csv"):
        frame = read_csv(reports_dir / name)
        if not frame.empty:
            frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


def show_table_or_missing(frame: pd.DataFrame, label: str) -> None:
    if frame.empty:
        st.info(f"No {label} file is available yet.")
    else:
        st.dataframe(frame, use_container_width=True)


def missing(label: str, location: Path) -> None:
    st.info(f"No {label} found under `{location}`.")


def show_existing_figure(path: Path) -> None:
    if path.exists():
        st.image(str(path), caption=path.name, use_container_width=True)


def plot_grid(
    values: np.ndarray,
    cfg: ThermalGuardConfig,
    title: str,
    selected_core: int | None = None,
    sensor_mask: np.ndarray | None = None,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    missing_value: float | None = None,
) -> plt.Figure:
    flat = np.asarray(values, dtype=float).reshape(-1)
    if missing_value is not None:
        flat = np.where(np.isfinite(flat), flat, missing_value)
    grid = flat.reshape(cfg.grid_size, cfg.grid_size)
    fig, ax = plt.subplots(figsize=(4.4, 3.9))
    image = ax.imshow(grid, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(cfg.grid_size))
    ax.set_yticks(range(cfg.grid_size))
    for row in range(cfg.grid_size):
        for col in range(cfg.grid_size):
            value = grid[row, col]
            label = "--" if not np.isfinite(value) else f"{value:.1f}"
            ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8)
    if sensor_mask is not None:
        for core, present in enumerate(sensor_mask.reshape(-1)):
            row, col = divmod(core, cfg.grid_size)
            color = "#00d5ff" if present else "#555555"
            ax.scatter(col, row, marker="s", s=220, facecolors="none", edgecolors=color, linewidths=2)
    if selected_core is not None:
        row, col = divmod(int(selected_core), cfg.grid_size)
        ax.scatter(col, row, marker="o", s=360, facecolors="none", edgecolors="#00ff66", linewidths=3)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def write_replay_log(
    output_dir: Path,
    frames: pd.DataFrame,
    preset: str,
    split: str,
    scheduler_name: str,
    seed: int,
) -> Path:
    log_dir = output_dir / "replays"
    log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in frames.iterrows():
        payload = {
            "timestep": int(row["timestep"]),
            "preset": preset,
            "split": split,
            "scheduler": scheduler_name,
            "seed": int(seed),
            "selected_core": row["selected_core"],
            "task_power": row["task_power"],
            "task_duration": row["task_duration"],
            "max_temperature": row["max_temperature"],
            "average_temperature": row["average_temperature"],
            "active_tasks": row["active_tasks"],
            "assigned_tasks": row["assigned_tasks"],
            "completed_tasks": row["completed_tasks"],
        }
        for idx, value in enumerate(row["temperatures"]):
            payload[f"temp_{idx}"] = value
        for idx, value in enumerate(row["sensor_readings"]):
            payload[f"sensor_{idx}"] = value
        for idx, value in enumerate(row["sensor_mask"]):
            payload[f"sensor_present_{idx}"] = bool(value)
        rows.append(payload)
    path = log_dir / f"replay_{preset}_{split}_{scheduler_name}_seed{seed}.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            {
                "preset": preset,
                "split": split,
                "scheduler": scheduler_name,
                "seed": seed,
                "rows": len(rows),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return path


if __name__ == "__main__":
    main()
