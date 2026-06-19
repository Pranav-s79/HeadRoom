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
            "Guided Tutorial",
            "Pitch Prep",
            "Data Flow",
            "Metric Explainer",
            "Result Verdict",
            "Results Explorer",
            "Calibration View",
            "Scheduler Comparison",
            "Heatmap Comparison",
            "Stress and Multiseed",
        ],
    )

    if page == "Simulation Replay":
        render_simulation_replay(output_dir)
    elif page == "Guided Tutorial":
        render_guided_tutorial(reports_dir, figures_dir)
    elif page == "Pitch Prep":
        render_pitch_prep(reports_dir, figures_dir)
    elif page == "Data Flow":
        render_data_flow()
    elif page == "Metric Explainer":
        render_metric_explainer(reports_dir)
    elif page == "Result Verdict":
        render_result_verdict(reports_dir)
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
    beginner_mode = controls.checkbox("Beginner mode", value=True)

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
    cfg = make_config("quick", preset=preset, random_seed=int(seed), episode_length=int(episode_length))
    replay_key = f"{output_dir}:{preset}:{split}:{scheduler_name}:{seed}:{episode_length}"
    min_timestep = int(frames["timestep"].min())
    max_timestep = int(frames["timestep"].max())
    if st.session_state.get("replay_key") != replay_key:
        st.session_state["replay_key"] = replay_key
        st.session_state["replay_timestep"] = best_default_timestep(frames, cfg) or min_timestep
    if "replay_timestep" not in st.session_state:
        st.session_state["replay_timestep"] = min_timestep
    st.session_state["replay_timestep"] = min(
        max(int(st.session_state["replay_timestep"]), min_timestep),
        max_timestep,
    )

    with st.expander("How to read this page", expanded=True):
        st.markdown(
            "- **True temperature** shows the real simulated chip state.\n"
            "- **Sensor readings** show what the scheduler/model can observe through sparse noisy sensors.\n"
            "- **Point prediction** is the model's best guess for future peak chip temperature.\n"
            "- **Quantile upper** is an uncalibrated risk estimate.\n"
            "- **Conformal upper** is the calibrated safety ceiling.\n"
            "- The conformal scheduler chooses the core with the safest upper-bound risk, then uses load balance as a fallback among near-tied safe choices."
        )
    if beginner_mode:
        st.success(
            "Beginner mode: follow the page from top to bottom. First read the plain-English decision, "
            "then compare the true heatmap with the sensor heatmap, then look at why the selected core was chosen."
        )

    jump_cols = st.columns(5)
    jump_targets = {
        "Jump to first decision": first_decision_timestep(frames),
        "Jump to hottest timestep": int(frames.loc[frames["max_temperature"].idxmax(), "timestep"]),
        "Jump to first hotspot": first_condition_timestep(frames, frames["max_temperature"] > cfg.thermal_limit),
        "Jump to first conformal violation": first_condition_timestep(
            frames,
            (pd.to_numeric(frames["actual_future_peak"], errors="coerce") > pd.to_numeric(frames["selected_conformal_upper"], errors="coerce")),
        ),
    }
    observed_failure_target = coolest_core_failure_timestep(
        output_dir=output_dir,
        preset=preset,
        split=split,
        seed=int(seed),
        episode_length=int(episode_length),
        cfg=cfg,
    )
    jump_targets["Jump to coolest-core failure"] = observed_failure_target
    for column, (label, target) in zip(jump_cols, jump_targets.items()):
        with column:
            if st.button(label, disabled=target is None, use_container_width=True):
                st.session_state["replay_timestep"] = int(target)

    timestep = st.slider(
        "Timestep",
        min_value=min_timestep,
        max_value=max_timestep,
        step=1,
        key="replay_timestep",
    )
    frame = dict(frames[frames["timestep"] == timestep].iloc[0])

    metric_cols = st.columns(5)
    metric_cols[0].metric("Max temp", f"{frame['max_temperature']:.1f} C")
    metric_cols[1].metric("Avg temp", f"{frame['average_temperature']:.1f} C")
    metric_cols[2].metric("Active tasks", int(frame["active_tasks"]))
    metric_cols[3].metric("Assigned", int(frame["assigned_tasks"]))
    selected = "none" if pd.isna(frame["selected_core"]) else int(frame["selected_core"])
    metric_cols[4].metric("Selected core", selected)

    st.subheader("What is happening now?")
    st.info(decision_text(frame, scheduler_name, cfg))
    st.subheader("Why this core?")
    st.write(why_this_core_text(frame, scheduler_name, cfg))

    mode = st.radio(
        "Replay display",
        ["Guided Replay", "Demo Mode"],
        horizontal=True,
        help="Demo Mode keeps only the core evidence needed for a live explanation.",
    )
    if mode == "Demo Mode":
        render_demo_mode(frame, frames, cfg)
        if st.button("Save replay log"):
            path = write_replay_log(output_dir, frames, preset, split, scheduler_name, int(seed))
            st.success(f"Replay log written to `{path}`")
        return

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
                vmin=35.0,
                vmax=90.0,
            )
        )
        st.caption(f"Temperature color scale fixed at 35-90 C. Thermal limit: {cfg.thermal_limit:g} C.")
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
                vmin=35.0,
                vmax=90.0,
            )
        )
        st.caption(f"Unobserved sensor cells show `--`. Thermal limit: {cfg.thermal_limit:g} C.")
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

    st.subheader("Selected-Core Detail Table")
    with st.expander("What this table means", expanded=beginner_mode):
        st.markdown(
            "Each row is one possible placement core. `selected=True` is the scheduler's choice. "
            "`point_prediction` is the model's best estimate, `quantile_upper` is an uncalibrated "
            "risk estimate, and `conformal_upper` is the calibrated safety ceiling. Lower predicted "
            "future peak temperature usually means safer placement. `current_temp`, `current_load`, "
            "and `sensor_observed` explain what the scheduler saw at decision time."
        )
    st.dataframe(candidate_detail_table(frame, cfg), use_container_width=True)

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
                        vmin=35.0,
                        vmax=90.0,
                    )
                )
                st.caption(f"Fixed temperature scale: 35-90 C. Thermal limit: {cfg.thermal_limit:g} C.")

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
        actual_future_peak: float | None = None
        point_scores: list[float] | None = None
        quantile_scores: list[float] | None = None
        conformal_scores: list[float] | None = None
        selected_point_prediction: float | None = None
        selected_quantile_upper: float | None = None
        selected_conformal_upper: float | None = None
        selected_sensor_observed: bool | None = None
        decision_temperatures = sim.true_temperatures.copy()
        decision_load = sim.load.copy()

        for task in arrivals.get(timestep, []):
            decision_temperatures = sim.true_temperatures.copy()
            decision_load = sim.load.copy()
            state = sim.build_state(readings, mask, trend)
            if bundle is not None:
                candidates = build_candidate_matrix(state, task, cfg)
                point_scores = bundle.predict_point(candidates).astype(float).tolist()
                quantile_scores = bundle.predict_quantile(candidates).astype(float).tolist()
                conformal_scores = bundle.predict_conformal_upper(candidates).astype(float).tolist()
            selected_core = scheduler.choose_core(state, task)
            actual_future_peak = float(
                sim.future_peak_after_assignment(task, selected_core, cfg.prediction_horizon)
            )
            if point_scores is not None:
                selected_point_prediction = float(point_scores[selected_core])
            if quantile_scores is not None:
                selected_quantile_upper = float(quantile_scores[selected_core])
            if conformal_scores is not None:
                selected_conformal_upper = float(conformal_scores[selected_core])
            selected_sensor_observed = bool(mask[selected_core])
            task_power = float(task.power)
            task_duration = int(task.duration)
            sim.assign_task(task, selected_core)

        sim.step()
        frames.append(
            {
                "timestep": timestep,
                "temperatures": decision_temperatures.astype(float).tolist(),
                "load": decision_load.astype(float).tolist(),
                "sensor_readings": readings.astype(float).tolist(),
                "sensor_mask": mask.astype(bool).tolist(),
                "selected_core": np.nan if selected_core is None else int(selected_core),
                "task_power": np.nan if task_power is None else float(task_power),
                "task_duration": np.nan if task_duration is None else int(task_duration),
                "point_scores": point_scores,
                "quantile_scores": quantile_scores,
                "conformal_scores": conformal_scores,
                "selected_point_prediction": np.nan if selected_point_prediction is None else selected_point_prediction,
                "selected_quantile_upper": np.nan if selected_quantile_upper is None else selected_quantile_upper,
                "selected_conformal_upper": np.nan if selected_conformal_upper is None else selected_conformal_upper,
                "actual_future_peak": np.nan if actual_future_peak is None else actual_future_peak,
                "selected_sensor_observed": np.nan if selected_sensor_observed is None else selected_sensor_observed,
                "max_temperature": float(np.max(sim.true_temperatures)),
                "average_temperature": float(np.mean(sim.true_temperatures)),
                "active_tasks": int(np.sum(sim.task_counts)),
                "assigned_tasks": int(sim.assigned_tasks),
                "completed_tasks": int(sim.completed_tasks),
            }
        )
    return frames


def first_decision_timestep(frames: pd.DataFrame) -> int | None:
    decisions = frames[pd.notna(frames["selected_core"])]
    if decisions.empty:
        return None
    return int(decisions["timestep"].iloc[0])


def first_condition_timestep(frames: pd.DataFrame, condition: pd.Series) -> int | None:
    if condition.empty:
        return None
    matches = frames[condition.fillna(False)]
    if matches.empty:
        return None
    return int(matches["timestep"].iloc[0])


def best_default_timestep(frames: pd.DataFrame, cfg: ThermalGuardConfig) -> int | None:
    violation = first_condition_timestep(
        frames,
        (pd.to_numeric(frames["actual_future_peak"], errors="coerce") > pd.to_numeric(frames["selected_conformal_upper"], errors="coerce")),
    )
    if violation is not None:
        return violation
    hotspot = first_condition_timestep(frames, frames["max_temperature"] > cfg.thermal_limit)
    if hotspot is not None:
        return hotspot
    decision = first_decision_timestep(frames)
    if decision is not None:
        return decision
    if frames.empty:
        return None
    return int(frames.loc[frames["max_temperature"].idxmax(), "timestep"])


def coolest_core_failure_timestep(
    output_dir: Path,
    preset: str,
    split: str,
    seed: int,
    episode_length: int,
    cfg: ThermalGuardConfig,
) -> int | None:
    observed_replay = pd.DataFrame(
        build_replay(
            output_dir=str(output_dir),
            preset=preset,
            split=split,
            scheduler_name="coolest_core_observed",
            seed=seed,
            episode_length=episode_length,
        )
    )
    if observed_replay.empty:
        return None
    badly_hot = observed_replay["max_temperature"] >= cfg.thermal_limit + 5.0
    target = first_condition_timestep(observed_replay, badly_hot)
    if target is not None:
        return target
    return first_condition_timestep(observed_replay, observed_replay["max_temperature"] > cfg.thermal_limit)


def why_this_core_text(frame: dict[str, Any], scheduler_name: str, cfg: ThermalGuardConfig) -> str:
    selected = selected_core_or_none(frame)
    if selected is None:
        return "No task arrived at this timestep, so the scheduler did not choose a core."
    table = candidate_detail_table(frame, cfg)
    selected_row = table[table["candidate_core"] == selected].iloc[0]
    if scheduler_name == "conformal_upper_bound" and pd.notna(selected_row["conformal_upper"]):
        ranked = table.dropna(subset=["conformal_upper"]).sort_values(
            ["conformal_upper", "current_load", "current_temp", "candidate_core"]
        )
        rank = int(np.flatnonzero(ranked["candidate_core"].to_numpy() == selected)[0]) + 1
        best_core = int(ranked["candidate_core"].iloc[0])
        best_bound = float(ranked["conformal_upper"].iloc[0])
        selected_bound = float(selected_row["conformal_upper"])
        return (
            f"The conformal scheduler compares calibrated upper-bound risk for all 16 candidate cores. "
            f"Core {selected} ranked #{rank} by conformal risk with bound {selected_bound:.1f} C. "
            f"The lowest-risk core by raw bound was Core {best_core} at {best_bound:.1f} C; if several "
            "cores are within the near-tie threshold, the scheduler falls back to lower current load and "
            "cooler observed temperature."
        )
    if scheduler_name == "uncalibrated_quantile" and pd.notna(selected_row["quantile_upper"]):
        best = table.dropna(subset=["quantile_upper"]).sort_values(["quantile_upper", "candidate_core"]).iloc[0]
        return (
            f"The uncalibrated quantile scheduler chooses the core with the lowest upper-quantile risk. "
            f"It selected Core {selected}; the lowest displayed quantile bound is Core "
            f"{int(best['candidate_core'])} at {float(best['quantile_upper']):.1f} C."
        )
    if scheduler_name == "point_prediction_rf" and pd.notna(selected_row["point_prediction"]):
        best = table.dropna(subset=["point_prediction"]).sort_values(["point_prediction", "candidate_core"]).iloc[0]
        return (
            f"The point-prediction scheduler chooses the lowest predicted future peak. It selected "
            f"Core {selected}; the lowest displayed point prediction is Core {int(best['candidate_core'])} "
            f"at {float(best['point_prediction']):.1f} C."
        )
    if scheduler_name == "coolest_core_observed":
        observed = "was" if bool(selected_row["sensor_observed"]) else "was not"
        return (
            f"The observed coolest-core heuristic picks the coolest imputed sensor temperature. It selected "
            f"Core {selected}, which {observed} directly observed by a sensor; missing sensor cells are filled "
            "from available sparse readings."
        )
    if scheduler_name == "coolest_core_oracle_true_temp":
        return (
            f"The oracle scheduler reads true simulator temperatures and chose Core {selected}. This is useful "
            "as a reference but is not deployable with sparse sensors."
        )
    if scheduler_name == "trend_aware_observed":
        return (
            f"The trend-aware heuristic combines sparse observed temperature with recent observed temperature "
            f"change. It selected Core {selected} based on that observed risk score."
        )
    if scheduler_name == "round_robin":
        return f"Round-robin ignores temperature and assigns the next task to Core {selected} in a fixed cycle."
    if scheduler_name == "random":
        return f"Random scheduling ignores thermal state and sampled Core {selected}."
    return f"The scheduler selected Core {selected}."


def decision_text(frame: dict[str, Any], scheduler_name: str, cfg: ThermalGuardConfig) -> str:
    timestep = int(frame["timestep"])
    selected_core = selected_core_or_none(frame)
    if selected_core is None:
        return (
            f"At timestep {timestep}, no task arrived. The dashboard is showing the current "
            "simulated chip state and the most recent peak-temperature trajectory."
        )

    point = optional_temp(frame.get("selected_point_prediction"))
    quantile = optional_temp(frame.get("selected_quantile_upper"))
    conformal = optional_temp(frame.get("selected_conformal_upper"))
    actual = optional_temp(frame.get("actual_future_peak"))
    observed = "was" if bool(frame.get("selected_sensor_observed")) else "was not"
    held_text = ""
    if pd.notna(frame.get("actual_future_peak")) and pd.notna(frame.get("selected_conformal_upper")):
        held = float(frame["actual_future_peak"]) <= float(frame["selected_conformal_upper"])
        held_text = " The conformal bound held." if held else " The conformal bound was violated."

    return (
        f"At timestep {timestep}, a task with power {float(frame['task_power']):.2f} "
        f"and duration {int(frame['task_duration'])} arrived. The `{scheduler_name}` scheduler "
        f"selected Core {selected_core}, which {observed} directly sensor-observed. "
        f"The point model predicted {point}, the uncalibrated quantile upper was {quantile}, "
        f"and the conformal upper bound was {conformal}. The actual realized future peak over "
        f"the horizon was {actual}.{held_text}"
    )


def optional_temp(value: object) -> str:
    if value is None or pd.isna(value):
        return "not available"
    return f"{float(value):.1f} C"


def selected_core_or_none(frame: dict[str, Any]) -> int | None:
    selected = frame.get("selected_core")
    if selected is None or pd.isna(selected):
        return None
    return int(selected)


def candidate_detail_table(frame: dict[str, Any], cfg: ThermalGuardConfig) -> pd.DataFrame:
    selected = selected_core_or_none(frame)
    temperatures = np.array(frame["temperatures"], dtype=float)
    loads = np.array(frame["load"], dtype=float)
    mask = np.array(frame["sensor_mask"], dtype=bool)
    point = score_array(frame.get("point_scores"), cfg)
    quantile = score_array(frame.get("quantile_scores"), cfg)
    conformal = score_array(frame.get("conformal_scores"), cfg)
    return pd.DataFrame(
        {
            "candidate_core": list(range(cfg.num_cores)),
            "selected": [core == selected for core in range(cfg.num_cores)],
            "point_prediction": point,
            "quantile_upper": quantile,
            "conformal_upper": conformal,
            "current_temp": np.round(temperatures, 3),
            "current_load": np.round(loads, 3),
            "sensor_observed": mask,
        }
    )


def score_array(scores: object, cfg: ThermalGuardConfig) -> np.ndarray:
    if scores is None or not isinstance(scores, list) or len(scores) != cfg.num_cores:
        return np.full(cfg.num_cores, np.nan)
    return np.round(np.array(scores, dtype=float), 3)


def render_demo_mode(frame: dict[str, Any], frames: pd.DataFrame, cfg: ThermalGuardConfig) -> None:
    selected = selected_core_or_none(frame)
    selected_label = "none" if selected is None else f"Core {selected}"
    bound = frame.get("selected_conformal_upper")
    actual = frame.get("actual_future_peak")
    if selected is None:
        sentence = "No task arrived at this timestep, so there is no scheduler decision to explain."
    elif pd.notna(bound) and pd.notna(actual):
        status = "held" if float(actual) <= float(bound) else "was violated"
        sentence = (
            f"The conformal scheduler selected {selected_label}; the future peak was "
            f"{float(actual):.1f} C against a calibrated bound of {float(bound):.1f} C, "
            f"so the bound {status}."
        )
    else:
        sentence = f"The scheduler selected {selected_label}; model bounds are not available for this frame."

    st.markdown(f"**Demo explanation:** {sentence}")
    cols = st.columns([1, 1, 0.9])
    with cols[0]:
        st.subheader("True Temperature")
        st.pyplot(
            plot_grid(
                np.array(frame["temperatures"], dtype=float),
                cfg,
                title="True temperature",
                selected_core=selected,
                sensor_mask=np.array(frame["sensor_mask"], dtype=bool),
                cmap="inferno",
                vmin=35.0,
                vmax=90.0,
            )
        )
        st.caption(f"Fixed color scale: 35-90 C. Thermal limit: {cfg.thermal_limit:g} C.")
    with cols[1]:
        st.subheader("Conformal Upper Bound")
        scores = frame.get("conformal_scores")
        if scores is None or not isinstance(scores, list) or len(scores) == 0:
            st.info("No conformal score for this frame.")
        else:
            st.pyplot(
                plot_grid(
                    np.array(scores, dtype=float),
                    cfg,
                    title="Calibrated upper-bound risk",
                    selected_core=selected,
                    cmap="cividis",
                    vmin=35.0,
                    vmax=90.0,
                )
            )
            st.caption(f"Lower bound means safer predicted placement. Thermal limit: {cfg.thermal_limit:g} C.")
    with cols[2]:
        st.metric("Selected core", selected_label)
        st.metric("Actual future peak", optional_temp(actual))
        st.metric("Selected conformal bound", optional_temp(bound))
    st.subheader("Peak Temperature Over Time")
    st.line_chart(frames[["timestep", "max_temperature"]].set_index("timestep"))


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


def render_guided_tutorial(reports_dir: Path, figures_dir: Path) -> None:
    st.header("Guided Tutorial")
    st.markdown(
        "Use this page as a first-pass walkthrough. It explains the project in the order a reviewer should learn it."
    )
    steps = [
        ("1. The problem", "A many-core chip can overheat when work is placed on already-hot or highly loaded cores."),
        ("2. The simulator", "ThermalGuard-Cal simulates a 4x4 chip with heat gain, ambient cooling, diffusion, and sparse noisy sensors."),
        ("3. The prediction target", "The models predict future peak whole-chip temperature after assigning a task to a candidate core."),
        ("4. The schedulers", "Baselines use random, round-robin, sparse observed temperatures, or oracle true temperature; model schedulers use learned risk."),
        ("5. The conformal layer", "The conformal calibrator widens an upper-bound model until calibration coverage reaches the target on calibration-like data."),
        ("6. The main lesson", "ID behavior can look safe while OOD or challenging workloads reveal where sparse-sensor heuristics and calibration assumptions break."),
    ]
    for title, body in steps:
        with st.expander(title, expanded=True):
            st.write(body)
    show_existing_figure(figures_dir / "executive_summary.png")
    st.info("Next: open Simulation Replay, keep Beginner mode enabled, and jump to the first decision.")


def render_pitch_prep(reports_dir: Path, figures_dir: Path) -> None:
    st.header("Pitch Prep")
    metrics = load_scheduler_metrics(reports_dir)
    diagnostics = read_csv(reports_dir / "conformal_diagnostics.csv")
    st.subheader("30-second pitch")
    st.write(
        "ThermalGuard-Cal is a research MVP that studies task placement on a simulated 4x4 many-core chip. "
        "It compares simple schedulers, sparse-sensor thermal heuristics, and learned future-temperature "
        "models with conformal upper bounds. The honest result is that calibration helps quantify trust on "
        "calibration-like data, while OOD workloads can still break coverage."
    )
    st.subheader("Supported claims")
    for claim in supported_claims(metrics, diagnostics):
        st.markdown(f"- {claim}")
    st.subheader("Unsupported claims")
    for claim in unsupported_claims():
        st.markdown(f"- {claim}")
    st.subheader("Likely reviewer questions")
    qa = {
        "What exactly does the model predict?": "Future peak whole-chip temperature over the prediction horizon after a candidate assignment.",
        "Is the oracle deployable?": "No. It reads true simulator temperatures and is only a reference baseline.",
        "Does conformal solve OOD safety?": "No. It provides marginal calibration-style evidence under calibration-like data, not an OOD guarantee.",
        "Why is selected-core coverage separate?": "The scheduler chooses one candidate out of 16; that selection step can change coverage.",
    }
    for question, answer in qa.items():
        with st.expander(question):
            st.write(answer)
    show_existing_figure(figures_dir / "coverage_id_vs_ood.png")


def render_data_flow() -> None:
    st.header("Data Flow")
    st.markdown(
        """
1. `run_generate_data.py` creates action-conditioned examples for every task/core candidate.
2. Feature builders use sparse sensor observations, task metadata, power, load, and candidate-core identity.
3. Labels are simulator-computed future peak chip temperature over the horizon.
4. `run_train_models.py` fits point and upper-quantile models, then fits the conformal correction on calibration data.
5. `run_evaluate_schedulers.py` replays schedulers on ID and OOD workloads and writes metrics, traces, coverage, drift, and heatmap snapshots.
6. `run_make_plots.py` turns CSV outputs into figures.
7. The Streamlit dashboard reads those same outputs and can run a small local replay for explanation.
"""
    )
    st.subheader("What does not flow into model features")
    st.write(
        "True temperatures are used by simulator physics and labels, but feature construction uses sparse/noisy sensor observations. "
        "That separation is what makes sparse-sensor failure and oracle-vs-deployable comparisons meaningful."
    )


def render_metric_explainer(reports_dir: Path) -> None:
    st.header("Metric Explainer")
    glossary = pd.DataFrame(
        [
            ("peak_temperature", "Maximum chip temperature reached during an evaluated rollout.", "Lower is safer."),
            ("average_max_temperature", "Average over timesteps of the chip's maximum core temperature.", "Lower means steadier thermal behavior."),
            ("hotspot_violations", "Number of timesteps above the thermal limit.", "Zero is best; nonzero means overheating occurred."),
            ("hotspot_timestep_pct", "Fraction of evaluated timesteps that were hotspots.", "Normalizes hotspot count by run length."),
            ("completed_tasks", "Number of tasks completed by the simulator.", "Checks whether thermal safety came at a throughput cost."),
            ("marginal_coverage", "Coverage over all candidate cores on visited states.", "Conformal's usual candidate-level view."),
            ("selected_core_coverage", "Coverage only after the scheduler chooses one core.", "Measures the deployed decision path."),
            ("selected_coverage_gap", "Selected coverage minus target coverage.", "Negative means below nominal."),
            ("drift_mean_abs_z", "Average feature shift vs calibration distribution.", "Higher means the policy visits less calibration-like states."),
        ],
        columns=["Metric", "Plain-English meaning", "How to interpret"],
    )
    st.dataframe(glossary, use_container_width=True)
    diagnostics = read_csv(reports_dir / "conformal_diagnostics.csv")
    if not diagnostics.empty:
        st.subheader("Current conformal diagnostics")
        st.dataframe(diagnostics, use_container_width=True)


def render_result_verdict(reports_dir: Path) -> None:
    st.header("Result Verdict")
    metrics = load_scheduler_metrics(reports_dir)
    diagnostics = read_csv(reports_dir / "conformal_diagnostics.csv")
    coverage = read_csv(reports_dir / "coverage_metrics.csv")
    if metrics.empty:
        missing("scheduler metrics", reports_dir)
        return
    verdict = build_verdict_table(metrics, diagnostics, coverage)
    st.dataframe(verdict, use_container_width=True)
    st.subheader("Honest interpretation")
    st.write(
        "The project is strongest as an explainable simulator and calibration study. It shows how scheduler choice, sparse sensing, "
        "distribution shift, and conformal calibration interact. The safest pitch is not that conformal always wins, but that the "
        "pipeline can measure when calibration helps and when OOD behavior invalidates stronger claims."
    )


def supported_claims(metrics: pd.DataFrame, diagnostics: pd.DataFrame) -> list[str]:
    claims = [
        "The codebase runs an end-to-end simulated thermal scheduling pipeline.",
        "The dashboard visualizes true simulated state, sparse sensor observations, scheduler choices, and model risk estimates.",
    ]
    if not diagnostics.empty:
        correction = diagnostic_value(diagnostics, "conformal_correction")
        before = diagnostic_value(diagnostics, "calibration_empirical_coverage_before_conformal")
        after = diagnostic_value(diagnostics, "calibration_empirical_coverage_after_conformal")
        if correction is not None and before is not None and after is not None:
            claims.append(
                f"On the current calibration split, conformal correction is {correction:.2f} C and coverage changes from {before:.3f} to {after:.3f}."
            )
    if not metrics.empty and "hotspot_violations" in metrics:
        deployable = metrics[metrics["baseline_type"] == "deployable_baseline_sensor_observed"]
        model = metrics[metrics["baseline_type"] == "model_based_sensor_observed"]
        if not deployable.empty and not model.empty:
            claims.append(
                "The current metrics compare sparse-sensor heuristics against model-based schedulers on the same rollout summaries."
            )
    return claims


def unsupported_claims() -> list[str]:
    return [
        "This is not validated against real silicon or a detailed industrial thermal tool.",
        "Conformal calibration is not proven to guarantee OOD safety.",
        "The oracle true-temperature scheduler is not deployable.",
        "The current MVP does not prove optimal scheduling, production readiness, or hardware implementation.",
    ]


def diagnostic_value(diagnostics: pd.DataFrame, metric: str) -> float | None:
    if diagnostics.empty:
        return None
    rows = diagnostics[diagnostics["metric"] == metric]
    if rows.empty:
        return None
    return float(rows["value"].iloc[0])


def build_verdict_table(metrics: pd.DataFrame, diagnostics: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    id_cov = coverage_value(coverage, "id", "selected")
    ood_cov = coverage_value(coverage, "ood", "selected")
    sparse_failure = bool(
        ((metrics["baseline_type"] == "deployable_baseline_sensor_observed") & (metrics["hotspot_violations"] > 0)).any()
    )
    correction = diagnostic_value(diagnostics, "conformal_correction")
    rows = [
        ("Pipeline status", "working", "Metrics and figures are present under outputs/reports and outputs/figures."),
        ("ID selected-core coverage", verdict_label(id_cov, 0.90), format_optional_float(id_cov)),
        ("OOD selected-core coverage", verdict_label(ood_cov, 0.90), format_optional_float(ood_cov)),
        ("Sparse-sensor baseline failure", "yes" if sparse_failure else "not in current metrics", "Nonzero hotspot violations in deployable heuristic rows."),
        ("Conformal correction", "positive" if correction and correction > 0 else "zero or missing", format_optional_float(correction)),
        ("Research claim strength", "promising, not proven", "Useful simulator evidence, not real-hardware proof."),
    ]
    return pd.DataFrame(rows, columns=["Question", "Verdict", "Evidence"])


def coverage_value(coverage: pd.DataFrame, split: str, token: str) -> float | None:
    if coverage.empty:
        return None
    rows = coverage[
        (coverage["split"] == split)
        & coverage["coverage_type"].astype(str).str.contains(token, case=False, na=False)
    ]
    if rows.empty:
        return None
    return float(rows["empirical_coverage"].iloc[0])


def verdict_label(value: float | None, target: float) -> str:
    if value is None:
        return "missing"
    return "good" if value >= target - 0.03 else "below target"


def format_optional_float(value: float | None) -> str:
    if value is None:
        return "not available"
    return f"{value:.3f}"


def bound_conservatism_table(
    model_metrics: pd.DataFrame,
    scheduler_metrics: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "quantity": "ID test conformal upper - true outcome, all saved test predictions",
                "average_c": model_metric_value(model_metrics, "test_id", "conformal_upper", "average_conservatism"),
            },
            {
                "quantity": "ID rollout conformal upper - realized outcome, all candidates on visited states",
                "average_c": coverage_metric_value(
                    coverage,
                    "id",
                    "marginal_all_candidates_on_visited_states",
                    "average_conservatism",
                ),
            },
            {
                "quantity": "ID rollout conformal upper - realized outcome, selected scheduler core",
                "average_c": scheduler_metric_value(
                    scheduler_metrics,
                    "id",
                    "conformal_upper_bound",
                    "average_selected_conservatism",
                ),
            },
        ]
    )


def model_metric_value(model_metrics: pd.DataFrame, split: str, model: str, metric: str) -> float | None:
    if model_metrics.empty:
        return None
    required = {"split", "model", "metric", "value"}
    if not required.issubset(model_metrics.columns):
        return None
    rows = model_metrics[
        (model_metrics["split"] == split)
        & (model_metrics["model"] == model)
        & (model_metrics["metric"] == metric)
    ]
    if rows.empty:
        return None
    return float(rows["value"].iloc[0])


def coverage_metric_value(coverage: pd.DataFrame, split: str, coverage_type: str, column: str) -> float | None:
    if coverage.empty or column not in coverage.columns:
        return None
    rows = coverage[(coverage["split"] == split) & (coverage["coverage_type"] == coverage_type)]
    if rows.empty or pd.isna(rows[column].iloc[0]):
        return None
    return float(rows[column].iloc[0])


def scheduler_metric_value(metrics: pd.DataFrame, split: str, scheduler: str, column: str) -> float | None:
    if metrics.empty or column not in metrics.columns:
        return None
    rows = metrics[(metrics["split"] == split) & (metrics["scheduler"] == scheduler)]
    if rows.empty or pd.isna(rows[column].iloc[0]):
        return None
    return float(rows[column].iloc[0])


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
    scheduler_metrics = load_scheduler_metrics(reports_dir)

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

    st.subheader("Bound Conservatism Audit")
    st.write(
        "Positive values mean the calibrated upper bound is above the realized future peak. "
        "Large positive values are safe but may be too loose for differentiated scheduling."
    )
    st.dataframe(bound_conservatism_table(model_metrics, scheduler_metrics, coverage), use_container_width=True)

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
    colormap = plt.get_cmap(cmap).copy()
    colormap.set_bad("#555555")
    image = ax.imshow(grid, cmap=colormap, vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(range(cfg.grid_size))
    ax.set_yticks(range(cfg.grid_size))
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    for row in range(cfg.grid_size):
        for col in range(cfg.grid_size):
            core = row * cfg.grid_size + col
            value = grid[row, col]
            label = f"{core}\n--" if not np.isfinite(value) else f"{core}\n{value:.1f}"
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
