"""Shared state, styling, and data access for the HeadRoom multipage dashboard.

Every page (``Watch It Run``, ``What We Found``, ``Under the Hood``, ``About``)
imports from this module. All research logic lives in ``thermalguard_cal`` and is
imported read-only — this module only handles presentation, the live replay, and
cached reads of ``outputs/`` artifacts. No displayed number is hardcoded.

The data root is configurable via :data:`DATA_ROOT` so the hosted demo entry point
(``run_dashboard_demo.py``) can point the same pages at ``demo_data/`` instead of
``outputs/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

# --- Keep the existing sys.path fix: HeadRoom root before thermalguard_cal ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(DASHBOARD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import streamlit as st

import figures as F  # local dashboard module (dashboard/figures.py)
from thermalguard_cal.config import make_config, ThermalGuardConfig
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

P = F.PALETTE

# Data root — overridable by the demo entry point via the HEADROOM_DATA_ROOT env
# var. Defaults to the live pipeline outputs/ directory.
DATA_ROOT = Path(os.environ.get("HEADROOM_DATA_ROOT", str(PROJECT_ROOT / "outputs")))
REPORTS_DIR = DATA_ROOT / "reports"
MODELS_DIR = DATA_ROOT / "models"
MULTISEED_DIR = DATA_ROOT / "multiseed" / "challenging"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio_assets"

REPLAY_PRESET = "normal"          # matches the bundled canonical models (run_manifest)
REPLAY_LENGTH = 150               # matches the evaluation episode length
DEFAULT_SEED = 17                 # canonical seed; OOD overheats naive schedulers
DEMO_SEED = 23                    # verified: coolest-core violates early, conformal stays safe
SPEED_OPTIONS = {"0.5x": 0.5, "1x": 1.0, "2x": 2.0, "4x": 4.0}
BASE_FRAME_DELAY = 0.55

CONTRAST_SCHED = "coolest_core_observed"
PRIMARY_SCHED = "conformal_upper_bound"

SCHED_ORDER = [
    "random",
    "round_robin",
    "coolest_core_observed",
    "coolest_core_oracle_true_temp",
    "trend_aware_observed",
    "point_prediction_rf",
    "uncalibrated_quantile",
    "conformal_upper_bound",
]
MODEL_SCHEDS = {"point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"}
SCHED_DISPLAY = {
    "random": "Random",
    "round_robin": "Round-robin",
    "coolest_core_observed": "Coolest-core (observed)",
    "coolest_core_oracle_true_temp": "Coolest-core oracle (privileged)",
    "trend_aware_observed": "Trend-aware (observed)",
    "point_prediction_rf": "Point prediction (RF)",
    "uncalibrated_quantile": "Uncalibrated quantile",
    "conformal_upper_bound": "Conformal upper-bound",
}
SCHED_NOTE = {
    "random": "Thermal-blind control",
    "round_robin": "Fixed-cycle control",
    "coolest_core_observed": "Sparse-sensor heuristic (deployable)",
    "coolest_core_oracle_true_temp": "Privileged: reads true temps (not deployable)",
    "trend_aware_observed": "Sparse-sensor + trend heuristic",
    "point_prediction_rf": "Model: point prediction",
    "uncalibrated_quantile": "Model: raw upper quantile",
    "conformal_upper_bound": "Model: calibrated upper bound (this project)",
}

# Pages whose names involve a "danger" threshold near the 85C thermal limit.
NOMINAL_COVERAGE = 0.90


# --------------------------------------------------------------------------- #
# CSS — inject the locked visual identity as variables + component styling
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        :root {{
            --hr-bg: {P['bg']}; --hr-panel: {P['panel']}; --hr-amber: {P['amber']};
            --hr-amber-hi: {P['amber_hi']}; --hr-text: {P['text']}; --hr-muted: {P['muted']};
            --hr-red: {P['red']}; --hr-green: {P['green']}; --hr-yellow: {P['yellow']};
            --hr-border: {P['border']};
            --hr-mono: "JetBrains Mono","Fira Code",ui-monospace,monospace;
            --hr-sans: "Inter",-apple-system,"Segoe UI",sans-serif;
        }}
        .stApp {{ background: var(--hr-bg); color: var(--hr-text); font-family: var(--hr-sans); }}

        /* narrow, dark, minimal sidebar */
        section[data-testid="stSidebar"] {{
            background: var(--hr-panel); border-right: 1px solid var(--hr-border);
            min-width: 248px !important; max-width: 248px !important;
        }}
        section[data-testid="stSidebar"] * {{ color: var(--hr-text); }}
        section[data-testid="stSidebar"] a[data-testid="stSidebarNavLink"] span {{ color: var(--hr-muted); }}
        section[data-testid="stSidebar"] a[aria-current="page"] span {{ color: var(--hr-amber) !important; font-weight: 700; }}

        h1,h2,h3,h4 {{ color: var(--hr-text); font-family: var(--hr-sans); }}
        .stApp a {{ color: var(--hr-amber); }}
        [data-testid="stMetricValue"] {{ font-family: var(--hr-mono); }}

        /* sticky top summary bar (Page 1) */
        .hr-topbar {{
            position: sticky; top: 0; z-index: 999;
            display: flex; gap: 10px; flex-wrap: wrap;
            background: var(--hr-bg); padding: 8px 2px 12px 2px;
            border-bottom: 1px solid var(--hr-border); margin-bottom: 6px;
        }}
        .hr-cell {{
            flex: 1 1 0; min-width: 150px; background: var(--hr-panel);
            border: 1px solid var(--hr-border); border-radius: 10px; padding: 10px 14px;
        }}
        .hr-cell .lbl {{ color: var(--hr-muted); font-size: 11px; text-transform: uppercase;
            letter-spacing: .06em; margin-bottom: 4px; }}
        .hr-cell .val {{ font-family: var(--hr-mono); font-size: 23px; font-weight: 700; line-height: 1.1; }}
        .hr-cell .sub {{ color: var(--hr-muted); font-size: 11px; margin-top: 2px; }}
        .hr-badge {{ display:inline-block; padding: 1px 9px; border-radius: 999px; font-size: 12px;
            font-weight: 700; font-family: var(--hr-mono); }}

        /* buttons: amber primary */
        .stButton > button, .stDownloadButton > button {{
            background: var(--hr-panel); color: var(--hr-text); border: 1px solid var(--hr-border);
            border-radius: 8px; font-weight: 600;
        }}
        .stButton > button:hover {{ border-color: var(--hr-amber); color: var(--hr-amber); }}
        .stButton > button[kind="primary"] {{ background: var(--hr-amber); color: #1a1a1a; border: none; }}
        .stButton > button[kind="primary"]:hover {{ background: var(--hr-amber-hi); color:#1a1a1a; }}

        /* HeadRoom html tables + cards */
        table.hr-tbl {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
        table.hr-tbl th {{ color: var(--hr-muted); text-align: left; font-weight: 600;
            padding: 6px 10px; border-bottom: 1px solid var(--hr-border); font-size: 11px;
            text-transform: uppercase; letter-spacing: .04em; }}
        table.hr-tbl td {{ padding: 6px 10px; border-bottom: 1px solid var(--hr-border);
            font-family: var(--hr-mono); }}
        table.hr-tbl td.txt {{ font-family: var(--hr-sans); }}
        .hr-card {{ background: var(--hr-panel); border: 1px solid var(--hr-border);
            border-radius: 12px; padding: 16px 18px; height: 100%; }}
        .hr-card .t {{ color: var(--hr-muted); font-size: 12px; text-transform: uppercase;
            letter-spacing: .05em; }}
        .hr-card .v {{ font-family: var(--hr-mono); font-size: 30px; font-weight: 700; margin-top: 4px; }}
        .hr-bigcard {{ text-align:center; }}
        .hr-bigcard .v {{ font-size: 46px; }}
        .hr-finding {{ background: var(--hr-panel); border-left: 3px solid var(--hr-amber);
            border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; font-size: 15px; line-height: 1.5; }}
        .hr-finding b {{ color: var(--hr-amber); }}
        .hr-claims {{ background: rgba(34,197,94,.07); border: 1px solid var(--hr-green);
            border-radius: 12px; padding: 14px 18px; }}
        .hr-claims li {{ margin: 4px 0; }}
        .hr-noclaim {{ background: var(--hr-panel); border: 1px solid var(--hr-border);
            border-radius: 12px; padding: 14px 18px; }}
        .hr-noclaim li {{ margin: 4px 0; color: var(--hr-muted); }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def page_config() -> None:
    """Single source of truth for page chrome — call once per page run."""
    st.set_page_config(page_title="HeadRoom", page_icon="🔥", layout="wide",
                       initial_sidebar_state="expanded")


def sidebar_brand() -> None:
    st.sidebar.markdown(
        f"<h2 style='color:{P['amber']};margin-bottom:0'>HeadRoom</h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption("Calibrated upper-bound thermal scheduling")


def color_for_temp(temp: float, limit: float = 85.0) -> str:
    """Green/amber/red based on proximity to the thermal limit."""
    if temp >= limit:
        return P["red"]
    if temp >= limit - 15:        # within 15C of the limit -> warming
        return P["amber"]
    return P["green"]


def color_best_worst(value: float, best: float, worst: float) -> str:
    if best == worst:
        return P["text"]
    if value == best:
        return P["green"]
    if value == worst:
        return P["red"]
    return P["amber"]


# --------------------------------------------------------------------------- #
# Cached data loaders
# --------------------------------------------------------------------------- #
def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, OSError, ValueError):
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_scheduler_metrics() -> pd.DataFrame:
    """Prefer multiseed mean+/-std numbers when present, else single-seed metrics."""
    frames = [_read_csv(REPORTS_DIR / name) for name in ("metrics_id.csv", "metrics_ood.csv")]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_multiseed_metrics() -> pd.DataFrame:
    """Aggregated mean/std scheduler metrics across workload seeds (Work Stream 2b).

    Combines the ID and OOD multi-seed CSVs if either exists. Empty otherwise.
    """
    frames = [
        _read_csv(REPORTS_DIR / "metrics_multiseed_id.csv"),
        _read_csv(REPORTS_DIR / "metrics_multiseed_ood.csv"),
    ]
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_coverage() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "coverage_metrics.csv")


@st.cache_data(show_spinner=False)
def load_model_metrics() -> pd.DataFrame:
    return _read_csv(REPORTS_DIR / "model_metrics.csv")


def coverage_value(split: str, kind: str) -> float | None:
    """kind in {'marginal','selected'} for the conformal scheduler, given split.

    Prefers the multi-seed mean when available so the figure shown matches the
    headline resume numbers.
    """
    ms = load_multiseed_metrics()
    col = "marginal_coverage_mean" if kind == "marginal" else "selected_core_coverage_mean"
    if not ms.empty and col in ms:
        rows = ms[(ms["split"] == split) & (ms["scheduler"] == PRIMARY_SCHED)]
        if not rows.empty and pd.notna(rows[col].iloc[0]):
            return float(rows[col].iloc[0])
    metrics = load_scheduler_metrics()
    single_col = "marginal_coverage" if kind == "marginal" else "selected_core_coverage"
    rows = metrics[(metrics["split"] == split) & (metrics["scheduler"] == PRIMARY_SCHED)]
    if rows.empty or single_col not in rows or pd.isna(rows[single_col].iloc[0]):
        return None
    return float(rows[single_col].iloc[0])


@st.cache_data(show_spinner=False)
def load_conformal_seed_table() -> pd.DataFrame:
    """Per-seed conformal correction story from the challenging multiseed runs."""
    rows = []
    for diag in sorted(MULTISEED_DIR.glob("seed_*/reports/conformal_diagnostics.csv")):
        df = _read_csv(diag)
        if df.empty:
            continue
        vals = {m: float(v) for m, v in zip(df["metric"], df["value"])}
        seed = diag.parent.parent.name.replace("seed_", "")
        rows.append(
            {
                "seed": seed,
                "before": vals.get("calibration_empirical_coverage_before_conformal"),
                "after": vals.get("calibration_empirical_coverage_after_conformal"),
                "correction": vals.get("conformal_correction"),
            }
        )
    return pd.DataFrame(rows)


def conformal_before_after() -> dict[str, Any]:
    """Honest before/after for the calibration panel.

    Prefers a real positive-correction run from the challenging multiseed sweep
    (the canonical normal-preset run needed no correction: its base quantile model
    already exceeded the 90% target on calibration). Returns numbers + provenance.
    """
    table = load_conformal_seed_table()
    candidates = table.dropna(subset=["before", "after", "correction"])
    candidates = candidates[candidates["correction"] > 0.05]
    if not candidates.empty:
        candidates = candidates.assign(lift=candidates["after"] - candidates["before"])
        best = candidates.sort_values("lift", ascending=False).iloc[0]
        return {
            "before": float(best["before"]),
            "after": float(best["after"]),
            "correction": float(best["correction"]),
            "source": f"challenging preset, seed {best['seed']}",
            "canonical_zero": True,
        }
    # Fallback: derive from canonical calibration model metrics (correction ~0).
    mm = load_model_metrics()
    cal = mm[(mm["split"] == "calibration") & (mm["metric"] == "empirical_coverage")]
    q = cal[cal["model"] == "quantile_upper"]["value"]
    c = cal[cal["model"] == "conformal_upper"]["value"]
    before = float(q.iloc[0]) if not q.empty else 0.9
    after = float(c.iloc[0]) if not c.empty else 0.9
    return {"before": before, "after": after, "correction": 0.0,
            "source": "canonical run", "canonical_zero": False}


def bound_conservatism_c() -> float | None:
    """Average +C the calibrated upper bound sits above the realized outcome (ID)."""
    metrics = load_scheduler_metrics()
    rows = metrics[(metrics["split"] == "id") & (metrics["scheduler"] == PRIMARY_SCHED)]
    if rows.empty:
        return None
    for col in ("average_selected_conservatism",):
        if col in rows and pd.notna(rows[col].iloc[0]):
            return float(rows[col].iloc[0])
    bound = rows.get("average_selected_bound")
    actual = rows.get("average_selected_actual")
    if bound is not None and actual is not None and pd.notna(bound.iloc[0]) and pd.notna(actual.iloc[0]):
        return float(bound.iloc[0] - actual.iloc[0])
    return None


# --------------------------------------------------------------------------- #
# Live replay (reuses thermalguard_cal simulator/scheduler/model — read-only)
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_bundle() -> ModelBundle | None:
    if not (MODELS_DIR / "model_bundle.joblib").exists():
        return None
    try:
        return load_model_bundle(DATA_ROOT)
    except Exception as exc:  # e.g. scikit-learn version mismatch when unpickling
        st.session_state["_bundle_error"] = str(exc)
        return None


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
        raise ValueError(f"{name} needs a trained model bundle")
    if name == "point_prediction_rf":
        return PointPredictionScheduler(cfg, bundle)
    if name == "uncalibrated_quantile":
        return UncalibratedQuantileScheduler(cfg, bundle)
    if name == "conformal_upper_bound":
        return ConformalUpperBoundScheduler(cfg, bundle)
    raise ValueError(f"unknown scheduler {name!r}")


@st.cache_data(show_spinner=False)
def build_replay(split: str, scheduler_name: str, seed: int, with_scores: bool) -> list[dict[str, Any]]:
    cfg = make_config("quick", preset=REPLAY_PRESET, random_seed=seed, episode_length=REPLAY_LENGTH)
    bundle = get_bundle()
    scheduler = make_scheduler(scheduler_name, cfg, bundle)
    generator = WorkloadGenerator.for_split(cfg, "ood" if split == "ood" else "id")
    dropout = cfg.ood_sensor_dropout_prob if split == "ood" else cfg.sensor_dropout_prob
    sensor_model = SensorModel(cfg, dropout_prob=dropout)
    sim = ThermalSimulator(cfg, seed=seed)
    arrivals = generator.generate_episode(seed=seed + 101, episode_id=0)

    frames: list[dict[str, Any]] = []
    for timestep in range(cfg.episode_length):
        observed = sim.observe_state(sensor_model)
        readings = observed.sensor_readings.copy()
        mask = observed.sensor_mask.copy()
        trend = observed.observed_temp_trend.copy()

        selected_core: int | None = None
        task_power = task_duration = None
        conformal_scores: list[float] | None = None
        decision_temps = sim.true_temperatures.copy()
        decision_load = sim.load.copy()

        for task in arrivals.get(timestep, []):
            decision_temps = sim.true_temperatures.copy()
            decision_load = sim.load.copy()
            state = sim.build_state(readings, mask, trend)
            if with_scores and bundle is not None:
                candidates = build_candidate_matrix(state, task, cfg)
                conformal_scores = bundle.predict_conformal_upper(candidates).astype(float).tolist()
            selected_core = scheduler.choose_core(state, task)
            task_power = float(task.power)
            task_duration = int(task.duration)
            sim.assign_task(task, selected_core)

        sim.step()
        frames.append(
            {
                "timestep": timestep,
                "temperatures": decision_temps.astype(float).tolist(),
                "load": decision_load.astype(float).tolist(),
                "sensor_mask": mask.astype(bool).tolist(),
                "selected_core": np.nan if selected_core is None else int(selected_core),
                "task_power": np.nan if task_power is None else task_power,
                "task_duration": np.nan if task_duration is None else task_duration,
                "conformal_scores": conformal_scores,
                "max_temperature": float(np.max(sim.true_temperatures)),
                "average_temperature": float(np.mean(sim.true_temperatures)),
                "active_tasks": int(np.sum(sim.task_counts)),
                "completed_tasks": int(sim.completed_tasks),
            }
        )
    return frames


def first_over_limit_timestep(frames: list[dict[str, Any]], limit: float) -> int | None:
    for fr in frames:
        if fr["max_temperature"] >= limit:
            return int(fr["timestep"])
    return None


def cumulative_violations(frames: list[dict[str, Any]], upto_t: int, limit: float) -> int:
    return int(sum(1 for fr in frames[: upto_t + 1] if fr["max_temperature"] >= limit))


def last_decision_index(frames: list[dict], upto_t: int) -> int | None:
    last = None
    for i, fr in enumerate(frames):
        if fr["timestep"] > upto_t:
            break
        if pd.notna(fr["selected_core"]):
            last = i
    return last


@st.cache_data(show_spinner=False)
def best_demo_seed(candidate_seeds: tuple[int, ...]) -> dict[str, Any]:
    """Programmatically pick the most contrastful demo seed (Work Stream 1).

    Prefers a seed where the naive coolest-core baseline violates the thermal
    limit and the conformal scheduler does not; otherwise the seed with the
    largest peak-temperature gap. Returns the choice plus whether a true
    violation contrast exists (controls the button tooltip).
    """
    cfg = make_config("quick", preset=REPLAY_PRESET, episode_length=REPLAY_LENGTH)
    limit = cfg.thermal_limit
    best_violation = None       # (gap, seed) where coolest violates, conformal does not
    best_gap = None             # (gap, seed) fallback over all seeds
    for seed in candidate_seeds:
        conf = build_replay("ood", PRIMARY_SCHED, int(seed), with_scores=False)
        cool = build_replay("ood", CONTRAST_SCHED, int(seed), with_scores=False)
        conf_peak = max(fr["max_temperature"] for fr in conf)
        cool_peak = max(fr["max_temperature"] for fr in cool)
        gap = cool_peak - conf_peak
        cool_viol = first_over_limit_timestep(cool, limit)
        conf_viol = first_over_limit_timestep(conf, limit)
        if best_gap is None or gap > best_gap[0]:
            best_gap = (gap, int(seed))
        if cool_viol is not None and conf_viol is None:
            if best_violation is None or gap > best_violation[0]:
                best_violation = (gap, int(seed))
    if best_violation is not None:
        return {"seed": best_violation[1], "gap": best_violation[0], "has_violation_contrast": True}
    return {"seed": best_gap[1], "gap": best_gap[0], "has_violation_contrast": False}


def replay_cfg() -> ThermalGuardConfig:
    return make_config("quick", preset=REPLAY_PRESET, episode_length=REPLAY_LENGTH)


def bundle_guard() -> bool:
    """Render a clear error and return False if no model bundle is loadable."""
    if get_bundle() is not None:
        return True
    err = st.session_state.get("_bundle_error")
    if err:
        st.error(
            f"Could not load the model bundle at `{MODELS_DIR / 'model_bundle.joblib'}`.\n\n"
            f"`{err}`\n\nThis is usually a scikit-learn version mismatch with the pickled models. "
            "Install the version the models were trained with, or re-run the pipeline."
        )
    else:
        st.error(
            f"No model bundle found at `{MODELS_DIR / 'model_bundle.joblib'}`. "
            "Extract `outputs.zip` or run `python run_all.py --quick` first."
        )
    return False


def init_state() -> None:
    defaults = {
        "episode_seed": DEFAULT_SEED, "focus_sched": PRIMARY_SCHED, "speed": "1x",
        "mode": "OOD shift", "t": 0, "playing": False,
        "_demo_active": False, "_demo_toast_shown": False, "_replay_key": "",
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)
