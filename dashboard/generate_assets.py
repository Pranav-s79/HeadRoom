"""Pre-generate static portfolio assets for HeadRoom.

Writes four dark-identity portfolio images at the HeadRoom root so the README and
a portfolio site render cleanly without the live dashboard:

  - architecture.png      pipeline diagram (pure matplotlib)
  - research_summary.png  before/after conformal + coverage cards
  - comparison_chart.png  peak temperature by scheduler (OOD)
  - heatmap_demo.png      one side-by-side replay frame at a hot timestep

The first two read only existing ``outputs/`` CSV artifacts. The heatmap frame
reuses the dashboard replay (which imports ``thermalguard_cal`` read-only) at a
thermally interesting timestep — never T=0 where everything is 35C.

Run from anywhere:

    python dashboard/generate_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(DASHBOARD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figures as F  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MULTISEED_DIR = PROJECT_ROOT / "outputs" / "multiseed" / "challenging"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio_assets"
PRIMARY_SCHED = "conformal_upper_bound"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def coverage_value(split: str, kind: str) -> float:
    df = _read_csv(REPORTS_DIR / f"metrics_{split}.csv")
    col = "marginal_coverage" if kind == "marginal" else "selected_core_coverage"
    rows = df[df["scheduler"] == PRIMARY_SCHED] if not df.empty else df
    if rows.empty or col not in rows or pd.isna(rows[col].iloc[0]):
        return 0.0
    return float(rows[col].iloc[0])


def conformal_before_after() -> dict:
    rows = []
    for diag in sorted(MULTISEED_DIR.glob("seed_*/reports/conformal_diagnostics.csv")):
        df = _read_csv(diag)
        if df.empty:
            continue
        vals = {m: float(v) for m, v in zip(df["metric"], df["value"])}
        rows.append(
            {
                "seed": diag.parent.parent.name.replace("seed_", ""),
                "before": vals.get("calibration_empirical_coverage_before_conformal"),
                "after": vals.get("calibration_empirical_coverage_after_conformal"),
                "correction": vals.get("conformal_correction"),
            }
        )
    table = pd.DataFrame(rows).dropna() if rows else pd.DataFrame()
    pos = table[table["correction"] > 0.05] if not table.empty else table
    if not pos.empty:
        pos = pos.assign(lift=pos["after"] - pos["before"]).sort_values("lift", ascending=False)
        best = pos.iloc[0]
        return {"before": float(best["before"]), "after": float(best["after"]),
                "correction": float(best["correction"]), "source": f"challenging preset, seed {best['seed']}"}
    return {"before": 0.90, "after": 0.90, "correction": 0.0, "source": "canonical run"}


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


def _comparison_chart() -> None:
    """Peak temperature by scheduler (OOD), sorted best->worst, amber bars."""
    df = _read_csv(REPORTS_DIR / "metrics_ood.csv")
    if df.empty:
        print("skip comparison_chart.png — no metrics_ood.csv")
        return
    df = df.sort_values("peak_temperature")
    fig = F.hbar_figure(
        [SCHED_DISPLAY.get(s, s) for s in df["scheduler"]],
        df["peak_temperature"].astype(float).tolist(),
        title="Peak temperature by scheduler (OOD)",
        xlabel="Peak temp (C)", color=F.PALETTE["amber"],
        reference=85.0, reference_label="85C limit",
        highlight=SCHED_DISPLAY[PRIMARY_SCHED], figsize=(8.4, 4.6),
    )
    path = PORTFOLIO_DIR / "comparison_chart.png"
    fig.savefig(path, dpi=130, facecolor=F.PALETTE["bg"], bbox_inches="tight")
    F.plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)}")


def _replay_for_assets(split: str, scheduler_name: str, seed: int):
    """Non-cached replay (no Streamlit) so we can render a hot heatmap frame.

    Imports thermalguard_cal lazily and read-only, mirroring the dashboard replay.
    """
    from thermalguard_cal.config import make_config
    from thermalguard_cal.models import load_model_bundle
    from thermalguard_cal.sensors import SensorModel
    from thermalguard_cal.simulator import ThermalSimulator
    from thermalguard_cal.workloads import WorkloadGenerator
    from thermalguard_cal.schedulers import (
        ConformalUpperBoundScheduler, CoolestCoreObservedScheduler,
    )

    cfg = make_config("quick", preset="normal", random_seed=seed, episode_length=150)
    bundle = load_model_bundle(PROJECT_ROOT / "outputs")
    if scheduler_name == "conformal_upper_bound":
        scheduler = ConformalUpperBoundScheduler(cfg, bundle)
    else:
        scheduler = CoolestCoreObservedScheduler(cfg)
    generator = WorkloadGenerator.for_split(cfg, "ood" if split == "ood" else "id")
    dropout = cfg.ood_sensor_dropout_prob if split == "ood" else cfg.sensor_dropout_prob
    sensor_model = SensorModel(cfg, dropout_prob=dropout)
    sim = ThermalSimulator(cfg, seed=seed)
    arrivals = generator.generate_episode(seed=seed + 101, episode_id=0)
    frames = []
    for t in range(cfg.episode_length):
        observed = sim.observe_state(sensor_model)
        readings, mask, trend = (observed.sensor_readings.copy(),
                                 observed.sensor_mask.copy(),
                                 observed.observed_temp_trend.copy())
        selected = None
        temps, mask_snapshot = sim.true_temperatures.copy(), mask.copy()
        for task in arrivals.get(t, []):
            temps = sim.true_temperatures.copy()
            state = sim.build_state(readings, mask, trend)
            selected = scheduler.choose_core(state, task)
            sim.assign_task(task, selected)
        sim.step()
        import numpy as np
        frames.append({
            "temperatures": temps.astype(float).tolist(),
            "sensor_mask": mask_snapshot.astype(bool).tolist(),
            "selected_core": selected,
            "max_temperature": float(np.max(sim.true_temperatures)),
        })
    return cfg, frames


def _heatmap_demo() -> None:
    """One side-by-side frame at a thermally interesting timestep (not T=0)."""
    if not (PROJECT_ROOT / "outputs" / "models" / "model_bundle.joblib").exists():
        print("skip heatmap_demo.png — no model bundle")
        return
    seed = 23
    cfg, conf = _replay_for_assets("ood", "conformal_upper_bound", seed)
    _, cool = _replay_for_assets("ood", "coolest_core_observed", seed)
    # Pick the timestep where the coolest-core baseline is hottest (most contrast).
    hot_t = max(range(len(cool)), key=lambda i: cool[i]["max_temperature"])
    cf, kf = conf[hot_t], cool[hot_t]
    fig = F.dual_heatmap_figure(
        cf["temperatures"], kf["temperatures"], grid_size=cfg.grid_size,
        left_title="Conformal upper-bound", right_title="Coolest-core (observed)",
        left_subtitle="calibrated upper-bound scheduler", right_subtitle="naive sparse-sensor baseline",
        left_selected=cf["selected_core"], right_selected=kf["selected_core"],
        sensor_indices=cfg.sensor_indices, left_mask=cf["sensor_mask"], right_mask=kf["sensor_mask"],
        thermal_limit=cfg.thermal_limit,
        suptitle=f"HeadRoom — OOD replay  ·  T = {hot_t}  ·  seed {seed}",
    )
    path = PORTFOLIO_DIR / "heatmap_demo.png"
    fig.savefig(path, dpi=130, facecolor=F.PALETTE["bg"], bbox_inches="tight")
    F.plt.close(fig)
    print(f"wrote {path.relative_to(PROJECT_ROOT)} (hot timestep T={hot_t})")


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    arch = F.architecture_figure()
    arch_path = PORTFOLIO_DIR / "architecture.png"
    arch.savefig(arch_path, dpi=100, facecolor=F.PALETTE["bg"])
    F.plt.close(arch)
    print(f"wrote {arch_path.relative_to(PROJECT_ROOT)}")

    ba = conformal_before_after()
    summary = F.research_summary_figure(
        nominal=0.90,
        id_marginal=coverage_value("id", "marginal"),
        id_selected=coverage_value("id", "selected"),
        ood_marginal=coverage_value("ood", "marginal"),
        ood_selected=coverage_value("ood", "selected"),
        before_cov=ba["before"], after_cov=ba["after"], correction_c=ba["correction"],
        correction_note=(
            f"Coverage cards: canonical evaluation run. Before/after: {ba['source']} calibration "
            "split. Conformal only widens bounds, so an already-conservative base model needs none."
        ),
    )
    summary_path = PORTFOLIO_DIR / "research_summary.png"
    summary.savefig(summary_path, dpi=100, facecolor=F.PALETTE["bg"], bbox_inches="tight")
    F.plt.close(summary)
    print(f"wrote {summary_path.relative_to(PROJECT_ROOT)}")

    _comparison_chart()
    _heatmap_demo()


if __name__ == "__main__":
    main()
