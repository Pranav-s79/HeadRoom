from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from thermalguard_cal.config import PRESET_OVERRIDES, make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models


SWEEP_COLUMNS = [
    "candidate",
    "preset",
    "split",
    "scheduler",
    "peak_temperature",
    "average_max_temperature",
    "hotspot_violations",
    "hotspot_timestep_pct",
    "completed_tasks",
    "assigned_tasks",
    "selected_core_coverage",
    "marginal_coverage",
    "hit_max_temperature_cap",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep ThermalGuard-Cal stress presets.")
    parser.add_argument("--quick", action="store_true", help="Use compact quick-mode sweep settings.")
    parser.add_argument("--full", action="store_true", help="Use full-mode base settings.")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    results, recommendation = run_stress_sweep(mode)
    print(results[SWEEP_COLUMNS].to_string(index=False))
    print("\nRecommended challenging preset:")
    print(recommendation.to_string())


def run_stress_sweep(mode: str = "quick") -> tuple[pd.DataFrame, pd.Series]:
    output_root = Path("outputs")
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []

    for candidate in _candidate_configs():
        cfg = make_config(
            mode,
            preset=str(candidate["preset"]),
            output_dir=str(output_root / "stress_sweep" / str(candidate["candidate"])),
            **candidate["overrides"],
        )
        print(f"Running stress sweep candidate {candidate['candidate']}")
        generate_datasets(cfg)
        train_and_save_models(cfg)
        metrics, _coverage = evaluate_and_report(cfg, write_plots=False)
        write_run_manifest(cfg, mode)
        metrics = metrics.copy()
        metrics["candidate"] = str(candidate["candidate"])
        metrics["preset"] = str(candidate["preset"])
        rows.append(metrics)

    results = pd.concat(rows, ignore_index=True)
    ranked = _rank_candidates(results)
    recommendable = ranked[ranked["preset"] != "stress"]
    recommendation = recommendable.iloc[0] if not recommendable.empty else ranked.iloc[0]
    results_path = reports_dir / "stress_sweep_results.csv"
    ranked_path = reports_dir / "stress_sweep_recommendations.csv"
    results[SWEEP_COLUMNS].to_csv(results_path, index=False)
    ranked.to_csv(ranked_path, index=False)
    (reports_dir / "stress_sweep_summary.md").write_text(
        _stress_sweep_markdown(results, ranked),
        encoding="utf-8",
    )
    return results, recommendation


def _candidate_configs() -> list[dict[str, Any]]:
    compact_overrides = {
        "episode_length": 70,
        "train_episodes": 4,
        "calibration_episodes": 2,
        "test_episodes": 2,
        "ood_episodes": 2,
        "random_forest_estimators": 16,
        "gradient_boosting_estimators": 36,
        "max_train_rows": 15_000,
    }
    return [
        {
            "candidate": "normal",
            "preset": "normal",
            "overrides": compact_overrides,
        },
        {
            "candidate": "challenging",
            "preset": "challenging",
            "overrides": compact_overrides,
        },
        {
            "candidate": "challenging_more_burst",
            "preset": "challenging",
            "overrides": {
                **compact_overrides,
                "id_burst_probability": 0.12,
                "id_burst_extra_rate": 1.1,
                "sensor_dropout_prob": 0.10,
            },
        },
        {
            "candidate": "stress",
            "preset": "stress",
            "overrides": compact_overrides,
        },
    ]


def _rank_candidates(results: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for candidate, group in results.groupby("candidate"):
        deployable = group[
            (group["baseline_type"] == "deployable_baseline_sensor_observed")
            & (~group["uses_oracle"].astype(bool))
        ]
        model_based = group[group["scheduler"].isin(["point_prediction_rf", "uncalibrated_quantile", "conformal_upper_bound"])]
        conformal = group[group["scheduler"] == "conformal_upper_bound"]
        baseline_violators = int((deployable["hotspot_violations"] > 0).sum())
        model_peak_std = float(model_based["peak_temperature"].std(ddof=0)) if len(model_based) > 1 else 0.0
        model_hotspot_std = float(model_based["hotspot_violations"].std(ddof=0)) if len(model_based) > 1 else 0.0
        cap_hit_count = int(group["hit_max_temperature_cap"].astype(bool).sum())
        all_schedulers_fail = bool((group["hotspot_violations"] > 0).all())
        model_all_identical = bool(model_peak_std < 0.25 and model_hotspot_std < 0.25)
        score = 0
        score += 4 if baseline_violators >= 2 else baseline_violators
        score += 2 if not model_all_identical else 0
        score += 2 if cap_hit_count == 0 else -cap_hit_count
        score += 2 if not all_schedulers_fail else -2
        score += 1 if not conformal.empty and conformal["selected_core_coverage"].notna().any() else 0
        rows.append(
            {
                "candidate": candidate,
                "preset": str(group["preset"].iloc[0]),
                "score": score,
                "baseline_violating_schedulers": baseline_violators,
                "model_peak_std": model_peak_std,
                "model_hotspot_std": model_hotspot_std,
                "cap_hit_schedulers": cap_hit_count,
                "all_schedulers_fail": all_schedulers_fail,
                "model_all_identical": model_all_identical,
                "recommended_settings": _settings_summary(str(group["preset"].iloc[0])),
            }
        )
    ranked = pd.DataFrame(rows)
    return ranked.sort_values(
        ["score", "baseline_violating_schedulers", "cap_hit_schedulers"],
        ascending=[False, False, True],
    ).reset_index(drop=True)


def _settings_summary(preset: str) -> str:
    settings = PRESET_OVERRIDES[preset]
    keys = [
        "heat_gain",
        "cooling_coeff",
        "diffusion_coeff",
        "sensor_dropout_prob",
        "id_arrival_rate",
        "id_task_mix",
        "id_burst_probability",
        "id_burst_extra_rate",
        "id_power_scale",
        "id_duration_scale",
    ]
    return "; ".join(f"{key}={settings[key]}" for key in keys if key in settings)


def _stress_sweep_markdown(results: pd.DataFrame, ranked: pd.DataFrame) -> str:
    view = results[SWEEP_COLUMNS].copy()
    ranking = ranked.copy()
    return f"""# Stress Sweep Summary

The sweep uses compact quick-mode settings to screen preset behavior before the
full quick pipeline is rerun with the recommended preset.

## Ranked Candidates

{ranking.to_markdown(index=False)}

## Scheduler Metrics

{view.to_markdown(index=False)}
"""


if __name__ == "__main__":
    main()
