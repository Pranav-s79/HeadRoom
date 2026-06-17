from pathlib import Path

import pandas as pd

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import generate_datasets, split_specs
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models


def test_tiny_end_to_end_output_contract(tmp_path):
    cfg = make_config(
        "quick",
        output_dir=str(tmp_path / "outputs"),
        episode_length=25,
        prediction_horizon=3,
        train_episodes=2,
        calibration_episodes=1,
        test_episodes=1,
        ood_episodes=1,
        random_forest_estimators=8,
        gradient_boosting_estimators=12,
        max_train_rows=2_000,
    )

    generate_datasets(cfg)
    train_and_save_models(cfg)
    metrics, coverage = evaluate_and_report(cfg)
    write_run_manifest(cfg, "quick")

    reports = Path(cfg.output_dir) / "reports"
    figures = Path(cfg.output_dir) / "figures"
    for rel in [
        reports / "metrics_id.csv",
        reports / "metrics_ood.csv",
        reports / "coverage_metrics.csv",
        reports / "policy_drift_metrics.csv",
        reports / "model_metrics.csv",
        reports / "final_report.md",
        reports / "run_manifest.json",
        figures / "max_temperature_by_scheduler.png",
        figures / "peak_temperature_bar.png",
        figures / "representative_heatmap.png",
    ]:
        assert rel.exists(), rel

    assert {"id", "ood"}.issubset(set(metrics["split"]))
    assert "baseline_type" in metrics.columns
    assert not coverage.empty

    id_trace = pd.read_csv(reports / "temperature_traces.csv")
    expected_ids = set({item.name: item for item in split_specs(cfg)}["test_id"].episode_ids)
    actual_ids = set(id_trace[id_trace["split"] == "id"]["episode_id"].unique())
    assert actual_ids == expected_ids
