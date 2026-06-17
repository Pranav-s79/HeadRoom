from __future__ import annotations

import argparse

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermalGuard-Cal end to end.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings.")
    parser.add_argument("--full", action="store_true", help="Use full research settings.")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    cfg = make_config(mode)

    print(f"Running ThermalGuard-Cal in {mode} mode")
    summaries = generate_datasets(cfg)
    for split, summary in summaries.items():
        print(f"{split}: {summary['rows']} rows")
    model_metrics = train_and_save_models(cfg)
    print("Model metrics:")
    print(model_metrics.to_string(index=False))
    scheduler_metrics, coverage = evaluate_and_report(cfg)
    write_run_manifest(cfg, mode)
    print("Scheduler metrics:")
    print(scheduler_metrics.to_string(index=False))
    print("Coverage metrics:")
    print(coverage.to_string(index=False))
    print("Outputs written under outputs/")


if __name__ == "__main__":
    main()
