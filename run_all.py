from __future__ import annotations

import argparse
from pathlib import Path

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models
from thermalguard_cal.review_bundle import create_research_review_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermalGuard-Cal end to end.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings.")
    parser.add_argument("--full", action="store_true", help="Use full research settings.")
    parser.add_argument(
        "--preset",
        choices=("easy", "normal", "challenging", "stress"),
        default="normal",
        help="Workload/simulator stress preset.",
    )
    parser.add_argument("--seed", type=int, default=17, help="Base random seed.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run a multiseed aggregate instead of a single pipeline run.",
    )
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    if args.seeds:
        from run_multiseed import run_multiseed

        run_multiseed(mode=mode, preset=args.preset, seeds=args.seeds)
        return
    cfg = make_config(mode, preset=args.preset, random_seed=args.seed)

    print(f"Running ThermalGuard-Cal in {mode} mode with {cfg.preset} preset and seed {cfg.random_seed}")
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
    bundle_path = create_research_review_bundle(Path.cwd(), cfg.output_path)
    print(f"Research review bundle: {bundle_path}")
    print("Outputs written under outputs/")


if __name__ == "__main__":
    main()
