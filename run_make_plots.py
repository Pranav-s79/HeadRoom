"""Regenerate all ThermalGuard-Cal figures from existing report outputs.

This is a standalone visual-results layer. It does not re-run the simulator,
models, or scheduler evaluation. It reads the CSVs and heatmap snapshots that
`run_all.py` (or `run_evaluate_schedulers.py`) already wrote under
`outputs/reports/` and rebuilds every figure under `outputs/figures/`, then
refreshes the "Visual Results" section of `outputs/reports/final_report.md`.

Run it any time after the evaluation step:

    python run_make_plots.py
    python run_make_plots.py --full      # use full-mode config (paths only)
    python run_make_plots.py --no-report # rebuild figures without touching the report
"""

from __future__ import annotations

import argparse

from thermalguard_cal.config import make_config
from thermalguard_cal.plots import make_plots_from_outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild ThermalGuard-Cal figures from existing report CSVs."
    )
    parser.add_argument("--full", action="store_true", help="Use full-mode config.")
    parser.add_argument("--quick", action="store_true", help="Use quick-mode config (default).")
    parser.add_argument(
        "--preset",
        choices=("easy", "normal", "challenging", "stress"),
        default="normal",
        help="Stress preset used for labels and thresholds.",
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Only regenerate figures; do not update final_report.md.",
    )
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    cfg = make_config(mode, preset=args.preset)

    reports = cfg.output_path / "reports"
    if not (reports / "metrics_id.csv").exists() and not (reports / "metrics_ood.csv").exists():
        raise SystemExit(
            f"No scheduler metrics found under {reports}. "
            "Run `python run_all.py --quick` (or run_evaluate_schedulers.py) first."
        )

    make_plots_from_outputs(cfg, update_report=not args.no_report)
    print(f"Figures written under {cfg.output_path / 'figures'}")
    if not args.no_report:
        print(f"Visual Results section refreshed in {reports / 'final_report.md'}")


if __name__ == "__main__":
    main()
