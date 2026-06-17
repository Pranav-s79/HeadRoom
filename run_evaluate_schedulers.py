from __future__ import annotations

import argparse

from thermalguard_cal.config import make_config
from thermalguard_cal.evaluate import evaluate_and_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate ThermalGuard-Cal schedulers.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings.")
    parser.add_argument("--full", action="store_true", help="Use full research settings.")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    cfg = make_config(mode)
    metrics, coverage = evaluate_and_report(cfg)
    print("Scheduler metrics:")
    print(metrics.to_string(index=False))
    print("\nCoverage metrics:")
    print(coverage.to_string(index=False))


if __name__ == "__main__":
    main()
