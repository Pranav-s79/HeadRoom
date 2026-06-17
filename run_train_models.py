from __future__ import annotations

import argparse

from thermalguard_cal.config import make_config
from thermalguard_cal.models import train_and_save_models


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ThermalGuard-Cal models.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings.")
    parser.add_argument("--full", action="store_true", help="Use full research settings.")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    cfg = make_config(mode)
    metrics = train_and_save_models(cfg)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
