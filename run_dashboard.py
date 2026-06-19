from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    app_path = Path(__file__).resolve().parent / "thermalguard_cal" / "dashboard.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(command))


if __name__ == "__main__":
    main()
