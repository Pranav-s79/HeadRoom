"""Hosted-demo entry point for the HeadRoom dashboard.

Identical to ``run_dashboard.py`` except it points the dashboard's data root at the
small, git-tracked ``demo_data/`` directory instead of the generated ``outputs/``.
This lets the dashboard run on Streamlit Community Cloud (or any clean clone)
without first executing the full pipeline.

The data root is switched via the ``HEADROOM_DATA_ROOT`` environment variable, which
``dashboard/shared.py`` reads at import time.

    streamlit run run_dashboard_demo.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parent
    demo_data = root / "demo_data"
    env = os.environ.copy()
    env["HEADROOM_DATA_ROOT"] = str(demo_data)
    app_path = root / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(command, env=env))


if __name__ == "__main__":
    main()
