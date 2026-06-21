from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    if is_streamlit_runtime():
        dashboard_dir = PROJECT_ROOT / "dashboard"
        if str(dashboard_dir) not in sys.path:
            sys.path.insert(0, str(dashboard_dir))
        from dashboard.app import main as dashboard_main

        dashboard_main()
        return

    app_path = PROJECT_ROOT / "dashboard" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    raise SystemExit(subprocess.call(command))


def is_streamlit_runtime() -> bool:
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:
        return False
    return get_script_run_ctx() is not None


if __name__ == "__main__":
    main()
