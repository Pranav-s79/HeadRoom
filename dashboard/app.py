"""HeadRoom Streamlit dashboard router.

This file owns only the Streamlit page shell. Presentation helpers live in
dashboard/shared.py and dashboard/figures.py; research logic remains in
thermalguard_cal and is imported read-only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
for path in (PROJECT_ROOT, DASHBOARD_DIR):
    path_text = str(path)
    if path_text not in sys.path:
        sys.path.insert(0, path_text)

import shared as S


def main() -> None:
    S.page_config()
    S.inject_css()
    S.init_state()
    S.sidebar_brand()

    pages = [
        st.Page(
            str(DASHBOARD_DIR / "pages" / "1_Watch_It_Run.py"),
            title="Watch It Run",
            default=True,
        ),
        st.Page(
            str(DASHBOARD_DIR / "pages" / "2_What_We_Found.py"),
            title="What We Found",
        ),
        st.Page(
            str(DASHBOARD_DIR / "pages" / "3_Under_The_Hood.py"),
            title="Under the Hood",
        ),
        st.Page(
            str(DASHBOARD_DIR / "pages" / "4_About.py"),
            title="About",
        ),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
