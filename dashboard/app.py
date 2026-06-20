"""HeadRoom dashboard — calibrated upper-bound thermal scheduling.

Multipage Streamlit app. This module is the router only: it injects the locked
visual identity, initializes shared session state, and wires the four pages into
the sidebar via ``st.navigation``. Each page lives in ``dashboard/pages/`` and
renders independently:

  1. Watch It Run    — the hero side-by-side replay (default landing page)
  2. What We Found    — the 30-second results summary
  3. Under the Hood   — methodology for reviewers
  4. About            — what this is, and isn't

All research logic lives in ``thermalguard_cal`` and is imported read-only by
``dashboard/shared.py``. Every number shown is read from ``outputs/`` artifacts
(or ``demo_data/`` via the hosted-demo entry point) — none are hardcoded.

The historical sys.path fix (HeadRoom root before ``thermalguard_cal``) lives in
``shared.py`` and is applied on import below.
"""

from __future__ import annotations

import streamlit as st

import shared as S  # applies the sys.path fix on import


def main() -> None:
    S.page_config()
    S.inject_css()
    S.init_state()
    S.sidebar_brand()

    pages = [
        st.Page("pages/1_Watch_It_Run.py", title="Watch It Run", icon="▶", default=True),
        st.Page("pages/2_What_We_Found.py", title="What We Found", icon="📊"),
        st.Page("pages/3_Under_The_Hood.py", title="Under the Hood", icon="🔧"),
        st.Page("pages/4_About.py", title="About", icon="ℹ"),
    ]
    nav = st.navigation(pages, position="sidebar")
    nav.run()


if __name__ == "__main__":
    main()
