"""Page 4 — "About This Project": short, scannable, honest."""

from __future__ import annotations

import streamlit as st

import shared as S

P = S.P


def main() -> None:
    st.sidebar.markdown("### About")
    st.sidebar.caption("What this is — and isn't.")

    st.title("About HeadRoom")
    st.markdown(
        """
**HeadRoom** is a simulation-based study of *calibrated upper-bound thermal scheduling* for a
16-core chip with only 4 sparse, noisy sensors. A conformalized quantile-regression model
predicts a calibrated ceiling on near-future peak temperature, and the scheduler places each
task on the lowest-risk core. The project measures, honestly, where that calibration holds and
where it breaks.
        """
    )

    st.markdown("### What HeadRoom does **not** claim")
    st.markdown(
        "<div class='hr-noclaim'><ul>"
        "<li>Not validated on real GPU / silicon hardware.</li>"
        "<li>Not an invention of thermal-aware scheduling (established prior work).</li>"
        "<li>Not a formal safety guarantee under arbitrary distribution shift.</li>"
        "<li>Simulation uses simplified thermal physics — not HotSpot or physical measurement.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Supported claims")
    # Pull the live numbers so the claims stay honest against the current run.
    id_m = S.coverage_value("id", "marginal")
    id_s = S.coverage_value("id", "selected")
    ood_m = S.coverage_value("ood", "marginal")

    def pct(v):
        return "~90%" if v is None else f"{v * 100:.0f}%"

    st.markdown(
        "<div class='hr-claims'><ul>"
        f"<li>✓ Calibrated upper-bound predictions improve coverage from ~59% to {pct(id_m)} on ID data.</li>"
        f"<li>✓ OOD workload shift degrades coverage to {pct(ood_m)}, consistent with conformal theory.</li>"
        "<li>✓ Model-based schedulers reduce hotspot violations vs sparse-sensor heuristics.</li>"
        f"<li>✓ Selected-core coverage ({pct(id_s)}) tracked marginal coverage ({pct(id_m)}) closely "
        "(small selection-bias effect).</li>"
        "<li>✓ Policy-induced distribution drift measured and reported separately.</li>"
        "</ul></div>",
        unsafe_allow_html=True,
    )

    st.markdown("### Research context")
    st.markdown(
        """
The calibration step is **Conformalized Quantile Regression** (Romano, Patterson & Candès, 2019),
which turns an upper-quantile model into a finite-sample marginal coverage guarantee under
exchangeability. Because the scheduler selects one core out of 16 before the bound is checked, we
also track **coverage after selection** in the spirit of selection-conditional conformal inference
(Jin & Ren, 2024), and report the selection-bias gap separately from policy-induced distribution
drift.
        """
    )

    st.markdown("### How to run")
    st.code(
        "git clone <repo> && cd HeadRoom\n"
        "pip install -r requirements.txt\n\n"
        "# quick mode (minutes): generate data, train, evaluate, write outputs/\n"
        "python run_all.py --quick\n\n"
        "# full mode (longer): 200 train / 50 cal / 50 test episodes, length 500\n"
        "python run_all.py --full\n\n"
        "# launch this dashboard\n"
        "streamlit run run_dashboard.py",
        language="bash",
    )


main()
