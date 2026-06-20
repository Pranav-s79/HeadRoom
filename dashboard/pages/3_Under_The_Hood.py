"""Page 3 — "Under the Hood": methodology for reviewers.

Coverage analysis, policy drift, the full 16-core decision table, bound
conservatism, and model performance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import shared as S

P = S.P
PRIMARY_SCHED = S.PRIMARY_SCHED
SCHED_DISPLAY = S.SCHED_DISPLAY
SCHED_ORDER = S.SCHED_ORDER
NOMINAL = S.NOMINAL_COVERAGE


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def cov_col(v):
    return P["green"] if (v is not None and v >= NOMINAL - 0.005) else P["red"]


def section_coverage() -> None:
    st.markdown("### Coverage analysis")
    st.caption("Empirical coverage of the calibrated upper bound vs the 90% nominal target.")
    id_m, id_s = S.coverage_value("id", "marginal"), S.coverage_value("id", "selected")
    ood_m, ood_s = S.coverage_value("ood", "marginal"), S.coverage_value("ood", "selected")
    for title, m, s in [("In-distribution", id_m, id_s),
                        ("Out-of-distribution (OOD shift)", ood_m, ood_s)]:
        st.markdown(f"**{title}**")
        cols = st.columns(3)
        for col, lbl, val, color in [
            (cols[0], "Nominal target", pct(NOMINAL), P["muted"]),
            (cols[1], "Marginal coverage", pct(m), cov_col(m)),
            (cols[2], "Selected-core coverage", pct(s), cov_col(s)),
        ]:
            col.markdown(
                f"<div class='hr-card'><div class='t'>{lbl}</div>"
                f"<div class='v' style='color:{color}'>{val}</div></div>",
                unsafe_allow_html=True,
            )
        st.write("")
    st.markdown(
        f"<div class='hr-finding'>In-distribution, calibration holds (marginal {pct(id_m)}). "
        f"Under OOD workload shift, coverage drops to {pct(ood_m)}. This is expected: conformal "
        f"guarantees require calibration and test data to come from the same distribution.</div>",
        unsafe_allow_html=True,
    )


def section_drift() -> None:
    st.markdown("### Policy drift")
    metrics = S.load_scheduler_metrics()
    if metrics.empty or "drift_mean_abs_z" not in metrics:
        st.info("No drift metrics available.")
        return
    id_d = metrics[metrics["split"] == "id"].set_index("scheduler")["drift_mean_abs_z"]
    ood_d = metrics[metrics["split"] == "ood"].set_index("scheduler")["drift_mean_abs_z"]

    def drift_col(v):
        if v < 0.3:
            return P["green"]
        if v <= 0.8:
            return P["amber"]
        return P["red"]

    rows = ""
    for s in SCHED_ORDER:
        if s not in id_d.index and s not in ood_d.index:
            continue
        iv = float(id_d.get(s, np.nan))
        ov = float(ood_d.get(s, np.nan))
        rows += (
            f"<tr><td class='txt'>{SCHED_DISPLAY.get(s, s)}</td>"
            f"<td style='color:{drift_col(iv)}'>{iv:.2f}</td>"
            f"<td style='color:{drift_col(ov)}'>{ov:.2f}</td></tr>"
        )
    st.markdown(
        '<table class="hr-tbl"><thead><tr><th>Scheduler</th>'
        "<th>ID drift (mean |z-shift|)</th><th>OOD drift</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        "High drift means the scheduler visits chip states different from what the calibrator was "
        "trained on. Green < 0.3, amber 0.3–0.8, red > 0.8."
    )


def section_decision_table() -> None:
    st.markdown("### Full 16-core decision table")
    if not S.bundle_guard():
        return
    cfg = S.replay_cfg()
    seed = int(st.session_state.get("episode_seed", S.DEFAULT_SEED))
    focus_name = st.session_state.get("focus_sched", PRIMARY_SCHED)
    split = "ood" if st.session_state.get("mode") == "OOD shift" else "id"
    t = int(st.session_state.get("t", 0))

    frames = S.build_replay(split, focus_name, seed, with_scores=True)
    t = max(0, min(t, len(frames) - 1))
    dec_idx = S.last_decision_index(frames, t)
    if dec_idx is None:
        st.info(f"No scheduling decision has occurred yet at T={t} for the current replay.")
        return
    frame = frames[dec_idx]
    scores = frame.get("conformal_scores")
    selected = None if pd.isna(frame["selected_core"]) else int(frame["selected_core"])
    loads = np.array(frame["load"], dtype=float)

    if scores is None:
        order = np.argsort(loads)
        rows = ""
        for rank, core in enumerate(order, start=1):
            sel = core == selected
            bg = ' style="background:rgba(245,158,11,.16)"' if sel else ""
            rows += (
                f"<tr{bg}><td>{rank}</td><td>{core}</td><td>—</td>"
                f"<td>{loads[core]:.2f}</td><td class='txt'>{'● selected' if sel else ''}</td></tr>"
            )
    else:
        arr = np.array(scores, dtype=float)
        order = np.argsort(arr)
        rows = ""
        for rank, core in enumerate(order, start=1):
            sel = core == selected
            bg = ' style="background:rgba(245,158,11,.18)"' if sel else ""
            col = P["red"] if arr[core] >= cfg.thermal_limit else (P["amber"] if arr[core] >= 70 else P["text"])
            rows += (
                f"<tr{bg}><td>{rank}</td><td>{core}</td>"
                f"<td style='color:{col}'>{arr[core]:.1f}</td>"
                f"<td>{loads[core]:.2f}</td>"
                f"<td class='txt'>{'● selected' if sel else ''}</td></tr>"
            )
    st.markdown(
        '<table class="hr-tbl"><thead><tr><th>Rank</th><th>Core</th>'
        "<th>Calibrated upper bound (°C)</th><th>Current load</th><th>Selected?</th>"
        f"</tr></thead><tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    note = (f"Most recent decision from the replay (scheduler "
            f"{SCHED_DISPLAY.get(focus_name, focus_name)}, {split.upper()}, seed {seed}, T={frame['timestep']}). "
            "Updates live if the replay is running on the Watch It Run page.")
    st.caption(note)


def section_conservatism() -> None:
    st.markdown("### Bound conservatism")
    c = S.bound_conservatism_c()
    val = "n/a" if c is None else f"+{c:.2f}°C"
    st.markdown(
        f"<div class='hr-card hr-bigcard' style='max-width:360px'>"
        f"<div class='t'>Average bound conservatism (ID)</div>"
        f"<div class='v' style='color:{P['amber']}'>{val}</div></div>",
        unsafe_allow_html=True,
    )
    if c is not None:
        st.caption(
            f"Calibrated upper bounds are on average {c:.2f}°C above realized outcomes. Conservative "
            "enough for reliable coverage, tight enough for meaningful differentiation between cores."
        )


def section_model_performance() -> None:
    st.markdown("### Model performance")
    mm = S.load_model_metrics()
    if mm.empty:
        st.info("No model metrics found.")
        return
    # Prefer the test_id split for headline accuracy; fall back to calibration.
    split = "test_id" if (mm["split"] == "test_id").any() else "calibration"
    sub = mm[mm["split"] == split]
    model_rows = [
        ("linear_point", "Linear baseline"),
        ("forest_point", "Random forest (point)"),
        ("quantile_upper", "Quantile upper (GBR)"),
        ("conformal_upper", "Conformal upper"),
    ]

    def val(model, metric):
        r = sub[(sub["model"] == model) & (sub["metric"] == metric)]["value"]
        return None if r.empty else float(r.iloc[0])

    def fmt(v, suffix=""):
        return "—" if v is None else f"{v:.2f}{suffix}"

    rows = ""
    for model, label in model_rows:
        mae, rmse, mx = val(model, "mae"), val(model, "rmse"), val(model, "max_abs_error")
        cov = val(model, "empirical_coverage")
        rows += (
            f"<tr><td class='txt'>{label}</td>"
            f"<td>{fmt(mae)}</td><td>{fmt(rmse)}</td><td>{fmt(mx)}</td>"
            f"<td>{'—' if cov is None else pct(cov)}</td></tr>"
        )
    st.markdown(
        '<table class="hr-tbl"><thead><tr><th>Model</th><th>MAE</th><th>RMSE</th>'
        "<th>Max error</th><th>Coverage</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    st.caption(
        f"Reported on the {split.replace('_', ' ')} split. MAE/RMSE/Max error in °C; coverage applies "
        "to the upper-bound models only (point models have no coverage notion)."
    )


def main() -> None:
    st.sidebar.markdown("### Under the Hood")
    st.sidebar.caption("Methodology for reviewers.")

    st.title("Under the Hood")
    section_coverage()
    st.markdown("---")
    section_drift()
    st.markdown("---")
    section_decision_table()
    st.markdown("---")
    section_conservatism()
    st.markdown("---")
    section_model_performance()


main()
