"""Page 2 — "What We Found": the 30-second results summary.

Section 1  the conformal correction (hero before/after cards)
Section 2  scheduler comparison table (ID/OOD toggle)
Section 3  peak-temp + violations horizontal bars
Section 4  safety-vs-throughput scatter
"""

from __future__ import annotations

import streamlit as st

import shared as S
import figures as F

P = S.P
PRIMARY_SCHED = S.PRIMARY_SCHED
SCHED_DISPLAY = S.SCHED_DISPLAY
SCHED_NOTE = S.SCHED_NOTE
SCHED_ORDER = S.SCHED_ORDER


def _metric_columns(df) -> tuple[str, str, str]:
    """Return (peak, violations, completed) column names, preferring multi-seed means."""
    if "peak_temperature_mean" in df.columns:
        return "peak_temperature_mean", "hotspot_violations_mean", "completed_tasks_mean"
    return "peak_temperature", "hotspot_violations", "completed_tasks"


def pct(v):
    return "n/a" if v is None else f"{v * 100:.1f}%"


def section_conformal_correction() -> None:
    st.markdown("### The conformal correction")
    ba = S.conformal_before_after()
    before, after, corr = ba["before"], ba["after"], ba["correction"]
    bcol = P["red"] if before < S.NOMINAL_COVERAGE - 0.005 else P["amber"]
    acol = P["green"] if after >= S.NOMINAL_COVERAGE - 0.005 else P["red"]

    c1, c2, c3 = st.columns([1, 0.22, 1])
    with c1:
        st.markdown(
            f"<div class='hr-card hr-bigcard' style='border-color:{P['red']}'>"
            f"<div class='t'>Before calibration</div>"
            f"<div class='v' style='color:{bcol}'>{pct(before)}</div>"
            f"<div class='t' style='text-transform:none'>base upper-quantile coverage</div></div>",
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"<div style='text-align:center;color:{P['amber']};font-size:44px;margin-top:34px'>→</div>"
            f"<div style='text-align:center;color:{P['amber']};font-family:var(--hr-mono);font-size:15px'>"
            f"+{corr:.1f}°C</div>",
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"<div class='hr-card hr-bigcard' style='border-color:{P['green']}'>"
            f"<div class='t'>After calibration (+{corr:.1f}°C)</div>"
            f"<div class='v' style='color:{acol}'>{pct(after)}</div>"
            f"<div class='t' style='text-transform:none'>calibrated upper-bound coverage</div></div>",
            unsafe_allow_html=True,
        )

    note = (
        f"The conformal step widened upper bounds by {corr:.1f}°C on average, bringing empirical "
        f"coverage from {pct(before)} to the {S.NOMINAL_COVERAGE*100:.0f}% target on in-distribution "
        f"calibration data ({ba['source']})."
    )
    if ba.get("canonical_zero"):
        note += (" Conformal only ever widens bounds — when the base quantile model is already "
                 "conservative the correction is +0.0°C, which is why this story is shown from the "
                 "challenging-preset seed where the base model under-covered.")
    st.markdown(f"<p style='color:{P['muted']};font-size:14px'>{note}</p>", unsafe_allow_html=True)


def section_table(split: str) -> None:
    ms = S.load_multiseed_metrics()
    use_ms = not ms.empty
    df = (ms if use_ms else S.load_scheduler_metrics()).copy()
    df = df[df["split"] == split].copy()
    if df.empty:
        st.info("No scheduler metrics found.")
        return
    pcol, vcol, ccol = _metric_columns(df)
    df["order"] = df["scheduler"].map({s: i for i, s in enumerate(SCHED_ORDER)})
    df = df.sort_values("order")

    peaks = df[pcol].astype(float)
    viols = df[vcol].astype(float)
    pbest, pworst = peaks.min(), peaks.max()
    vbest, vworst = viols.min(), viols.max()

    def std_suffix(row, base):
        scol = f"{base}_std"
        if use_ms and scol in row and row[scol] == row[scol]:
            return f" <span style='color:{P['muted']};font-size:11px'>±{float(row[scol]):.1f}</span>"
        return ""

    rows = ""
    for _, r in df.iterrows():
        name = r["scheduler"]
        is_conf = name == PRIMARY_SCHED
        pc = S.color_best_worst(float(r[pcol]), pbest, pworst)
        vc = S.color_best_worst(float(r[vcol]), vbest, vworst)
        weight = "font-weight:700" if is_conf else ""
        edge = ' style="border-left:3px solid #f59e0b"' if is_conf else ""
        rows += (
            f"<tr{edge}>"
            f"<td class='txt' style='{weight}'>{SCHED_DISPLAY.get(name, name)}</td>"
            f"<td style='color:{pc};{weight}'>{float(r[pcol]):.1f}{std_suffix(r, 'peak_temperature')}</td>"
            f"<td style='color:{vc};{weight}'>{float(r[vcol]):.0f}{std_suffix(r, 'hotspot_violations')}</td>"
            f"<td>{float(r[ccol]):.0f}</td></tr>"
        )
    st.markdown(
        '<table class="hr-tbl"><thead><tr><th>Scheduler</th><th>Peak temp (°C)</th>'
        "<th>Violations</th><th>Tasks done</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>",
        unsafe_allow_html=True,
    )
    src = "mean ± std across workload seeds" if use_ms else "single canonical seed"
    st.caption(
        f"Values: {src}. Conformal upper-bound row bolded. "
        "Coolest-Core Oracle uses true temperatures unavailable to a real sparse-sensor system — "
        "labeled as a privileged baseline."
    )


def section_bars(split: str) -> None:
    ms = S.load_multiseed_metrics()
    use_ms = not ms.empty
    df = (ms if use_ms else S.load_scheduler_metrics()).copy()
    df = df[df["split"] == split].copy()
    if df.empty:
        return
    pcol, vcol, _ = _metric_columns(df)

    c1, c2 = st.columns(2)
    with c1:
        order_p = df.sort_values(pcol)
        fig = F.hbar_figure(
            [SCHED_DISPLAY.get(s, s) for s in order_p["scheduler"]],
            order_p[pcol].astype(float).tolist(),
            title="Peak temperature by scheduler", xlabel="Peak temp (°C)", color=P["amber"],
            reference=85.0, reference_label="85°C limit", highlight=SCHED_DISPLAY[PRIMARY_SCHED],
        )
        st.pyplot(fig); F.plt.close(fig)
    with c2:
        order_v = df.sort_values(vcol)
        fig = F.hbar_figure(
            [SCHED_DISPLAY.get(s, s) for s in order_v["scheduler"]],
            order_v[vcol].astype(float).tolist(),
            title="Hotspot violations by scheduler", xlabel="Violations (timesteps over limit)",
            color=P["red"], value_fmt="{:.0f}", highlight=SCHED_DISPLAY[PRIMARY_SCHED],
        )
        st.pyplot(fig); F.plt.close(fig)


def section_scatter(split: str) -> None:
    ms = S.load_multiseed_metrics()
    use_ms = not ms.empty
    df = (ms if use_ms else S.load_scheduler_metrics()).copy()
    df = df[df["split"] == split].copy()
    if df.empty:
        return
    pcol, _, ccol = _metric_columns(df)
    rows = [
        {
            "label": SCHED_DISPLAY.get(r["scheduler"], r["scheduler"]),
            "completed": int(float(r[ccol])),
            "peak": float(r[pcol]),
            "is_conformal": r["scheduler"] == PRIMARY_SCHED,
            "is_oracle": r["scheduler"] == "coolest_core_oracle_true_temp",
        }
        for _, r in df.iterrows()
    ]
    fig = F.safety_throughput_figure(rows, thermal_limit=85.0)
    st.pyplot(fig); F.plt.close(fig)


def main() -> None:
    st.sidebar.markdown("### What We Found")
    st.sidebar.caption("Did it work? The 30-second answer.")

    st.title("What We Found")
    section_conformal_correction()
    st.markdown("---")

    split = st.radio(
        "Workload split", ["ood", "id"], horizontal=True, key="found_split",
        format_func=lambda s: "OOD shift" if s == "ood" else "In-distribution",
    )
    st.markdown("### Scheduler comparison")
    section_table(split)
    st.markdown("### Peak temperature & violations")
    section_bars(split)
    st.markdown("### Safety vs throughput")
    section_scatter(split)


main()
