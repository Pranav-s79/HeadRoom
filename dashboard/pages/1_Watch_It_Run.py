"""Page 1 — "Watch It Run": the hero replay page.

One story at a time: a live side-by-side heatmap replay of the selected scheduler
versus the coolest-core baseline, transport controls, a single "why this core?"
sentence, and the max-temperature-over-time chart. No tables, no calibration math.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import streamlit as st

import shared as S
import figures as F

P = S.P
PRIMARY_SCHED = S.PRIMARY_SCHED
CONTRAST_SCHED = S.CONTRAST_SCHED
SCHED_DISPLAY = S.SCHED_DISPLAY
SPEED_OPTIONS = S.SPEED_OPTIONS

# Seeds scanned by the one-click demo button to find the most contrastful episode.
DEMO_CANDIDATE_SEEDS = (S.DEMO_SEED, 17, 7, 11, 19, 23, 29, 31, 41, 47)


# --------------------------------------------------------------------------- #
# Sidebar controls (Page 1 only)
# --------------------------------------------------------------------------- #
def _step(delta: int, total: int) -> None:
    st.session_state["playing"] = False
    st.session_state["_demo_active"] = False
    st.session_state["t"] = max(0, min(total - 1, int(st.session_state.get("t", 0)) + delta))


def _toggle_play() -> None:
    st.session_state["playing"] = not st.session_state.get("playing", False)
    if not st.session_state["playing"]:
        st.session_state["_demo_active"] = False


def _set_speed(label: str) -> None:
    st.session_state["speed"] = label


def _run_demo() -> None:
    choice = S.best_demo_seed(DEMO_CANDIDATE_SEEDS)
    st.session_state["mode"] = "OOD shift"
    st.session_state["speed"] = "2x"
    st.session_state["focus_sched"] = PRIMARY_SCHED
    st.session_state["episode_seed"] = int(choice["seed"])
    st.session_state["t"] = 0
    st.session_state["playing"] = True
    st.session_state["_demo_active"] = True
    st.session_state["_demo_toast_shown"] = False
    st.session_state["_demo_choice"] = choice


def render_sidebar() -> dict:
    sb = st.sidebar
    sb.markdown("### Replay")
    seed = sb.number_input("Episode seed", min_value=0, max_value=9999, step=1, key="episode_seed")
    available = [s for s in S.SCHED_ORDER if S.get_bundle() is not None or s not in S.MODEL_SCHEDS]
    focus = sb.selectbox(
        "Scheduler (left heatmap)", available,
        format_func=lambda s: SCHED_DISPLAY.get(s, s), key="focus_sched",
    )
    sb.selectbox("Speed", list(SPEED_OPTIONS), key="speed")
    sb.radio("Workload mode", ["In-distribution", "OOD shift"], key="mode")

    # One-click demo: pre-identified contrastful seed, 2x, autoplay from T=0.
    choice = st.session_state.get("_demo_choice")
    if choice is not None and not choice["has_violation_contrast"]:
        demo_help = ("Increase OOD stress for stronger contrast — no seed in the current dataset "
                     "drives the coolest-core baseline over the limit, so the demo shows the "
                     "largest peak-temp gap instead.")
    else:
        demo_help = ("Auto-loads the episode where the naive coolest-core scheduler overheats and "
                     "the conformal scheduler holds headroom, at 2x speed from T=0.")
    sb.button("▶ Run Demo", on_click=_run_demo, use_container_width=True, type="primary", help=demo_help)

    sb.markdown("### Export")
    save_frame = sb.button("Save current frame", use_container_width=True,
                           help=f"Writes a PNG under {S.PORTFOLIO_DIR.name}/")

    split = "ood" if st.session_state.get("mode") == "OOD shift" else "id"
    return {"seed": int(seed), "focus": focus, "split": split, "save_frame": save_frame}


# --------------------------------------------------------------------------- #
# Top bar (always visible, ~60px)
# --------------------------------------------------------------------------- #
def render_top_bar(t: int, total: int, focus_frames: list[dict], split: str, focus_name: str, cfg) -> None:
    frame = focus_frames[t]
    peak = float(frame["max_temperature"])
    peak_col = S.color_for_temp(peak, cfg.thermal_limit)
    viols = S.cumulative_violations(focus_frames, t, cfg.thermal_limit)
    viol_col = P["red"] if viols else P["green"]
    viol_badge = (
        f"<span class='hr-badge' style='background:rgba(239,68,68,.18);color:{P['red']}'>{viols}</span>"
        if viols else f"<span style='color:{P['green']}'>0</span>"
    )

    cells = [
        ("Episode timestep", f"T = {t} / {total - 1}", P["text"], "live simulation replay"),
        ("Peak chip temp", f"{peak:.1f}°C", peak_col, f"limit {cfg.thermal_limit:g}°C"),
        ("Violations so far", viol_badge, viol_col, "timesteps over the limit"),
        ("Active scheduler", SCHED_DISPLAY.get(focus_name, focus_name), P["amber"], "left heatmap"),
    ]
    html = '<div class="hr-topbar">'
    for lbl, val, col, sub in cells:
        html += (
            f'<div class="hr-cell"><div class="lbl">{lbl}</div>'
            f'<div class="val" style="color:{col}">{val}</div>'
            f'<div class="sub">{sub}</div></div>'
        )
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Replay body
# --------------------------------------------------------------------------- #
def render_heatmaps(focus_name, contrast_name, ff, cf, cfg) -> None:
    fig = F.dual_heatmap_figure(
        ff["temperatures"], cf["temperatures"], grid_size=cfg.grid_size,
        left_title=SCHED_DISPLAY.get(focus_name, focus_name),
        right_title=SCHED_DISPLAY.get(contrast_name, contrast_name),
        left_subtitle="calibrated upper-bound scheduler" if focus_name == PRIMARY_SCHED else "selected scheduler",
        right_subtitle="naive sparse-sensor baseline",
        left_selected=None if pd.isna(ff["selected_core"]) else int(ff["selected_core"]),
        right_selected=None if pd.isna(cf["selected_core"]) else int(cf["selected_core"]),
        sensor_indices=cfg.sensor_indices,
        left_mask=ff["sensor_mask"], right_mask=cf["sensor_mask"],
        thermal_limit=cfg.thermal_limit,
    )
    st.pyplot(fig)
    F.plt.close(fig)
    st.caption(
        f"Fixed 35–90°C scale on both panels. Amber ○ = sensor cores (4 of 16), "
        f"amber ◆ = core assigned this step, red border = core over {cfg.thermal_limit:g}°C limit. "
        f"Same workload + seed on both — only the scheduler differs."
    )


def render_transport(total: int) -> int:
    cprev, cplay, cnext = st.columns([1, 1.4, 1])
    cprev.button("⏮ Step back", use_container_width=True, on_click=_step, args=(-1, total))
    play_label = "⏸ Pause" if st.session_state["playing"] else "▶ Play"
    cplay.button(play_label, use_container_width=True, type="primary", on_click=_toggle_play)
    cnext.button("⏭ Step fwd", use_container_width=True, on_click=_step, args=(1, total))

    # pill-style speed toggle
    scols = st.columns([1.2, 1, 1, 1, 1, 1.2])
    scols[0].markdown("<div style='text-align:right;color:#9ca3af;padding-top:6px'>Speed</div>",
                      unsafe_allow_html=True)
    for i, label in enumerate(SPEED_OPTIONS):
        active = st.session_state["speed"] == label
        scols[i + 1].button(label, use_container_width=True,
                            type="primary" if active else "secondary",
                            on_click=_set_speed, args=(label,), key=f"speed_pill_{label}")

    st.slider("Timestep", 0, total - 1, key="t", label_visibility="collapsed")
    st.progress((int(st.session_state["t"]) + 1) / total)
    return int(st.session_state["t"])


def render_why_panel(focus_frames: list[dict], t: int, focus_name: str, cfg) -> None:
    """One sentence only. The full 16-core table lives on Page 3."""
    st.markdown("##### Why this core?")
    dec_idx = S.last_decision_index(focus_frames, t)
    current_decision = dec_idx is not None and focus_frames[dec_idx]["timestep"] == t
    if dec_idx is None or not current_decision:
        st.markdown(
            f"<div class='hr-finding' style='border-left-color:{P['border']}'>"
            f"No task arrived at T={t}. Thermal state advancing.</div>",
            unsafe_allow_html=True,
        )
        return

    dec_frame = focus_frames[dec_idx]
    sel = int(dec_frame["selected_core"])
    scores = dec_frame.get("conformal_scores")
    if scores is not None:
        arr = np.array(scores, dtype=float)
        st.markdown(
            f"<div class='hr-finding'>Core <b>{sel}</b> selected — lowest calibrated thermal risk "
            f"(<b>{arr[sel]:.1f}°C</b> upper bound) among lightly-loaded candidates.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div class='hr-finding'>Core <b>{sel}</b> selected by the "
            f"{SCHED_DISPLAY.get(focus_name, focus_name)} rule "
            f"(this scheduler has no calibrated bound to display).</div>",
            unsafe_allow_html=True,
        )


def render_chart(focus_frames, contrast_frames, t, focus_name, contrast_name, cfg) -> None:
    st.markdown("##### Max chip temperature over time")
    ts = [fr["timestep"] for fr in focus_frames[: t + 1]]
    fig = F.max_temp_chart_figure(
        ts,
        [fr["max_temperature"] for fr in focus_frames[: t + 1]],
        [fr["max_temperature"] for fr in contrast_frames[: t + 1]],
        conformal_label=SCHED_DISPLAY.get(focus_name, focus_name),
        coolest_label=SCHED_DISPLAY.get(contrast_name, contrast_name),
        thermal_limit=cfg.thermal_limit, current_t=t,
    )
    st.pyplot(fig)
    F.plt.close(fig)
    st.caption(
        "The single clearest view of whether the calibrated scheduler keeps the chip cooler. "
        "The naive baseline drifts toward and past the 85°C limit under OOD load."
    )


def maybe_demo_toast(contrast_frames, t, focus_name, contrast_name, cfg) -> None:
    if st.session_state.get("_demo_active") and not st.session_state.get("_demo_toast_shown"):
        viol_t = S.first_over_limit_timestep(contrast_frames, cfg.thermal_limit)
        if viol_t is not None and t >= viol_t:
            st.toast(
                f"⚠ {SCHED_DISPLAY[contrast_name]} triggered a hotspot at T={viol_t}. "
                f"{SCHED_DISPLAY.get(focus_name, focus_name)} avoided it.",
                icon="🔥",
            )
            st.session_state["_demo_toast_shown"] = True


def save_frame_export(focus_name, contrast_name, ff, cf, split, t, cfg) -> None:
    S.PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)
    fig = F.dual_heatmap_figure(
        ff["temperatures"], cf["temperatures"], grid_size=cfg.grid_size,
        left_title=SCHED_DISPLAY.get(focus_name, focus_name),
        right_title=SCHED_DISPLAY.get(contrast_name, contrast_name),
        left_subtitle="calibrated upper-bound scheduler", right_subtitle="naive sparse-sensor baseline",
        left_selected=None if pd.isna(ff["selected_core"]) else int(ff["selected_core"]),
        right_selected=None if pd.isna(cf["selected_core"]) else int(cf["selected_core"]),
        sensor_indices=cfg.sensor_indices, left_mask=ff["sensor_mask"], right_mask=cf["sensor_mask"],
        thermal_limit=cfg.thermal_limit,
        suptitle=f"HeadRoom — {split.upper()} replay  ·  T = {t}",
    )
    path = S.PORTFOLIO_DIR / f"heatmap_frame_T{t}.png"
    fig.savefig(path, dpi=130, bbox_inches="tight")
    F.plt.close(fig)
    st.sidebar.success(f"Saved {path.name}")


# --------------------------------------------------------------------------- #
# Page entry
# --------------------------------------------------------------------------- #
def main() -> None:
    if not S.bundle_guard():
        return
    cfg = S.replay_cfg()
    controls = render_sidebar()
    seed, focus_name, split = controls["seed"], controls["focus"], controls["split"]
    contrast_name = CONTRAST_SCHED if focus_name != CONTRAST_SCHED else PRIMARY_SCHED

    # Reset timestep when the replay identity changes.
    replay_key = f"{seed}:{focus_name}:{split}"
    if st.session_state["_replay_key"] != replay_key:
        st.session_state["_replay_key"] = replay_key
        st.session_state["t"] = 0

    focus_frames = S.build_replay(split, focus_name, seed, with_scores=True)
    contrast_frames = S.build_replay(split, contrast_name, seed, with_scores=False)
    total = len(focus_frames)

    # Auto-advance one frame per run while playing — BEFORE the slider widget is made.
    if st.session_state.get("playing"):
        nxt = min(total - 1, max(0, int(st.session_state.get("t", 0))) + 1)
        st.session_state["t"] = nxt
        if nxt >= total - 1:
            st.session_state["playing"] = False
            st.session_state["_demo_active"] = False
    st.session_state["t"] = max(0, min(int(st.session_state.get("t", 0)), total - 1))
    t = int(st.session_state["t"])

    render_top_bar(t, total, focus_frames, split, focus_name, cfg)
    render_heatmaps(focus_name, contrast_name, focus_frames[t], contrast_frames[t], cfg)
    t = render_transport(total)
    render_why_panel(focus_frames, t, focus_name, cfg)
    render_chart(focus_frames, contrast_frames, t, focus_name, contrast_name, cfg)
    maybe_demo_toast(contrast_frames, t, focus_name, contrast_name, cfg)

    if controls["save_frame"]:
        save_frame_export(focus_name, contrast_name, focus_frames[t], contrast_frames[t], split, t, cfg)

    # Auto-play pacing.
    if st.session_state["playing"] and t < total - 1:
        time.sleep(S.BASE_FRAME_DELAY / SPEED_OPTIONS[st.session_state["speed"]])
        st.rerun()


main()
