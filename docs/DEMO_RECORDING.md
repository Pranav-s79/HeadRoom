# HeadRoom — 60-Second Demo Recording Script

A repeatable script for capturing the portfolio demo (`portfolio_assets/demo.mp4`
or `demo.gif`). Total runtime target: **~60 seconds**. Record at 1080p or higher,
dark room, browser zoomed so the dual heatmap fills most of the frame.

## Before recording

1. Make sure the data is present (either `unzip -o outputs.zip` or
   `python run_all.py --quick`, or use the demo build with `demo_data/`).
2. Launch the dashboard:
   ```bash
   streamlit run run_dashboard.py        # full pipeline outputs
   # or, for the lightweight tracked dataset:
   streamlit run run_dashboard_demo.py
   ```
3. Set the browser to a clean window (hide bookmarks bar). Light/dark OS theme does
   not matter — the app is fully dark.
4. Start your screen recorder (e.g. ScreenToGif, OBS, or the macOS/Windows built-in).

## The 60-second take

| Time | Action | What the viewer sees |
|---|---|---|
| 0:00–0:05 | Land on **Watch It Run**. Pause on the top bar. | The hero page: top bar (T, peak temp, violations, scheduler), dual heatmaps. |
| 0:05–0:08 | Click **▶ Run Demo** in the sidebar. | Seed switches to the most contrastful OOD episode, speed jumps to 2x, replay auto-plays from T=0. |
| 0:08–0:30 | Let the replay run. | Right (coolest-core) heatmap heats up and cores cross 85 °C with red borders; left (conformal) stays cool. The max-temp chart diverges. |
| ~0:18 | The amber toast fires automatically. | "Coolest-core triggered a hotspot at T=X. Conformal scheduler avoided it." Pause ~2 s to let it read. |
| 0:30–0:40 | Click **What We Found** in the sidebar. | Before/after conformal cards (59% → 90%), comparison table, peak-temp + violations bars. Scroll slowly. |
| 0:40–0:52 | Click **Under the Hood**. | Coverage cards: ID green, OOD red. Scroll past the 16-core decision table. |
| 0:52–1:00 | Click **About** (optional) or end on the coverage cards. | The honest-claims card. Fade out. |

## Tips
- If the toast moment feels too fast, drop the speed pill to **1x** before clicking
  Run Demo, or step with **⏭ Step fwd** around the violation timestep for a clean still.
- For a GIF, trim to the 0:05–0:30 window (Run Demo → toast) — that single clip is
  the strongest standalone asset.
- A good still frame for thumbnails: the moment the right heatmap has multiple
  red-bordered cores while the left has none. `portfolio_assets/heatmap_demo.png`
  is exactly this frame, pre-rendered by `python dashboard/generate_assets.py`.

## Saving
Save the final recording as `portfolio_assets/demo.mp4` (preferred) or
`portfolio_assets/demo.gif`. Keep it under ~15 MB for GitHub/LinkedIn embeds; a
GIF of the 0:05–0:30 window at 12–15 fps and 720p usually lands well under that.
