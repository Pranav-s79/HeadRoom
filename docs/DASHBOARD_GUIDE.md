# Dashboard Guide

## Launch

```bash
streamlit run dashboard/app.py
```

or:

```bash
python run_dashboard.py
```

## Recommended Workflow

1. Open `Guided Tutorial`.
2. Open `Simulation Replay` with Beginner mode enabled.
3. Use jump buttons to move to the first decision, hottest timestep, or first
   conformal violation if one exists.
4. Use `Why this core?` and the candidate table to explain the selected core.
5. Open `Metric Explainer` before interpreting CSV columns.
6. Open `Result Verdict` before making claims.
7. Use `Pitch Prep` for a concise interview or portfolio explanation.

## Page Summary

- `Simulation Replay`: Local replay with heatmaps, selected core, predictions,
  bounds, actual future peak, candidate table, and replay logging.
- `Guided Tutorial`: Beginner-friendly project walkthrough.
- `Pitch Prep`: Short pitch, supported claims, unsupported claims, and likely
  questions.
- `Data Flow`: How simulator data becomes features, labels, models, metrics,
  figures, and dashboard views.
- `Metric Explainer`: Plain-English metric definitions.
- `Result Verdict`: Current evidence and honest claim status.
- `Results Explorer`: Raw scheduler, model, and coverage tables.
- `Calibration View`: Conformal correction and coverage before/after
  calibration.
- `Scheduler Comparison`: Direct model-scheduler comparison.
- `Heatmap Comparison`: Saved heatmap figures and raw snapshot grids.
- `Stress and Multiseed`: Stress sweep and multiseed summaries.

## Missing Files

The dashboard is designed to degrade gracefully. If a CSV, model bundle, or
figure is missing, the relevant page shows a message instead of failing.
