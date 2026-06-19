# ThermalGuard-Cal Tutorial

## Goal

ThermalGuard-Cal studies how task placement affects temperature on a simulated
4x4 many-core chip. The core question is whether learned future-temperature
risk estimates can help a scheduler avoid thermal hotspots better than simple
or sparse-sensor heuristics.

## What Was Built

The project includes:

- A stochastic thermal simulator with heat gain, cooling, diffusion, inertia,
  and noise.
- Sparse noisy sensors placed on a subset of cores.
- Workload generators for in-distribution, out-of-distribution, challenging,
  and stress-style experiments.
- Action-conditioned datasets where each task/core candidate becomes one row.
- Point prediction, upper-quantile, and conformal upper-bound models.
- Scheduler baselines and model-based schedulers.
- Reports, plots, stress sweeps, multiseed summaries, and a Streamlit dashboard.

## How To Learn The Dashboard

1. Open `Guided Tutorial`.
2. Read `Data Flow` to understand what data enters the model.
3. Open `Simulation Replay` with Beginner mode on.
4. Click `Jump to first decision`.
5. Read `What is happening now?`.
6. Compare the true heatmap, sensor heatmap, and conformal upper-bound heatmap.
7. Read `Why this core?`.
8. Open `Result Verdict` before making claims about the result.

## Important Mental Model

The model predicts future peak chip temperature for each candidate placement.
It does not predict only the candidate core temperature. The scheduler uses
those candidate-level risk estimates to choose one placement.

## What ID And OOD Mean

ID means the workload is drawn from the same kind of distribution used for
training and calibration. OOD means the workload is shifted toward hotter,
burstier behavior and different sensor reliability. Conformal calibration is
most defensible on calibration-like data, not under arbitrary OOD shift.

## What To Say Carefully

Say: "The dashboard shows when conformal calibration improves coverage on
calibration-like data and when OOD behavior weakens that claim."

Do not say: "This proves guaranteed thermal safety on real hardware."
