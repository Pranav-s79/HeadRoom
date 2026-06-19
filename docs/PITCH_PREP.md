# ThermalGuard-Cal Pitch Prep

## 30-Second Pitch

ThermalGuard-Cal is a research MVP for thermal-aware task scheduling on a
simulated 4x4 many-core chip. It compares simple heuristics, sparse-sensor
thermal baselines, learned future-temperature models, and conformal calibrated
upper bounds. The project is valuable because it makes safety claims measurable:
it reports peak temperature, hotspot violations, candidate coverage,
selected-core coverage, and distribution drift.

## 2-Minute Walkthrough

1. Start on the dashboard `Guided Tutorial` page.
2. Show `Simulation Replay` in Beginner mode.
3. Explain that the true heatmap is simulator reality, while the sensor heatmap
   is what the deployable scheduler can observe.
4. Show point, quantile, and conformal upper-bound heatmaps.
5. Use `Why this core?` to explain the selected placement.
6. Move to `Calibration View` and show before/after conformal coverage.
7. Move to `Result Verdict` and state what is supported and unsupported.

## Supported Claims

- The simulator, dataset, model, conformal calibration, scheduler evaluation,
  plots, and dashboard run end to end.
- Sparse-sensor heuristics can fail under harder workloads.
- Model-based schedulers can reduce hotspots in several evaluated settings.
- Conformal calibration improves or verifies coverage on calibration-like data.
- OOD coverage can drop below nominal, so the dashboard does not overclaim OOD
  guarantees.

## Unsupported Claims

- Real silicon validation.
- Formal OOD safety guarantee.
- Production scheduler optimality.
- HotSpot/OpenROAD-level physical accuracy.
- Hardware implementation readiness.

## Interview Questions

### Why use conformal prediction?

Point predictions can underestimate risk. A calibrated upper bound provides a
measurable coverage target on calibration-like data.

### What is selected-core coverage?

Coverage after the scheduler chooses one core. It matters because the scheduler
selects one candidate from 16, and that selection step can change realized
coverage.

### Why include an oracle?

The oracle reads true simulator temperatures. It is a reference baseline, not a
deployable scheduler.

### What is the main limitation?

The simulator is useful for controlled experiments, but it is not a validated
chip thermal model.
