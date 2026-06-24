# Supported Claims

- The project runs an end-to-end simulated thermal scheduling pipeline.
- The model target is future peak whole-chip temperature over a prediction
  horizon.
- The dashboard can replay simulator decisions and show true state, sparse
  sensors, selected core, model predictions, and conformal bounds.
- Conformal diagnostics explicitly report target coverage, correction,
  calibration sample count, and empirical coverage before/after calibration.
- Stress and multiseed artifacts are available under `outputs/reports` and
  `outputs/figures`.
- The project reports limitations instead of claiming unconditional safety.
