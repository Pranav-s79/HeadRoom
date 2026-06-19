# ThermalGuard-Cal

ThermalGuard-Cal is a modular Python research MVP for conformal upper-bound
thermal scheduling on a simulated 4x4 many-core chip. It compares random,
round-robin, observed coolest-core, oracle coolest-core, trend-aware,
point-prediction, uncalibrated quantile, and conformal upper-bound schedulers.

The model target is **future peak chip temperature over the prediction
horizon**, a global whole-chip quantity. It is not the candidate core's own
future temperature. Selected-core coverage therefore means coverage of the
global outcome that results from a placement choice.

## Why Calibrated Upper Bounds

Point predictions can underestimate thermal risk. ThermalGuard-Cal trains an
upper-quantile model and wraps it with one-sided conformal calibration, then
uses calibrated upper bounds to choose task placements. The project separately
measures marginal candidate coverage, selected-core coverage after choosing one
candidate out of 16, and policy-induced feature distribution drift.

## Install

```bash
python -m pip install -r requirements.txt
```

## Quick Mode

Quick mode is the default and is intended for development and smoke tests.

```bash
python run_generate_data.py
python run_train_models.py
python run_evaluate_schedulers.py
```

Or run the whole pipeline:

```bash
python run_all.py --quick
```

The default preset is `normal`, which preserves the original quick-mode behavior:

```bash
python run_all.py --quick --preset normal
```

## Challenge Presets

Preset options are `easy`, `normal`, `challenging`, and `stress`. The
`challenging` preset is the recommended research-hardening setting: it increases
reasonable workload and simulator stress so sparse-sensor heuristic schedulers
can overheat while model-based schedulers remain comparable.

```bash
python run_all.py --quick --preset challenging
python run_make_plots.py --preset challenging
```

Stress-sweep screening runs compact generate/train/evaluate loops for preset
candidates and writes `outputs/reports/stress_sweep_results.csv`,
`outputs/reports/stress_sweep_recommendations.csv`, and
`outputs/reports/stress_sweep_summary.md`:

```bash
python run_stress_sweep.py
```

Five-seed challenging validation writes aggregate CSV/Markdown and figures:

```bash
python run_multiseed.py --preset challenging --quick --seeds 1 2 3 4 5
```

Primary multiseed outputs:

- `outputs/reports/multiseed_metrics_challenging.csv`
- `outputs/reports/multiseed_summary_challenging.md`
- `outputs/figures/multiseed_hotspots_challenging.png`
- `outputs/figures/multiseed_peak_temp_challenging.png`
- `outputs/figures/multiseed_coverage_challenging.png`

## Local Dashboard

The lightweight Streamlit dashboard reads existing `outputs/` artifacts and can
also run a local simulator replay for demo inspection. Start it from the project
root:

```bash
python run_dashboard.py
```

Equivalent direct command:

```bash
streamlit run thermalguard_cal/dashboard.py
```

Dashboard views:

- Simulation Replay: 4x4 chip heatmap over time, sensor locations, selected
  core, scheduler decision, point/quantile/conformal risk heatmaps, and peak
  temperature trace.
- Results Explorer: scheduler metrics, model metrics, coverage metrics, and
  executive figure.
- Calibration View: conformal correction, calibration sample count, and
  coverage before/after conformal calibration.
- Scheduler Comparison: direct comparison of point prediction, uncalibrated
  quantile, and conformal upper-bound schedulers.
- Heatmap Comparison: existing ID/OOD heatmap figures and saved raw heatmap
  snapshots.
- Stress and Multiseed: stress sweep tables, multiseed aggregate metrics, and
  multiseed figures.

Simulation Replay can optionally write flat CSV replay logs under
`outputs/replays/` from the dashboard.

## Visual Results

`run_all.py` and `run_evaluate_schedulers.py` already write all figures and a
"Visual Results" section in `outputs/reports/final_report.md`. To rebuild only
the figures and that report section from the existing metric CSVs, without
re-running the simulator or models:

```bash
python run_make_plots.py
```

This is a matplotlib-only, saved-files visual layer (no seaborn). It reads
`outputs/reports/metrics_id.csv`, `metrics_ood.csv`,
`coverage_metrics.csv`, `policy_drift_metrics.csv`, and `heatmap_snapshots.npz`,
and writes the stable figure filenames below under `outputs/figures/`:

| File | What it shows |
|------|---------------|
| `executive_summary.png` | One portfolio/README overview: OOD peak temperature, OOD hotspots, coverage collapse, and drift |
| `peak_temperature_by_scheduler_id.png` / `_ood.png` | Peak temperature per scheduler vs the 85 C limit, ID and OOD |
| `hotspot_violations_id.png` / `_ood.png` | Hotspot timestep counts per scheduler, ID and OOD |
| `coverage_id_vs_ood.png` | Nominal vs marginal vs selected-core coverage, ID vs OOD |
| `selected_core_coverage_gap.png` | Selected-core coverage minus nominal, ID vs OOD |
| `policy_drift_id_vs_ood.png` | Policy-induced feature drift per scheduler, ID vs OOD |
| `safety_vs_throughput_id.png` / `_ood.png` | Peak temperature vs completed tasks, one point per scheduler |
| `heatmap_conformal_id.png` / `_ood.png` | Representative final 4x4 chip frame for the conformal scheduler |
| `heatmap_comparison_ood.png` | Conformal vs a bad observed-sensor baseline under OOD |
| `max_temperature_by_scheduler.png` | ID max-temperature traces over time |

## Full Mode

Full mode uses 200 train, 50 calibration, 50 ID test, and 50 OOD episodes with
500 timesteps each. It is slower.

```bash
python run_all.py --full
```

## Outputs

- `outputs/data/`: generated feature arrays, labels, metadata, feature names
- `outputs/models/`: trained sklearn models and conformal calibrator
- `outputs/figures/`: scheduler, coverage, drift, and heatmap plots
- `outputs/reports/`: metrics CSVs, final report, and run manifest
- `outputs/reports/research_review_bundle.zip`: source, tests, scripts,
  reports, figures, config, manifests, summaries, and metrics for review

## Interpreting Results

Marginal coverage is measured across all candidate predictions on visited
states. Selected-core coverage is measured after the conformal scheduler selects
one placement. The gap between those values reflects scheduler selection
effects. Policy-induced distribution drift is reported separately by comparing
visited rollout features against the calibration feature distribution, even on
in-distribution workloads.

Conformal diagnostics are written to `outputs/reports/conformal_diagnostics.csv`.
They explicitly report target coverage, quantile alpha, conformal correction,
calibration sample count, and empirical coverage before and after conformal
calibration for calibration, ID, and OOD splits. If the correction is zero, the
quantile model was already conservative on the calibration split; if the
correction is positive, conformal widened the upper bound.

Use scheduler metrics conservatively. A conformal scheduler can improve
coverage/calibration trust without winning peak temperature or hotspot counts.
The final report includes a direct comparison of `point_prediction_rf`,
`uncalibrated_quantile`, and `conformal_upper_bound` so the claim is not
overstated.

This is a simulation MVP, not real silicon validation. It does not include
HotSpot, OpenROAD, chiplets, DVFS, GNNs, active sensing, or FPGA logic.
