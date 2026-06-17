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

## Interpreting Results

Marginal coverage is measured across all candidate predictions on visited
states. Selected-core coverage is measured after the conformal scheduler selects
one placement. The gap between those values reflects scheduler selection
effects. Policy-induced distribution drift is reported separately by comparing
visited rollout features against the calibration feature distribution, even on
in-distribution workloads.

This is a simulation MVP, not real silicon validation. It does not include
HotSpot, OpenROAD, chiplets, DVFS, GNNs, active sensing, FPGA logic, or a web
dashboard.
