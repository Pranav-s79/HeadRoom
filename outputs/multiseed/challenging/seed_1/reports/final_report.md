# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:01:21+00:00

## Student Summary: What This Run Means

ThermalGuard-Cal builds a 4x4 chip thermal simulator, generates task-placement
datasets, trains point and upper-bound models, calibrates an upper bound with
one-sided conformal prediction, and compares schedulers. The model predicts the
future peak temperature of the whole chip after assigning a task to a candidate
core.

Scheduler categories: simple baselines use randomness or round-robin placement,
sparse-sensor heuristics use only noisy observed temperatures, the oracle uses
privileged true temperatures for reference, and model-based schedulers use
learned future-temperature predictions. ID means the test workload matches the
training/calibration distribution. OOD means the workload is shifted to hotter,
burstier behavior with more sensor dropout. Coverage means the true future peak
temperature is below the predicted upper bound. Selected-core coverage measures
that coverage only after the scheduler has selected one core.

Schedulers with at least one hotspot violation in this run: coolest_core_observed, trend_aware_observed.
Schedulers with zero hotspot violations in this run: random, round_robin, coolest_core_oracle_true_temp, trend_aware_observed, point_prediction_rf, uncalibrated_quantile, conformal_upper_bound.
Conformal status: Coverage improved or was verified on calibration-like data. What is not proven yet: this is not real
silicon validation, and OOD coverage is not guaranteed. The next experiment is
the challenging preset plus multiseed validation, which checks whether scheduler
choice matters in a harder but non-saturating regime.

| Result verdict | Status |
|---|---|
| Pipeline status | working |
| ID calibration | good |
| OOD calibration | bad |
| Sparse-sensor baseline failure | yes |
| Model-based scheduler advantage | promising |
| Conformal advantage over model baselines | mixed |
| Next needed work | challenging preset + multiseed validation |


## Project Summary

ThermalGuard-Cal is a simulation-based Python MVP for conformal upper-bound
thermal scheduling on a 4x4 many-core chip. It implements a stochastic thermal
simulator, stable in-distribution and separate OOD workload generators, sparse
noisy sensors, action-conditioned datasets, point/quantile/conformal models,
scheduler baselines, fair ID/OOD evaluation, plots, and reports.

Current run preset: **challenging**.

## Prediction Target

The model predicts **future peak chip temperature over the horizon** as a
global whole-chip quantity. It does **not** predict the candidate core's own
future temperature in isolation. Therefore selected-core coverage means
coverage of the global outcome that results from a placement choice.

## Simulator And Workloads

The simulator uses heat gain from active task power, ambient cooling,
4-neighbor diffusion, mild thermal inertia, and small stochastic noise. ID
train/calibration/test episodes share the same stable workload distribution.
OOD episodes use a separate higher-power, burstier mix and higher sensor
dropout. Model features only use sparse/noisy sensor observations and workload
metadata; true temperatures are reserved for simulator physics and labels.

The `coolest_core_oracle_true_temp` row is intentionally labeled as a privileged
oracle baseline because it reads true current temperatures. It is useful as a
simulation reference, not as a deployable sparse-sensor scheduler.

## Model Metrics

| split       | model           | metric               |     value |
|:------------|:----------------|:---------------------|----------:|
| calibration | linear_point    | mae                  |  1.7555   |
| calibration | linear_point    | rmse                 |  2.49492  |
| calibration | linear_point    | max_abs_error        |  7.39838  |
| calibration | forest_point    | mae                  |  1.62362  |
| calibration | forest_point    | rmse                 |  2.05521  |
| calibration | forest_point    | max_abs_error        |  5.0806   |
| calibration | quantile_upper  | empirical_coverage   |  1        |
| calibration | quantile_upper  | average_bound        | 52.9002   |
| calibration | quantile_upper  | average_conservatism |  8.22992  |
| calibration | conformal_upper | empirical_coverage   |  1        |
| calibration | conformal_upper | average_bound        | 52.9002   |
| calibration | conformal_upper | average_conservatism |  8.22992  |
| test_id     | linear_point    | mae                  |  2.6678   |
| test_id     | linear_point    | rmse                 |  3.74092  |
| test_id     | linear_point    | max_abs_error        | 11.8655   |
| test_id     | forest_point    | mae                  |  2.616    |
| test_id     | forest_point    | rmse                 |  4.24133  |
| test_id     | forest_point    | max_abs_error        | 16.8804   |
| test_id     | quantile_upper  | empirical_coverage   |  0.779661 |
| test_id     | quantile_upper  | average_bound        | 52.5898   |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |       value |
|:------------------------------------------------|:------------|------------:|
| target_coverage                                 | all         |    0.9      |
| quantile_model_alpha                            | all         |    0.9      |
| conformal_correction                            | calibration |    0        |
| conformal_quantile_level                        | calibration |    0.900678 |
| calibration_samples                             | calibration | 2064        |
| calibration_empirical_coverage_before_conformal | calibration |    1        |
| calibration_empirical_coverage_after_conformal  | calibration |    1        |
| id_empirical_coverage_before_conformal          | test_id     |    0.779661 |
| id_empirical_coverage_after_conformal           | test_id     |    0.779661 |
| ood_empirical_coverage_before_conformal         | test_ood    |    0.719618 |
| ood_empirical_coverage_after_conformal          | test_ood    |    0.719618 |

In this run, the quantile model was already conservative on the calibration split, so conformal calibration verified the bound but did not widen it.


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            63.5486 |                   46.9406 |                    0 |                65 |              118 |                 nan |                      nan |           0.342718 |       0.899225 |
| round_robin                   | deployable_baseline_sensor_observed |            68.3625 |                   45.9663 |                    0 |                65 |              118 |                 nan |                      nan |           0.397809 |       0.762186 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            85.8936 |                   55.4642 |                    2 |                65 |              118 |                 nan |                      nan |           1.97243  |       0.949153 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            55.5788 |                   43.948  |                    0 |                65 |              118 |                 nan |                      nan |           0.397959 |       0.685915 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            83.6669 |                   54.1203 |                    0 |                65 |              118 |                 nan |                      nan |           1.79079  |       0.949153 |
| point_prediction_rf           | model_based_sensor_observed         |            55.134  |                   43.2161 |                    0 |                65 |              118 |                 nan |                      nan |           0.432427 |       0.813559 |
| uncalibrated_quantile         | model_based_sensor_observed         |            65.08   |                   50.1298 |                    0 |                65 |              118 |                 nan |                      nan |           0.836238 |       0.957627 |
| conformal_upper_bound         | model_based_sensor_observed         |            51.7905 |                   42.6474 |                    0 |                65 |              118 |                   1 |                        1 |           0.405398 |       0.872356 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            74.4243 |                   52.5759 |                    0 |                63 |              144 |          nan        |               nan        |           0.890219 |       0.922804 |
| round_robin                   | deployable_baseline_sensor_observed |            67.3323 |                   47.9277 |                    0 |                63 |              144 |          nan        |               nan        |           0.785899 |       0.9375   |
| coolest_core_observed         | deployable_baseline_sensor_observed |           114.241  |                   66.453  |                   50 |                63 |              144 |          nan        |               nan        |           3.14799  |       0.958333 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            72.1069 |                   49.8309 |                    0 |                63 |              144 |          nan        |               nan        |           0.944726 |       0.851421 |
| trend_aware_observed          | deployable_baseline_sensor_observed |           105.559  |                   63.3831 |                   37 |                63 |              144 |          nan        |               nan        |           2.82194  |       0.930233 |
| point_prediction_rf           | model_based_sensor_observed         |            60.8016 |                   46.209  |                    0 |                63 |              144 |          nan        |               nan        |           0.895176 |       0.847222 |
| uncalibrated_quantile         | model_based_sensor_observed         |            69.7701 |                   52.9732 |                    0 |                63 |              144 |          nan        |               nan        |           0.981616 |       0.965278 |
| conformal_upper_bound         | model_based_sensor_observed         |            68.4206 |                   47.036  |                    0 |                63 |              144 |            0.685764 |                 0.680556 |           0.935091 |       0.909722 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             1        | 1888 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             1        |  118 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.685764 | 2304 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.680556 |  144 |

Marginal candidate coverage and selected-core coverage are reported separately.
The difference captures the selection step where the scheduler chooses one
candidate out of 16. Distribution drift is reported separately below and should
not be conflated with the selection-bias coverage gap.

## Does Conformal Add Scheduling Value?

This section compares only `point_prediction_rf`, `uncalibrated_quantile`, and
`conformal_upper_bound`. Conformal should not be called best unless both the
scheduling metrics and coverage metrics support that claim.

| split   | lowest_peak_temperature   | fewest_hotspot_violations   | best_measured_coverage   | conformal_interpretation               |
|:--------|:--------------------------|:----------------------------|:-------------------------|:---------------------------------------|
| id      | conformal_upper_bound     | point_prediction_rf         | conformal_upper_bound    | safer in this split                    |
| ood     | point_prediction_rf       | point_prediction_rf         | conformal_upper_bound    | coverage value, similar safety outcome |

- For ID, conformal_upper_bound has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.
- For OOD, point_prediction_rf has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.

Calibration can improve statistical trust even when scheduler outcomes are
similar, because it turns an uncalibrated quantile model into an auditable
coverage claim on calibration-like data. It does not create an OOD guarantee.


## Preset Results

This separates saved preset snapshots from the current canonical CSVs, so the
original normal/easy quick result can be compared against a challenging run when
both have been executed.

| result_group   | scheduler                     |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   selected_core_coverage |
|:---------------|:------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-------------------------:|
| challenging_id | random                        |            63.5486 |                   46.9406 |                    0 |                65 |                      nan |
| challenging_id | round_robin                   |            68.3625 |                   45.9663 |                    0 |                65 |                      nan |
| challenging_id | coolest_core_observed         |            85.8936 |                   55.4642 |                    2 |                65 |                      nan |
| challenging_id | coolest_core_oracle_true_temp |            55.5788 |                   43.948  |                    0 |                65 |                      nan |
| challenging_id | trend_aware_observed          |            83.6669 |                   54.1203 |                    0 |                65 |                      nan |
| challenging_id | point_prediction_rf           |            55.134  |                   43.2161 |                    0 |                65 |                      nan |
| challenging_id | uncalibrated_quantile         |            65.08   |                   50.1298 |                    0 |                65 |                      nan |
| challenging_id | conformal_upper_bound         |            51.7905 |                   42.6474 |                    0 |                65 |                        1 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | coolest_core_observed         |           1.97243  |       0.949153 |
| id      | trend_aware_observed          |           1.79079  |       0.949153 |
| id      | uncalibrated_quantile         |           0.836238 |       0.957627 |
| id      | point_prediction_rf           |           0.432427 |       0.813559 |
| id      | conformal_upper_bound         |           0.405398 |       0.872356 |
| id      | coolest_core_oracle_true_temp |           0.397959 |       0.685915 |
| id      | round_robin                   |           0.397809 |       0.762186 |
| id      | random                        |           0.342718 |       0.899225 |
| ood     | coolest_core_observed         |           3.14799  |       0.958333 |
| ood     | trend_aware_observed          |           2.82194  |       0.930233 |
| ood     | uncalibrated_quantile         |           0.981616 |       0.965278 |
| ood     | coolest_core_oracle_true_temp |           0.944726 |       0.851421 |
| ood     | conformal_upper_bound         |           0.935091 |       0.909722 |
| ood     | point_prediction_rf           |           0.895176 |       0.847222 |
| ood     | random                        |           0.890219 |       0.922804 |
| ood     | round_robin                   |           0.785899 |       0.9375   |

Largest feature-level shifts:

| split   | scheduler             | feature                |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:----------------------|:-----------------------|-------------------:|---------------:|-------------------:|----------:|
| id      | coolest_core_observed | power_12               |           0.898279 |        9.1619  |           12.0761  |  0.949153 |
| id      | trend_aware_observed  | power_12               |           0.898279 |        8.64761 |           11.3246  |  0.949153 |
| id      | coolest_core_observed | load_12                |          15.5349   |      149.78    |            9.85273 |  0.855932 |
| id      | trend_aware_observed  | load_12                |          15.5349   |      130.102   |            8.40849 |  0.855932 |
| id      | coolest_core_observed | power_0                |           1.21489  |        8.98466 |            8.22055 |  0.830508 |
| id      | trend_aware_observed  | power_0                |           1.21489  |        8.41516 |            7.61801 |  0.830508 |
| id      | coolest_core_observed | sensor_temp_imputed_0  |          37.1477   |       50.7188  |            6.45274 |  0.621732 |
| id      | coolest_core_observed | sensor_temp_imputed_12 |          37.3128   |       50.1467  |            6.45113 |  0.673105 |
| id      | trend_aware_observed  | sensor_temp_imputed_0  |          37.1477   |       49.9138  |            6.06995 |  0.572132 |
| id      | uncalibrated_quantile | power_0                |           1.21489  |        6.85768 |            5.97016 |  0.957627 |
| id      | coolest_core_observed | power_15               |           1.64465  |       10.1453  |            5.8564  |  0.847458 |
| id      | trend_aware_observed  | sensor_temp_imputed_12 |          37.3128   |       48.7434  |            5.74575 |  0.673105 |
| id      | trend_aware_observed  | load_0                 |          21.7132   |      146.144   |            5.71108 |  0.84818  |
| id      | coolest_core_observed | load_0                 |          21.7132   |      142.347   |            5.53683 |  0.783997 |
| id      | coolest_core_observed | load_3                 |          40.0155   |      202.441   |            5.17218 |  0.752989 |
| id      | coolest_core_observed | power_3                |           2.04298  |       10.3881  |            5.06251 |  0.70339  |
| id      | coolest_core_observed | load_15                |          29.2326   |      165.949   |            5.03311 |  0.70339  |
| id      | coolest_core_observed | sensor_temp_imputed_1  |          37.9081   |       50.4527  |            4.91231 |  0.542373 |
| id      | coolest_core_observed | sensor_temp_imputed_2  |          37.9081   |       50.4527  |            4.91231 |  0.542373 |
| id      | coolest_core_observed | sensor_temp_imputed_4  |          37.9081   |       50.4527  |            4.91231 |  0.542373 |

The drift table compares features visited by each scheduler's rollout against
the calibration feature distribution. This measures policy-induced state
distribution shift even when the workload generator remains in-distribution.

## Figures

Stable figure filenames under `outputs/figures/` (see the Visual Results
section below for plain-English explanations). Regenerate any time with
`python run_make_plots.py`.

- `outputs/figures/executive_summary.png` (portfolio/README overview)
- `outputs/figures/peak_temperature_by_scheduler_id.png`
- `outputs/figures/peak_temperature_by_scheduler_ood.png`
- `outputs/figures/hotspot_violations_id.png`
- `outputs/figures/hotspot_violations_ood.png`
- `outputs/figures/coverage_id_vs_ood.png`
- `outputs/figures/selected_core_coverage_gap.png`
- `outputs/figures/policy_drift_id_vs_ood.png`
- `outputs/figures/safety_vs_throughput_id.png`
- `outputs/figures/safety_vs_throughput_ood.png`
- `outputs/figures/heatmap_conformal_id.png`
- `outputs/figures/heatmap_conformal_ood.png`
- `outputs/figures/heatmap_comparison_ood.png`
- `outputs/figures/max_temperature_by_scheduler.png`

## Visual Results

The figures below turn the raw CSV metrics into the main result story for the
current quick run. All figures live under `outputs/figures/` with stable
filenames and can be regenerated independently with `python run_make_plots.py`.

![Executive summary](../figures/executive_summary.png)

The executive-summary figure is a single portfolio/README overview: OOD peak
temperature by scheduler, OOD hotspot violations, the conformal coverage
collapse from ID to OOD, and conformal policy drift. The detailed per-figure
breakdown follows.

![ID peak temperature by scheduler](../figures/peak_temperature_by_scheduler_id.png)

![OOD peak temperature by scheduler](../figures/peak_temperature_by_scheduler_ood.png)

Peak-temperature comparisons show the thermal safety margin against the 85 C
limit. In this quick run, several model-based schedulers avoid hotspots while
some sparse-observed baselines overheat badly under OOD conditions.

![ID hotspot violations](../figures/hotspot_violations_id.png)

![OOD hotspot violations](../figures/hotspot_violations_ood.png)

Hotspot bars make the safety failures more direct: ID is thermally easy here,
but OOD separates robust policies from brittle observed-sensor heuristics.

![Coverage ID vs OOD](../figures/coverage_id_vs_ood.png)

Conformal coverage is reported three ways: nominal target, marginal candidate
coverage, and selected-core coverage. ID selected coverage is near nominal; OOD
coverage collapses under OOD shift. This is why OOD calibration performance should not be
claimed as a formal safety guarantee.

![Selected-core coverage gap](../figures/selected_core_coverage_gap.png)

Selected-core coverage is measured separately because the scheduler chooses one
candidate out of 16. That selection step can create a gap from nominal coverage
even before considering policy-induced state drift.

![Policy drift ID vs OOD](../figures/policy_drift_id_vs_ood.png)

Policy drift compares rollout feature distributions against the calibration
feature distribution. It is a distinct effect from candidate selection bias and
from the deliberate OOD workload shift.

![ID safety vs throughput](../figures/safety_vs_throughput_id.png)

![OOD safety vs throughput](../figures/safety_vs_throughput_ood.png)

The safety/throughput scatter shows whether lower temperature is being bought
with lost completed work. In the current quick run, model-based schedulers avoid
hotspots without losing completed tasks, but harder presets are still needed if
ID remains too thermally easy.

![Conformal ID heatmap](../figures/heatmap_conformal_id.png)

![Conformal OOD heatmap](../figures/heatmap_conformal_ood.png)

![Conformal vs observed-baseline OOD heatmaps](../figures/heatmap_comparison_ood.png)

Representative 4x4 heatmaps show the final recorded chip-temperature frame for
the conformal scheduler. The OOD comparison heatmap puts the conformal scheduler
next to a bad observed-sensor baseline (`coolest_core_observed`), which chases
cool sensor readings into a hotspot under the OOD workload.

### What the figures say in plain English

- **ID calibration holds.** On in-distribution workloads, conformal
  selected-core coverage sits near the 0.9 nominal
  target.
- **OOD calibration breaks.** Under the shifted OOD workload, both marginal and
  selected coverage drop well below nominal, so the conformal guarantee should
  not be claimed as a formal OOD safety guarantee.
- **Selected-core coverage is measured separately** from marginal candidate
  coverage, because the scheduler picks one core out of 16 and that selection
  step can move coverage on its own.
- **Some observed-sensor baselines overheat badly in OOD.** Sparse-sensor
  heuristics such as coolest-core-observed and trend-aware-observed run far past
  the 85 C limit on OOD workloads.
- **Model-based schedulers avoid hotspots in this quick run**, staying under the
  thermal limit without losing completed tasks.
- **This quick run still needs harder presets.** ID is thermally easy here, so a
  more challenging preset is needed to validate that the model-based advantage is
  not just an artifact of an easy in-distribution regime.

## Limitations

This is a research MVP, not a validated chip thermal model. It does not include
HotSpot, OpenROAD, chiplets, DVFS, GNNs, active sensing, FPGA logic, or a web
dashboard. The conformal guarantee is marginal on calibration-like data and is
not a formal OOD safety guarantee.
