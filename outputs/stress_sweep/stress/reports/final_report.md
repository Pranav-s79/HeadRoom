# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:18:47+00:00

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

Schedulers with at least one hotspot violation in any evaluated split: coolest_core_observed, random, trend_aware_observed, uncalibrated_quantile.
Schedulers with zero hotspot violations across all evaluated splits: conformal_upper_bound, coolest_core_oracle_true_temp, point_prediction_rf, round_robin.
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

Current run preset: **stress**.

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
| calibration | linear_point    | mae                  |  2.45167  |
| calibration | linear_point    | rmse                 |  3.03229  |
| calibration | linear_point    | max_abs_error        |  8.11     |
| calibration | forest_point    | mae                  |  1.64743  |
| calibration | forest_point    | rmse                 |  2.39692  |
| calibration | forest_point    | max_abs_error        |  8.04452  |
| calibration | quantile_upper  | empirical_coverage   |  0.879545 |
| calibration | quantile_upper  | average_bound        | 58.4175   |
| calibration | quantile_upper  | average_conservatism |  9.97493  |
| calibration | conformal_upper | empirical_coverage   |  0.900568 |
| calibration | conformal_upper | average_bound        | 58.84     |
| calibration | conformal_upper | average_conservatism | 10.3974   |
| test_id     | linear_point    | mae                  |  2.27491  |
| test_id     | linear_point    | rmse                 |  2.70684  |
| test_id     | linear_point    | max_abs_error        |  6.01698  |
| test_id     | forest_point    | mae                  |  2.75178  |
| test_id     | forest_point    | rmse                 |  3.44833  |
| test_id     | forest_point    | max_abs_error        |  9.10309  |
| test_id     | quantile_upper  | empirical_coverage   |  0.832921 |
| test_id     | quantile_upper  | average_bound        | 58.3005   |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |       value |
|:------------------------------------------------|:------------|------------:|
| target_coverage                                 | all         |    0.9      |
| quantile_model_alpha                            | all         |    0.9      |
| conformal_correction                            | calibration |    0.422431 |
| conformal_quantile_level                        | calibration |    0.900568 |
| calibration_samples                             | calibration | 1760        |
| calibration_empirical_coverage_before_conformal | calibration |    0.879545 |
| calibration_empirical_coverage_after_conformal  | calibration |    0.900568 |
| id_empirical_coverage_before_conformal          | test_id     |    0.832921 |
| id_empirical_coverage_after_conformal           | test_id     |    0.844059 |
| ood_empirical_coverage_before_conformal         | test_ood    |    0.58479  |
| ood_empirical_coverage_after_conformal          | test_ood    |    0.612325 |


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            78.5609 |                   47.3906 |                    0 |                27 |              101 |                 nan |                      nan |           0.398532 |       0.909091 |
| round_robin                   | deployable_baseline_sensor_observed |            60.2497 |                   42.9316 |                    0 |                27 |              101 |                 nan |                      nan |           0.264393 |       0.770477 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            88.8759 |                   54.2587 |                    4 |                27 |              101 |                 nan |                      nan |           1.43173  |       0.981818 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            59.0142 |                   43.143  |                    0 |                27 |              101 |                 nan |                      nan |           0.219436 |       0.672727 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            84.7109 |                   52.4036 |                    0 |                27 |              101 |                 nan |                      nan |           1.27486  |       0.981818 |
| point_prediction_rf           | model_based_sensor_observed         |            55.2675 |                   42.9702 |                    0 |                27 |              101 |                 nan |                      nan |           0.235577 |       0.753735 |
| uncalibrated_quantile         | model_based_sensor_observed         |            70.4268 |                   52.4639 |                    0 |                27 |              101 |                 nan |                      nan |           0.735426 |       0.945455 |
| conformal_upper_bound         | model_based_sensor_observed         |            55.6361 |                   42.2978 |                    0 |                27 |              101 |                   1 |                        1 |           0.241264 |       0.618182 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |           124.698  |                   65.9341 |                   35 |                25 |              143 |          nan        |               nan        |           0.934951 |       0.902098 |
| round_robin                   | deployable_baseline_sensor_observed |            82.8436 |                   53.4089 |                    0 |                25 |              143 |          nan        |               nan        |           0.834499 |       0.802797 |
| coolest_core_observed         | deployable_baseline_sensor_observed |           130      |                   77.3781 |                   56 |                25 |              143 |          nan        |               nan        |           2.94209  |       0.981818 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            78.1651 |                   56.0858 |                    0 |                25 |              143 |          nan        |               nan        |           0.820519 |       0.762238 |
| trend_aware_observed          | deployable_baseline_sensor_observed |           130      |                   75.6387 |                   51 |                25 |              143 |          nan        |               nan        |           2.58208  |       0.981818 |
| point_prediction_rf           | model_based_sensor_observed         |            70.8114 |                   49.4497 |                    0 |                25 |              143 |          nan        |               nan        |           0.813165 |       0.826573 |
| uncalibrated_quantile         | model_based_sensor_observed         |            97.0275 |                   58.3957 |                    7 |                25 |              143 |          nan        |               nan        |           0.923971 |       0.965035 |
| conformal_upper_bound         | model_based_sensor_observed         |            77.0598 |                   50.1396 |                    0 |                25 |              143 |            0.770542 |                 0.762238 |           0.800602 |       0.823776 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             1        | 1616 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             1        |  101 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.770542 | 2288 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.762238 |  143 |

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
| id      | point_prediction_rf       | point_prediction_rf         | conformal_upper_bound    | coverage value, similar safety outcome |
| ood     | point_prediction_rf       | point_prediction_rf         | conformal_upper_bound    | coverage value, similar safety outcome |

- For ID, point_prediction_rf has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.
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
| stress_id      | random                        |            78.5609 |                   47.3906 |                    0 |                27 |                      nan |
| stress_id      | round_robin                   |            60.2497 |                   42.9316 |                    0 |                27 |                      nan |
| stress_id      | coolest_core_observed         |            88.8759 |                   54.2587 |                    4 |                27 |                      nan |
| stress_id      | coolest_core_oracle_true_temp |            59.0142 |                   43.143  |                    0 |                27 |                      nan |
| stress_id      | trend_aware_observed          |            84.7109 |                   52.4036 |                    0 |                27 |                      nan |
| stress_id      | point_prediction_rf           |            55.2675 |                   42.9702 |                    0 |                27 |                      nan |
| stress_id      | uncalibrated_quantile         |            70.4268 |                   52.4639 |                    0 |                27 |                      nan |
| stress_id      | conformal_upper_bound         |            55.6361 |                   42.2978 |                    0 |                27 |                        1 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | coolest_core_observed         |           1.43173  |       0.981818 |
| id      | trend_aware_observed          |           1.27486  |       0.981818 |
| id      | uncalibrated_quantile         |           0.735426 |       0.945455 |
| id      | random                        |           0.398532 |       0.909091 |
| id      | round_robin                   |           0.264393 |       0.770477 |
| id      | conformal_upper_bound         |           0.241264 |       0.618182 |
| id      | point_prediction_rf           |           0.235577 |       0.753735 |
| id      | coolest_core_oracle_true_temp |           0.219436 |       0.672727 |
| ood     | coolest_core_observed         |           2.94209  |       0.981818 |
| ood     | trend_aware_observed          |           2.58208  |       0.981818 |
| ood     | random                        |           0.934951 |       0.902098 |
| ood     | uncalibrated_quantile         |           0.923971 |       0.965035 |
| ood     | round_robin                   |           0.834499 |       0.802797 |
| ood     | coolest_core_oracle_true_temp |           0.820519 |       0.762238 |
| ood     | point_prediction_rf           |           0.813165 |       0.826573 |
| ood     | conformal_upper_bound         |           0.800602 |       0.823776 |

Largest feature-level shifts:

| split   | scheduler             | feature               |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:----------------------|:----------------------|-------------------:|---------------:|-------------------:|----------:|
| id      | trend_aware_observed  | load_12               |           56.5818  |      202.149   |            4.92508 |  0.724392 |
| id      | coolest_core_observed | power_0               |            2.76935 |       10.5412  |            4.80628 |  0.722772 |
| id      | coolest_core_observed | load_3                |           42.5     |      198.96    |            4.46118 |  0.683978 |
| id      | trend_aware_observed  | load_3                |           42.5     |      195       |            4.34825 |  0.822592 |
| id      | coolest_core_observed | power_3               |            2.45226 |       10.482   |            4.2868  |  0.772277 |
| id      | coolest_core_observed | power_15              |            2.99518 |       10.5982  |            4.26335 |  0.739154 |
| id      | coolest_core_observed | sensor_temp_imputed_3 |           38.685   |       52.966   |            4.177   |  0.612061 |
| id      | trend_aware_observed  | power_3               |            2.45226 |       10.1548  |            4.11213 |  0.831683 |
| id      | coolest_core_observed | load_12               |           56.5818  |      172.733   |            3.92983 |  0.763996 |
| id      | coolest_core_observed | load_0                |           49.6091  |      208.554   |            3.82818 |  0.623762 |
| id      | trend_aware_observed  | power_12              |            3.78818 |       10.4675  |            3.81066 |  0.663366 |
| id      | trend_aware_observed  | power_0               |            2.76935 |        8.74052 |            3.69271 |  0.754095 |
| id      | trend_aware_observed  | sensor_temp_imputed_3 |           38.685   |       50.9174  |            3.57782 |  0.581548 |
| id      | coolest_core_observed | power_12              |            3.78818 |        9.98787 |            3.53701 |  0.742574 |
| id      | trend_aware_observed  | power_15              |            2.99518 |        8.73358 |            3.21779 |  0.659946 |
| id      | coolest_core_observed | load_15               |           58.0364  |      200.307   |            3.10998 |  0.623762 |
| id      | coolest_core_observed | sensor_temp_imputed_1 |           40.3245  |       53.2274  |            3.06441 |  0.49586  |
| id      | coolest_core_observed | sensor_temp_imputed_2 |           40.3245  |       53.2274  |            3.06441 |  0.49586  |
| id      | coolest_core_observed | sensor_temp_imputed_4 |           40.3245  |       53.2274  |            3.06441 |  0.49586  |
| id      | coolest_core_observed | sensor_temp_imputed_5 |           40.3245  |       53.2274  |            3.06441 |  0.49586  |

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

## Limitations

This is a research MVP, not a validated chip thermal model. It does not include
HotSpot, OpenROAD, chiplets, DVFS, GNNs, active sensing, FPGA logic, or a web
dashboard. The conformal guarantee is marginal on calibration-like data and is
not a formal OOD safety guarantee.
