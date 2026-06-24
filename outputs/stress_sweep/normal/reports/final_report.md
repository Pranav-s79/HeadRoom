# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:15:33+00:00

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

Schedulers with at least one hotspot violation in any evaluated split: none.
Schedulers with zero hotspot violations across all evaluated splits: conformal_upper_bound, coolest_core_observed, coolest_core_oracle_true_temp, point_prediction_rf, random, round_robin, trend_aware_observed, uncalibrated_quantile.
Conformal status: Coverage improved or was verified on calibration-like data. What is not proven yet: this is not real
silicon validation, and OOD coverage is not guaranteed. The next experiment is
the challenging preset plus multiseed validation, which checks whether scheduler
choice matters in a harder but non-saturating regime.

| Result verdict | Status |
|---|---|
| Pipeline status | working |
| ID calibration | good |
| OOD calibration | bad |
| Sparse-sensor baseline failure | no |
| Model-based scheduler advantage | not proven |
| Conformal advantage over model baselines | mixed |
| Next needed work | challenging preset + multiseed validation |


## Project Summary

ThermalGuard-Cal is a simulation-based Python MVP for conformal upper-bound
thermal scheduling on a 4x4 many-core chip. It implements a stochastic thermal
simulator, stable in-distribution and separate OOD workload generators, sparse
noisy sensors, action-conditioned datasets, point/quantile/conformal models,
scheduler baselines, fair ID/OOD evaluation, plots, and reports.

Current run preset: **normal**.

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
| calibration | linear_point    | mae                  |  1.10899  |
| calibration | linear_point    | rmse                 |  1.43848  |
| calibration | linear_point    | max_abs_error        |  3.58038  |
| calibration | forest_point    | mae                  |  1.76702  |
| calibration | forest_point    | rmse                 |  1.97818  |
| calibration | forest_point    | max_abs_error        |  3.62195  |
| calibration | quantile_upper  | empirical_coverage   |  0.998512 |
| calibration | quantile_upper  | average_bound        | 42.3778   |
| calibration | quantile_upper  | average_conservatism |  3.76417  |
| calibration | conformal_upper | empirical_coverage   |  0.998512 |
| calibration | conformal_upper | average_bound        | 42.3778   |
| calibration | conformal_upper | average_conservatism |  3.76417  |
| test_id     | linear_point    | mae                  |  1.40929  |
| test_id     | linear_point    | rmse                 |  2.21924  |
| test_id     | linear_point    | max_abs_error        |  5.84477  |
| test_id     | forest_point    | mae                  |  2.02069  |
| test_id     | forest_point    | rmse                 |  2.51104  |
| test_id     | forest_point    | max_abs_error        |  5.2472   |
| test_id     | quantile_upper  | empirical_coverage   |  0.888587 |
| test_id     | quantile_upper  | average_bound        | 42.3988   |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |      value |
|:------------------------------------------------|:------------|-----------:|
| target_coverage                                 | all         |   0.9      |
| quantile_model_alpha                            | all         |   0.9      |
| conformal_correction                            | calibration |   0        |
| conformal_quantile_level                        | calibration |   0.901786 |
| calibration_samples                             | calibration | 672        |
| calibration_empirical_coverage_before_conformal | calibration |   0.998512 |
| calibration_empirical_coverage_after_conformal  | calibration |   0.998512 |
| id_empirical_coverage_before_conformal          | test_id     |   0.888587 |
| id_empirical_coverage_after_conformal           | test_id     |   0.888587 |
| ood_empirical_coverage_before_conformal         | test_ood    |   0.505435 |
| ood_empirical_coverage_after_conformal          | test_ood    |   0.505435 |

In this run, the quantile model was already conservative on the calibration split, so conformal calibration verified the bound but did not widen it.


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            47.7986 |                   38.0087 |                    0 |                22 |               46 |          nan        |               nan        |             509511 |       0.652174 |
| round_robin                   | deployable_baseline_sensor_observed |            44.7799 |                   37.5295 |                    0 |                22 |               46 |          nan        |               nan        |             509511 |       0.582816 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            48.4145 |                   38.7598 |                    0 |                22 |               46 |          nan        |               nan        |             509512 |       0.782609 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            43.168  |                   37.5062 |                    0 |                22 |               46 |          nan        |               nan        |             509511 |       0.650104 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            48.0361 |                   38.6314 |                    0 |                22 |               46 |          nan        |               nan        |             509512 |       0.73913  |
| point_prediction_rf           | model_based_sensor_observed         |            42.9593 |                   37.424  |                    0 |                22 |               46 |          nan        |               nan        |             509511 |       0.76087  |
| uncalibrated_quantile         | model_based_sensor_observed         |            46.4057 |                   39.2242 |                    0 |                22 |               46 |          nan        |               nan        |             509512 |       0.782609 |
| conformal_upper_bound         | model_based_sensor_observed         |            43.2676 |                   37.5468 |                    0 |                22 |               46 |            0.976902 |                 0.978261 |             509511 |       0.717391 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            67.1716 |                   47.2636 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.956522 |
| round_robin                   | deployable_baseline_sensor_observed |            55.3096 |                   43.285  |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.978261 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            77.5665 |                   53.9677 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.956522 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            58.9717 |                   46.1894 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.978261 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            72.9046 |                   52.2623 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.956522 |
| point_prediction_rf           | model_based_sensor_observed         |            49.9454 |                   41.7327 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.913043 |
| uncalibrated_quantile         | model_based_sensor_observed         |            49.5424 |                   42.0351 |                    0 |                36 |               92 |          nan        |               nan        |        2.37772e+06 |       0.978261 |
| conformal_upper_bound         | model_based_sensor_observed         |            50.6771 |                   42.0679 |                    0 |                36 |               92 |            0.482337 |                 0.478261 |        2.37772e+06 |       0.956522 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.976902 |  736 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.978261 |   46 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.482337 | 1472 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.478261 |   92 |

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
| ood     | uncalibrated_quantile     | point_prediction_rf         | conformal_upper_bound    | coverage value, similar safety outcome |

- For ID, point_prediction_rf has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.
- For OOD, uncalibrated_quantile has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.

Calibration can improve statistical trust even when scheduler outcomes are
similar, because it turns an uncalibrated quantile model into an auditable
coverage claim on calibration-like data. It does not create an OOD guarantee.


## Preset Results

This separates saved preset snapshots from the current canonical CSVs, so the
original normal/easy quick result can be compared against a challenging run when
both have been executed.

| result_group   | scheduler                     |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   selected_core_coverage |
|:---------------|:------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-------------------------:|
| normal_id      | random                        |            47.7986 |                   38.0087 |                    0 |                22 |               nan        |
| normal_id      | round_robin                   |            44.7799 |                   37.5295 |                    0 |                22 |               nan        |
| normal_id      | coolest_core_observed         |            48.4145 |                   38.7598 |                    0 |                22 |               nan        |
| normal_id      | coolest_core_oracle_true_temp |            43.168  |                   37.5062 |                    0 |                22 |               nan        |
| normal_id      | trend_aware_observed          |            48.0361 |                   38.6314 |                    0 |                22 |               nan        |
| normal_id      | point_prediction_rf           |            42.9593 |                   37.424  |                    0 |                22 |               nan        |
| normal_id      | uncalibrated_quantile         |            46.4057 |                   39.2242 |                    0 |                22 |               nan        |
| normal_id      | conformal_upper_bound         |            43.2676 |                   37.5468 |                    0 |                22 |                 0.978261 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | coolest_core_observed         |   509512           |       0.782609 |
| id      | trend_aware_observed          |   509512           |       0.73913  |
| id      | uncalibrated_quantile         |   509512           |       0.782609 |
| id      | conformal_upper_bound         |   509511           |       0.717391 |
| id      | random                        |   509511           |       0.652174 |
| id      | point_prediction_rf           |   509511           |       0.76087  |
| id      | round_robin                   |   509511           |       0.582816 |
| id      | coolest_core_oracle_true_temp |   509511           |       0.650104 |
| ood     | coolest_core_observed         |        2.37772e+06 |       0.956522 |
| ood     | trend_aware_observed          |        2.37772e+06 |       0.956522 |
| ood     | round_robin                   |        2.37772e+06 |       0.978261 |
| ood     | coolest_core_oracle_true_temp |        2.37772e+06 |       0.978261 |
| ood     | conformal_upper_bound         |        2.37772e+06 |       0.956522 |
| ood     | point_prediction_rf           |        2.37772e+06 |       0.913043 |
| ood     | uncalibrated_quantile         |        2.37772e+06 |       0.978261 |
| ood     | random                        |        2.37772e+06 |       0.956522 |

Largest feature-level shifts:

| split   | scheduler                     | feature   |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:------------------------------|:----------|-------------------:|---------------:|-------------------:|----------:|
| id      | trend_aware_observed          | load_3    |           2.64286  |       51.5652  |            9.47529 |  0.608696 |
| id      | point_prediction_rf           | load_5    |           2.57143  |       38.913   |            7.9653  |  0.73913  |
| id      | coolest_core_observed         | load_3    |           2.64286  |       41       |            7.42901 |  0.695652 |
| id      | uncalibrated_quantile         | load_0    |           3.66667  |       50.6739  |            6.92845 |  0.673913 |
| id      | coolest_core_observed         | load_12   |           3.40476  |       57.413   |            6.92843 |  0.669772 |
| id      | round_robin                   | load_5    |           2.57143  |       34.087   |            6.90753 |  0.565217 |
| id      | coolest_core_observed         | load_0    |           3.66667  |       48.1087  |            6.55036 |  0.695652 |
| id      | trend_aware_observed          | power_3   |           0.253489 |        2.85036 |            6.47031 |  0.717391 |
| id      | random                        | load_5    |           2.57143  |       31.5217  |            6.34529 |  0.521739 |
| id      | conformal_upper_bound         | load_3    |           2.64286  |       34.7174  |            6.2122  |  0.671843 |
| id      | coolest_core_observed         | power_0   |           0.328763 |        3.26093 |            5.68037 |  0.687371 |
| id      | coolest_core_observed         | power_12  |           0.268391 |        2.78697 |            5.44621 |  0.650104 |
| id      | trend_aware_observed          | load_0    |           3.66667  |       39.8261  |            5.32958 |  0.73913  |
| id      | coolest_core_observed         | power_3   |           0.253489 |        2.31517 |            5.13684 |  0.782609 |
| id      | coolest_core_observed         | load_15   |           3.28571  |       42.1304  |            5.12739 |  0.645963 |
| id      | uncalibrated_quantile         | power_0   |           0.328763 |        2.90667 |            4.99408 |  0.752588 |
| id      | trend_aware_observed          | load_15   |           3.28571  |       40.6304  |            4.9294  |  0.535197 |
| id      | point_prediction_rf           | power_5   |           0.216225 |        1.85576 |            4.78359 |  0.76087  |
| id      | trend_aware_observed          | power_0   |           0.328763 |        2.78545 |            4.75924 |  0.70911  |
| id      | coolest_core_oracle_true_temp | power_10  |           0.16825  |        1.60214 |            4.68788 |  0.478261 |

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
