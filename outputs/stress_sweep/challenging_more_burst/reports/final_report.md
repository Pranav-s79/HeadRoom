# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:17:39+00:00

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

Schedulers with at least one hotspot violation in any evaluated split: coolest_core_observed, trend_aware_observed.
Schedulers with zero hotspot violations across all evaluated splits: conformal_upper_bound, coolest_core_oracle_true_temp, point_prediction_rf, random, round_robin, uncalibrated_quantile.
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
| calibration | linear_point    | mae                  |  1.28797  |
| calibration | linear_point    | rmse                 |  1.85648  |
| calibration | linear_point    | max_abs_error        |  5.88279  |
| calibration | forest_point    | mae                  |  1.64358  |
| calibration | forest_point    | rmse                 |  2.07359  |
| calibration | forest_point    | max_abs_error        |  4.42575  |
| calibration | quantile_upper  | empirical_coverage   |  0.924513 |
| calibration | quantile_upper  | average_bound        | 46.2327   |
| calibration | quantile_upper  | average_conservatism |  4.89642  |
| calibration | conformal_upper | empirical_coverage   |  0.924513 |
| calibration | conformal_upper | average_bound        | 46.2327   |
| calibration | conformal_upper | average_conservatism |  4.89642  |
| test_id     | linear_point    | mae                  |  2.66163  |
| test_id     | linear_point    | rmse                 |  3.20406  |
| test_id     | linear_point    | max_abs_error        |  6.73356  |
| test_id     | forest_point    | mae                  |  1.65782  |
| test_id     | forest_point    | rmse                 |  2.11577  |
| test_id     | forest_point    | max_abs_error        |  6.19886  |
| test_id     | quantile_upper  | empirical_coverage   |  0.798333 |
| test_id     | quantile_upper  | average_bound        | 45.934    |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |       value |
|:------------------------------------------------|:------------|------------:|
| target_coverage                                 | all         |    0.9      |
| quantile_model_alpha                            | all         |    0.9      |
| conformal_correction                            | calibration |    0        |
| conformal_quantile_level                        | calibration |    0.900974 |
| calibration_samples                             | calibration | 1232        |
| calibration_empirical_coverage_before_conformal | calibration |    0.924513 |
| calibration_empirical_coverage_after_conformal  | calibration |    0.924513 |
| id_empirical_coverage_before_conformal          | test_id     |    0.798333 |
| id_empirical_coverage_after_conformal           | test_id     |    0.798333 |
| ood_empirical_coverage_before_conformal         | test_ood    |    0.544471 |
| ood_empirical_coverage_after_conformal          | test_ood    |    0.544471 |

In this run, the quantile model was already conservative on the calibration split, so conformal calibration verified the bound but did not widen it.


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            64.0814 |                   42.9704 |                    0 |                28 |               75 |                 nan |                      nan |           0.594339 |       0.76     |
| round_robin                   | deployable_baseline_sensor_observed |            49.7815 |                   39.5088 |                    0 |                28 |               75 |                 nan |                      nan |           0.349299 |       0.709437 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            65.5646 |                   45.1712 |                    0 |                28 |               75 |                 nan |                      nan |           1.29094  |       0.948052 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            50.7332 |                   39.9594 |                    0 |                28 |               75 |                 nan |                      nan |           0.336939 |       0.736104 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            65.6325 |                   45.1743 |                    0 |                28 |               75 |                 nan |                      nan |           1.29102  |       0.948052 |
| point_prediction_rf           | model_based_sensor_observed         |            47.284  |                   39.716  |                    0 |                28 |               75 |                 nan |                      nan |           0.383075 |       0.746667 |
| uncalibrated_quantile         | model_based_sensor_observed         |            52.1448 |                   43.2131 |                    0 |                28 |               75 |                 nan |                      nan |           0.712605 |       0.922078 |
| conformal_upper_bound         | model_based_sensor_observed         |            47.5133 |                   39.1739 |                    0 |                28 |               75 |                   1 |                        1 |           0.388305 |       0.786667 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            75.065  |                   50.7513 |                    0 |                40 |              104 |          nan        |               nan        |           1.12212  |       0.961538 |
| round_robin                   | deployable_baseline_sensor_observed |            68.7522 |                   47.3824 |                    0 |                40 |              104 |          nan        |               nan        |           1.13875  |       0.942308 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            93.3454 |                   58.1571 |                    8 |                40 |              104 |          nan        |               nan        |           2.46481  |       0.951923 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            61.4923 |                   48.2045 |                    0 |                40 |              104 |          nan        |               nan        |           1.094    |       0.817308 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            91.6303 |                   58.5911 |                   10 |                40 |              104 |          nan        |               nan        |           2.33698  |       0.951923 |
| point_prediction_rf           | model_based_sensor_observed         |            54.0963 |                   43.6588 |                    0 |                40 |              104 |          nan        |               nan        |           0.929295 |       0.8745   |
| uncalibrated_quantile         | model_based_sensor_observed         |            65.6851 |                   48.5945 |                    0 |                40 |              104 |          nan        |               nan        |           1.05244  |       0.932692 |
| conformal_upper_bound         | model_based_sensor_observed         |            56.7686 |                   43.6335 |                    0 |                40 |              104 |            0.767428 |                 0.759615 |           0.973379 |       0.951923 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             1        | 1200 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             1        |   75 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.767428 | 1664 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.759615 |  104 |

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
| challenging_id | random                        |            64.0814 |                   42.9704 |                    0 |                28 |                      nan |
| challenging_id | round_robin                   |            49.7815 |                   39.5088 |                    0 |                28 |                      nan |
| challenging_id | coolest_core_observed         |            65.5646 |                   45.1712 |                    0 |                28 |                      nan |
| challenging_id | coolest_core_oracle_true_temp |            50.7332 |                   39.9594 |                    0 |                28 |                      nan |
| challenging_id | trend_aware_observed          |            65.6325 |                   45.1743 |                    0 |                28 |                      nan |
| challenging_id | point_prediction_rf           |            47.284  |                   39.716  |                    0 |                28 |                      nan |
| challenging_id | uncalibrated_quantile         |            52.1448 |                   43.2131 |                    0 |                28 |                      nan |
| challenging_id | conformal_upper_bound         |            47.5133 |                   39.1739 |                    0 |                28 |                        1 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | trend_aware_observed          |           1.29102  |       0.948052 |
| id      | coolest_core_observed         |           1.29094  |       0.948052 |
| id      | uncalibrated_quantile         |           0.712605 |       0.922078 |
| id      | random                        |           0.594339 |       0.76     |
| id      | conformal_upper_bound         |           0.388305 |       0.786667 |
| id      | point_prediction_rf           |           0.383075 |       0.746667 |
| id      | round_robin                   |           0.349299 |       0.709437 |
| id      | coolest_core_oracle_true_temp |           0.336939 |       0.736104 |
| ood     | coolest_core_observed         |           2.46481  |       0.951923 |
| ood     | trend_aware_observed          |           2.33698  |       0.951923 |
| ood     | round_robin                   |           1.13875  |       0.942308 |
| ood     | random                        |           1.12212  |       0.961538 |
| ood     | coolest_core_oracle_true_temp |           1.094    |       0.817308 |
| ood     | uncalibrated_quantile         |           1.05244  |       0.932692 |
| ood     | conformal_upper_bound         |           0.973379 |       0.951923 |
| ood     | point_prediction_rf           |           0.929295 |       0.8745   |

Largest feature-level shifts:

| split   | scheduler                     | feature               |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:------------------------------|:----------------------|-------------------:|---------------:|-------------------:|----------:|
| id      | random                        | power_4               |           0.363851 |        6.44018 |           17.6416  |  0.76     |
| id      | uncalibrated_quantile         | load_2                |           9.53247  |       84.7333  |            7.82965 |  0.866667 |
| id      | random                        | load_4                |           8.01299  |       71.7867  |            7.17157 |  0.733333 |
| id      | uncalibrated_quantile         | power_2               |           0.799968 |        5.48382 |            6.19027 |  0.85368  |
| id      | trend_aware_observed          | power_12              |           1.74756  |        6.6346  |            5.98711 |  0.629437 |
| id      | trend_aware_observed          | power_0               |           1.44681  |        7.53339 |            5.50835 |  0.72     |
| id      | coolest_core_observed         | power_12              |           1.74756  |        6.07825 |            5.30553 |  0.709437 |
| id      | coolest_core_observed         | power_0               |           1.44681  |        7.16212 |            5.17235 |  0.613333 |
| id      | trend_aware_observed          | load_3                |          18.2987   |      112.947   |            4.78821 |  0.787013 |
| id      | point_prediction_rf           | power_4               |           0.363851 |        1.90844 |            4.48445 |  0.746667 |
| id      | coolest_core_observed         | load_3                |          18.2987   |      104.76    |            4.37405 |  0.653333 |
| id      | trend_aware_observed          | power_3               |           1.26981  |        6.90384 |            4.36178 |  0.786667 |
| id      | coolest_core_observed         | sensor_temp_imputed_3 |          36.5699   |       44.5122  |            4.19968 |  0.631169 |
| id      | trend_aware_observed          | sensor_temp_imputed_3 |          36.5699   |       44.4064  |            4.14372 |  0.524502 |
| id      | trend_aware_observed          | load_12               |          27.6753   |       96.8267  |            3.91811 |  0.508398 |
| id      | conformal_upper_bound         | load_1                |          11.4026   |       53.32    |            3.90848 |  0.680346 |
| id      | coolest_core_observed         | power_3               |           1.26981  |        6.23099 |            3.84087 |  0.64     |
| id      | round_robin                   | power_4               |           0.363851 |        1.58008 |            3.53111 |  0.613333 |
| id      | coolest_core_oracle_true_temp | load_2                |           9.53247  |       41.88    |            3.36791 |  0.334026 |
| id      | point_prediction_rf           | load_1                |          11.4026   |       46.5867  |            3.28065 |  0.600346 |

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
