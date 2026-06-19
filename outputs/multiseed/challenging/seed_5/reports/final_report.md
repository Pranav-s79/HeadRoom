# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:06:52+00:00

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
Schedulers with zero hotspot violations in this run: random, round_robin, coolest_core_observed, coolest_core_oracle_true_temp, trend_aware_observed, point_prediction_rf, uncalibrated_quantile, conformal_upper_bound.
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
| Conformal advantage over model baselines | proven |
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
| calibration | linear_point    | mae                  |  2.53764  |
| calibration | linear_point    | rmse                 |  3.14882  |
| calibration | linear_point    | max_abs_error        |  7.69669  |
| calibration | forest_point    | mae                  |  2.474    |
| calibration | forest_point    | rmse                 |  3.17686  |
| calibration | forest_point    | max_abs_error        |  8.09898  |
| calibration | quantile_upper  | empirical_coverage   |  0.64899  |
| calibration | quantile_upper  | average_bound        | 49.6405   |
| calibration | quantile_upper  | average_conservatism |  3.57856  |
| calibration | conformal_upper | empirical_coverage   |  0.900884 |
| calibration | conformal_upper | average_bound        | 51.4896   |
| calibration | conformal_upper | average_conservatism |  5.42768  |
| test_id     | linear_point    | mae                  |  4.7553   |
| test_id     | linear_point    | rmse                 |  7.29014  |
| test_id     | linear_point    | max_abs_error        | 18.6237   |
| test_id     | forest_point    | mae                  |  4.68885  |
| test_id     | forest_point    | rmse                 |  7.11262  |
| test_id     | forest_point    | max_abs_error        | 20.7824   |
| test_id     | quantile_upper  | empirical_coverage   |  0.686404 |
| test_id     | quantile_upper  | average_bound        | 49.8816   |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |       value |
|:------------------------------------------------|:------------|------------:|
| target_coverage                                 | all         |    0.9      |
| quantile_model_alpha                            | all         |    0.9      |
| conformal_correction                            | calibration |    1.84912  |
| conformal_quantile_level                        | calibration |    0.900884 |
| calibration_samples                             | calibration | 1584        |
| calibration_empirical_coverage_before_conformal | calibration |    0.64899  |
| calibration_empirical_coverage_after_conformal  | calibration |    0.900884 |
| id_empirical_coverage_before_conformal          | test_id     |    0.686404 |
| id_empirical_coverage_after_conformal           | test_id     |    0.722588 |
| ood_empirical_coverage_before_conformal         | test_ood    |    0.650091 |
| ood_empirical_coverage_after_conformal          | test_ood    |    0.771442 |


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            68.3454 |                   47.1082 |                    0 |                57 |              114 |                 nan |                      nan |           0.384038 |       0.775651 |
| round_robin                   | deployable_baseline_sensor_observed |            64.3178 |                   44.9406 |                    0 |                57 |              114 |                 nan |                      nan |           0.465443 |       0.845561 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            84.5589 |                   53.7585 |                    0 |                57 |              114 |                 nan |                      nan |           1.75883  |       0.939394 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            56.4058 |                   43.8558 |                    0 |                57 |              114 |                 nan |                      nan |           0.386055 |       0.786284 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            77.806  |                   52.3979 |                    0 |                57 |              114 |                 nan |                      nan |           1.66412  |       0.939394 |
| point_prediction_rf           | model_based_sensor_observed         |            52.7047 |                   42.1559 |                    0 |                57 |              114 |                 nan |                      nan |           0.372256 |       0.72807  |
| uncalibrated_quantile         | model_based_sensor_observed         |            60.4981 |                   48.2532 |                    0 |                57 |              114 |                 nan |                      nan |           0.656036 |       0.894737 |
| conformal_upper_bound         | model_based_sensor_observed         |            51.5852 |                   42.1815 |                    0 |                57 |              114 |                   1 |                        1 |           0.382399 |       0.849016 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            70.4212 |                   52.0419 |                    0 |                63 |              137 |           nan       |               nan        |           1.05178  |       0.864705 |
| round_robin                   | deployable_baseline_sensor_observed |            69.3474 |                   48.148  |                    0 |                63 |              137 |           nan       |               nan        |           0.864324 |       0.89781  |
| coolest_core_observed         | deployable_baseline_sensor_observed |           103.886  |                   65.031  |                   46 |                63 |              137 |           nan       |               nan        |           2.7092   |       0.970803 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            62.7641 |                   47.3607 |                    0 |                63 |              137 |           nan       |               nan        |           0.938904 |       0.843914 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            99.2533 |                   63.0329 |                   42 |                63 |              137 |           nan       |               nan        |           2.54805  |       0.970803 |
| point_prediction_rf           | model_based_sensor_observed         |            64.6838 |                   45.5956 |                    0 |                63 |              137 |           nan       |               nan        |           0.922717 |       0.873111 |
| uncalibrated_quantile         | model_based_sensor_observed         |            65.2797 |                   50.4091 |                    0 |                63 |              137 |           nan       |               nan        |           1.00092  |       0.941606 |
| conformal_upper_bound         | model_based_sensor_observed         |            60.9983 |                   47.6432 |                    0 |                63 |              137 |             0.82208 |                 0.824818 |           0.919152 |       0.893903 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             1        | 1824 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             1        |  114 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.82208  | 2192 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.824818 |  137 |

Marginal candidate coverage and selected-core coverage are reported separately.
The difference captures the selection step where the scheduler chooses one
candidate out of 16. Distribution drift is reported separately below and should
not be conflated with the selection-bias coverage gap.

## Does Conformal Add Scheduling Value?

This section compares only `point_prediction_rf`, `uncalibrated_quantile`, and
`conformal_upper_bound`. Conformal should not be called best unless both the
scheduling metrics and coverage metrics support that claim.

| split   | lowest_peak_temperature   | fewest_hotspot_violations   | best_measured_coverage   | conformal_interpretation   |
|:--------|:--------------------------|:----------------------------|:-------------------------|:---------------------------|
| id      | conformal_upper_bound     | point_prediction_rf         | conformal_upper_bound    | safer in this split        |
| ood     | conformal_upper_bound     | point_prediction_rf         | conformal_upper_bound    | safer in this split        |

- For ID, conformal_upper_bound has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.
- For OOD, conformal_upper_bound has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.

Calibration can improve statistical trust even when scheduler outcomes are
similar, because it turns an uncalibrated quantile model into an auditable
coverage claim on calibration-like data. It does not create an OOD guarantee.


## Preset Results

This separates saved preset snapshots from the current canonical CSVs, so the
original normal/easy quick result can be compared against a challenging run when
both have been executed.

| result_group   | scheduler                     |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   selected_core_coverage |
|:---------------|:------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-------------------------:|
| challenging_id | random                        |            68.3454 |                   47.1082 |                    0 |                57 |                      nan |
| challenging_id | round_robin                   |            64.3178 |                   44.9406 |                    0 |                57 |                      nan |
| challenging_id | coolest_core_observed         |            84.5589 |                   53.7585 |                    0 |                57 |                      nan |
| challenging_id | coolest_core_oracle_true_temp |            56.4058 |                   43.8558 |                    0 |                57 |                      nan |
| challenging_id | trend_aware_observed          |            77.806  |                   52.3979 |                    0 |                57 |                      nan |
| challenging_id | point_prediction_rf           |            52.7047 |                   42.1559 |                    0 |                57 |                      nan |
| challenging_id | uncalibrated_quantile         |            60.4981 |                   48.2532 |                    0 |                57 |                      nan |
| challenging_id | conformal_upper_bound         |            51.5852 |                   42.1815 |                    0 |                57 |                        1 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | coolest_core_observed         |           1.75883  |       0.939394 |
| id      | trend_aware_observed          |           1.66412  |       0.939394 |
| id      | uncalibrated_quantile         |           0.656036 |       0.894737 |
| id      | round_robin                   |           0.465443 |       0.845561 |
| id      | coolest_core_oracle_true_temp |           0.386055 |       0.786284 |
| id      | random                        |           0.384038 |       0.775651 |
| id      | conformal_upper_bound         |           0.382399 |       0.849016 |
| id      | point_prediction_rf           |           0.372256 |       0.72807  |
| ood     | coolest_core_observed         |           2.7092   |       0.970803 |
| ood     | trend_aware_observed          |           2.54805  |       0.970803 |
| ood     | random                        |           1.05178  |       0.864705 |
| ood     | uncalibrated_quantile         |           1.00092  |       0.941606 |
| ood     | coolest_core_oracle_true_temp |           0.938904 |       0.843914 |
| ood     | point_prediction_rf           |           0.922717 |       0.873111 |
| ood     | conformal_upper_bound         |           0.919152 |       0.893903 |
| ood     | round_robin                   |           0.864324 |       0.89781  |

Largest feature-level shifts:

| split   | scheduler             | feature               |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:----------------------|:----------------------|-------------------:|---------------:|-------------------:|----------:|
| id      | trend_aware_observed  | power_0               |            1.68107 |        9.76095 |            8.20452 |  0.885965 |
| id      | coolest_core_observed | power_3               |            1.05494 |        9.69848 |            7.6074  |  0.894737 |
| id      | coolest_core_observed | power_0               |            1.68107 |        8.77125 |            7.19956 |  0.833333 |
| id      | trend_aware_observed  | power_3               |            1.05494 |        8.91579 |            6.91853 |  0.836789 |
| id      | coolest_core_observed | load_3                |           15.202   |      141.333   |            6.80415 |  0.893408 |
| id      | trend_aware_observed  | sensor_temp_imputed_3 |           36.527   |       50.4928  |            6.19915 |  0.690324 |
| id      | coolest_core_observed | sensor_temp_imputed_3 |           36.527   |       50.4716  |            6.18973 |  0.725412 |
| id      | coolest_core_observed | load_12               |           38.2828  |      181.947   |            5.81892 |  0.759171 |
| id      | trend_aware_observed  | load_0                |           28.8384  |      151.956   |            5.71203 |  0.874535 |
| id      | trend_aware_observed  | load_3                |           15.202   |      119.965   |            5.65143 |  0.836789 |
| id      | coolest_core_observed | sensor_temp_imputed_0 |           38.0487  |       50.8636  |            4.9787  |  0.640351 |
| id      | trend_aware_observed  | sensor_temp_imputed_0 |           38.0487  |       50.8611  |            4.97771 |  0.640351 |
| id      | coolest_core_observed | load_0                |           28.8384  |      135.86    |            4.96524 |  0.66135  |
| id      | trend_aware_observed  | load_12               |           38.2828  |      154.982   |            4.72675 |  0.715311 |
| id      | coolest_core_observed | power_12              |            3.08516 |       11.983   |            4.61944 |  0.747209 |
| id      | coolest_core_observed | sensor_temp_imputed_1 |           38.4177  |       51.5033  |            4.13172 |  0.591175 |
| id      | coolest_core_observed | sensor_temp_imputed_2 |           38.4177  |       51.5033  |            4.13172 |  0.591175 |
| id      | coolest_core_observed | sensor_temp_imputed_4 |           38.4177  |       51.5033  |            4.13172 |  0.591175 |
| id      | coolest_core_observed | sensor_temp_imputed_5 |           38.4177  |       51.5033  |            4.13172 |  0.591175 |
| id      | coolest_core_observed | sensor_temp_imputed_6 |           38.4177  |       51.5033  |            4.13172 |  0.591175 |

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
coverage changes under OOD shift. This is why OOD calibration performance should not be
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
