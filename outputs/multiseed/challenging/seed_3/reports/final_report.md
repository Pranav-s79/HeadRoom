# ThermalGuard-Cal Final Report

Generated: 2026-06-19T05:04:20+00:00

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
| calibration | linear_point    | mae                  |  1.35763  |
| calibration | linear_point    | rmse                 |  1.85338  |
| calibration | linear_point    | max_abs_error        |  5.88894  |
| calibration | forest_point    | mae                  |  1.48249  |
| calibration | forest_point    | rmse                 |  2.11637  |
| calibration | forest_point    | max_abs_error        |  7.71752  |
| calibration | quantile_upper  | empirical_coverage   |  0.760417 |
| calibration | quantile_upper  | average_bound        | 46.5704   |
| calibration | quantile_upper  | average_conservatism |  3.03759  |
| calibration | conformal_upper | empirical_coverage   |  0.900768 |
| calibration | conformal_upper | average_bound        | 48.4798   |
| calibration | conformal_upper | average_conservatism |  4.94703  |
| test_id     | linear_point    | mae                  |  3.47817  |
| test_id     | linear_point    | rmse                 |  3.98284  |
| test_id     | linear_point    | max_abs_error        |  8.3425   |
| test_id     | forest_point    | mae                  |  2.82478  |
| test_id     | forest_point    | rmse                 |  3.64945  |
| test_id     | forest_point    | max_abs_error        |  9.06248  |
| test_id     | quantile_upper  | empirical_coverage   |  0.502694 |
| test_id     | quantile_upper  | average_bound        | 46.9177   |

## Conformal Calibration Diagnostics

Target coverage, quantile alpha, conformal correction, calibration sample count,
and before/after empirical coverage are reported explicitly below.

| metric                                          | split       |       value |
|:------------------------------------------------|:------------|------------:|
| target_coverage                                 | all         |    0.9      |
| quantile_model_alpha                            | all         |    0.9      |
| conformal_correction                            | calibration |    1.90944  |
| conformal_quantile_level                        | calibration |    0.900768 |
| calibration_samples                             | calibration | 1824        |
| calibration_empirical_coverage_before_conformal | calibration |    0.760417 |
| calibration_empirical_coverage_after_conformal  | calibration |    0.900768 |
| id_empirical_coverage_before_conformal          | test_id     |    0.502694 |
| id_empirical_coverage_after_conformal           | test_id     |    0.663793 |
| ood_empirical_coverage_before_conformal         | test_ood    |    0.427632 |
| ood_empirical_coverage_after_conformal          | test_ood    |    0.497122 |


## ID Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            61.3468 |                   46.2266 |                    0 |                75 |              116 |          nan        |               nan        |           0.35764  |       0.66092  |
| round_robin                   | deployable_baseline_sensor_observed |            57.8381 |                   44.9688 |                    0 |                75 |              116 |          nan        |               nan        |           0.217374 |       0.636872 |
| coolest_core_observed         | deployable_baseline_sensor_observed |            75.3902 |                   53.728  |                    0 |                75 |              116 |          nan        |               nan        |           1.80493  |       0.95614  |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            53.2475 |                   42.9331 |                    0 |                75 |              116 |          nan        |               nan        |           0.261236 |       0.617665 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            80.1143 |                   52.0857 |                    0 |                75 |              116 |          nan        |               nan        |           1.6164   |       0.95614  |
| point_prediction_rf           | model_based_sensor_observed         |            51.5228 |                   42.5283 |                    0 |                75 |              116 |          nan        |               nan        |           0.232346 |       0.583938 |
| uncalibrated_quantile         | model_based_sensor_observed         |            55.854  |                   45.9182 |                    0 |                75 |              116 |          nan        |               nan        |           0.466312 |       0.824561 |
| conformal_upper_bound         | model_based_sensor_observed         |            51.7072 |                   42.6428 |                    0 |                75 |              116 |            0.973599 |                 0.974138 |           0.314617 |       0.948276 |

## OOD Scheduler Metrics

| scheduler                     | baseline_type                       |   peak_temperature |   average_max_temperature |   hotspot_violations |   completed_tasks |   assigned_tasks |   marginal_coverage |   selected_core_coverage |   drift_mean_abs_z |   drift_max_ks |
|:------------------------------|:------------------------------------|-------------------:|--------------------------:|---------------------:|------------------:|-----------------:|--------------------:|-------------------------:|-------------------:|---------------:|
| random                        | deployable_baseline_sensor_observed |            75.2847 |                   53.3993 |                    0 |                66 |              152 |          nan        |               nan        |           1.03596  |       0.907895 |
| round_robin                   | deployable_baseline_sensor_observed |            69.2268 |                   47.9556 |                    0 |                66 |              152 |          nan        |               nan        |           0.953923 |       0.789474 |
| coolest_core_observed         | deployable_baseline_sensor_observed |           102.507  |                   61.9979 |                   32 |                66 |              152 |          nan        |               nan        |           3.35484  |       0.967105 |
| coolest_core_oracle_true_temp | oracle_privileged_true_temperature  |            59.7581 |                   47.9108 |                    0 |                66 |              152 |          nan        |               nan        |           1.0095   |       0.767544 |
| trend_aware_observed          | deployable_baseline_sensor_observed |            99.0396 |                   62.1233 |                   35 |                66 |              152 |          nan        |               nan        |           3.08746  |       0.95614  |
| point_prediction_rf           | model_based_sensor_observed         |            60.625  |                   45.0507 |                    0 |                66 |              152 |          nan        |               nan        |           1.06141  |       0.894737 |
| uncalibrated_quantile         | model_based_sensor_observed         |            63.9517 |                   48.6464 |                    0 |                66 |              152 |          nan        |               nan        |           1.00071  |       0.85307  |
| conformal_upper_bound         | model_based_sensor_observed         |            59.9554 |                   45.377  |                    0 |                66 |              152 |            0.616776 |                 0.611842 |           1.0281   |       0.967105 |

## Coverage Metrics

| split   | scheduler             | coverage_type                             |   nominal_coverage |   empirical_coverage |    n |
|:--------|:----------------------|:------------------------------------------|-------------------:|---------------------:|-----:|
| id      | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.973599 | 1856 |
| id      | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.974138 |  116 |
| ood     | conformal_upper_bound | marginal_all_candidates_on_visited_states |                0.9 |             0.616776 | 2432 |
| ood     | conformal_upper_bound | selected_core_after_scheduler_selection   |                0.9 |             0.611842 |  152 |

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
| ood     | conformal_upper_bound     | point_prediction_rf         | conformal_upper_bound    | safer in this split                    |

- For ID, point_prediction_rf has the lowest peak temperature and point_prediction_rf has the fewest hotspot violations.
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
| challenging_id | random                        |            61.3468 |                   46.2266 |                    0 |                75 |               nan        |
| challenging_id | round_robin                   |            57.8381 |                   44.9688 |                    0 |                75 |               nan        |
| challenging_id | coolest_core_observed         |            75.3902 |                   53.728  |                    0 |                75 |               nan        |
| challenging_id | coolest_core_oracle_true_temp |            53.2475 |                   42.9331 |                    0 |                75 |               nan        |
| challenging_id | trend_aware_observed          |            80.1143 |                   52.0857 |                    0 |                75 |               nan        |
| challenging_id | point_prediction_rf           |            51.5228 |                   42.5283 |                    0 |                75 |               nan        |
| challenging_id | uncalibrated_quantile         |            55.854  |                   45.9182 |                    0 |                75 |               nan        |
| challenging_id | conformal_upper_bound         |            51.7072 |                   42.6428 |                    0 |                75 |                 0.974138 |






## Policy-Induced Distribution Drift

Scheduler-level drift summary:

| split   | scheduler                     |   drift_mean_abs_z |   drift_max_ks |
|:--------|:------------------------------|-------------------:|---------------:|
| id      | coolest_core_observed         |           1.80493  |       0.95614  |
| id      | trend_aware_observed          |           1.6164   |       0.95614  |
| id      | uncalibrated_quantile         |           0.466312 |       0.824561 |
| id      | random                        |           0.35764  |       0.66092  |
| id      | conformal_upper_bound         |           0.314617 |       0.948276 |
| id      | coolest_core_oracle_true_temp |           0.261236 |       0.617665 |
| id      | point_prediction_rf           |           0.232346 |       0.583938 |
| id      | round_robin                   |           0.217374 |       0.636872 |
| ood     | coolest_core_observed         |           3.35484  |       0.967105 |
| ood     | trend_aware_observed          |           3.08746  |       0.95614  |
| ood     | point_prediction_rf           |           1.06141  |       0.894737 |
| ood     | random                        |           1.03596  |       0.907895 |
| ood     | conformal_upper_bound         |           1.0281   |       0.967105 |
| ood     | coolest_core_oracle_true_temp |           1.0095   |       0.767544 |
| ood     | uncalibrated_quantile         |           1.00071  |       0.85307  |
| ood     | round_robin                   |           0.953923 |       0.789474 |

Largest feature-level shifts:

| split   | scheduler             | feature                |   calibration_mean |   visited_mean |   abs_mean_z_shift |   ks_stat |
|:--------|:----------------------|:-----------------------|-------------------:|---------------:|-------------------:|----------:|
| id      | coolest_core_observed | load_12                |           10.0614  |      130.147   |           11.8069  |  0.861918 |
| id      | coolest_core_observed | power_12               |            0.80488 |        8.20657 |           10.2238  |  0.913793 |
| id      | trend_aware_observed  | load_12                |           10.0614  |       97.2931  |            8.57675 |  0.818966 |
| id      | trend_aware_observed  | power_12               |            0.80488 |        6.81593 |            8.30291 |  0.862069 |
| id      | coolest_core_observed | sensor_temp_imputed_12 |           36.9675  |       48.2363  |            6.75028 |  0.63778  |
| id      | trend_aware_observed  | load_15                |           19.2719  |      151.603   |            6.68281 |  0.732305 |
| id      | coolest_core_observed | power_3                |            1.38595 |        8.57897 |            6.59151 |  0.767241 |
| id      | coolest_core_observed | load_3                 |           18.1228  |      143.172   |            6.39837 |  0.80974  |
| id      | trend_aware_observed  | sensor_temp_imputed_12 |           36.9675  |       47.3539  |            6.22167 |  0.619782 |
| id      | trend_aware_observed  | power_15               |            1.55803 |        8.70118 |            5.3236  |  0.774955 |
| id      | trend_aware_observed  | power_3                |            1.38595 |        7.09196 |            5.22884 |  0.826376 |
| id      | coolest_core_observed | sensor_temp_imputed_1  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_2  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_4  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_5  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_6  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_7  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_8  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_9  |           38.2721  |       48.9688  |            4.57303 |  0.568966 |
| id      | coolest_core_observed | sensor_temp_imputed_10 |           38.2721  |       48.9688  |            4.57303 |  0.568966 |

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
