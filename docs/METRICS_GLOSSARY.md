# Metrics Glossary

## peak_temperature

The highest chip temperature reached during a scheduler rollout. Lower is
better.

## average_max_temperature

The average, over timesteps, of the hottest core temperature. This captures
steady thermal behavior better than one isolated peak.

## hotspot_violations

The number of timesteps where the chip exceeded the thermal limit.

## hotspot_timestep_pct

The fraction of evaluated timesteps that exceeded the thermal limit.

## completed_tasks

The number of tasks completed by the simulator. This helps check whether a
scheduler reduced temperature by sacrificing throughput.

## marginal_coverage

Coverage across all candidate-core predictions on visited states. This is close
to the usual conformal candidate-level view.

## selected_core_coverage

Coverage only on the core selected by the scheduler. This is the deployed
decision-path coverage.

## selected_coverage_gap

Selected-core empirical coverage minus the nominal target coverage.

## conformal_correction

The additive widening applied to the quantile model on the calibration split.
If this is zero, the base quantile model was already conservative on the
calibration split. If it is positive, conformal widened the upper bound.

## drift_mean_abs_z

Average absolute feature shift between scheduler rollout states and calibration
features. Higher values indicate the policy is visiting less calibration-like
states.
