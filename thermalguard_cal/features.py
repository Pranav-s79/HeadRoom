from __future__ import annotations

import numpy as np

from .config import ThermalGuardConfig
from .sensors import impute_sensor_readings
from .utils import ChipState, Task, core_to_rc, neighbor_indices


def feature_names(cfg: ThermalGuardConfig) -> list[str]:
    n = cfg.num_cores
    names: list[str] = []
    names += [f"sensor_temp_imputed_{i}" for i in range(n)]
    names += [f"sensor_mask_{i}" for i in range(n)]
    names += [f"power_{i}" for i in range(n)]
    names += [f"load_{i}" for i in range(n)]
    names += [f"task_count_{i}" for i in range(n)]
    names += [f"observed_trend_{i}" for i in range(n)]
    names += [f"candidate_core_{i}" for i in range(n)]
    names += [
        "candidate_row_norm",
        "candidate_col_norm",
        "candidate_task_power",
        "candidate_task_duration_norm",
        "total_power",
        "total_load",
        "max_observed_temp",
        "mean_observed_temp",
        "candidate_estimated_temp",
        "candidate_current_power",
        "candidate_current_load",
        "neighbor_power_mean",
        "neighbor_load_mean",
        "neighbor_observed_temp_mean",
        "active_task_total",
        "time_norm",
    ]
    return names


def build_candidate_features(
    state: ChipState,
    task: Task,
    candidate_core: int,
    cfg: ThermalGuardConfig,
) -> np.ndarray:
    """Build model features from sparse sensor observations and workload metadata only."""
    n = cfg.num_cores
    if not 0 <= candidate_core < n:
        raise ValueError(f"candidate_core must be in [0, {n}), got {candidate_core}")

    sensor_values = impute_sensor_readings(
        state.sensor_readings,
        state.sensor_mask,
        cfg.ambient_temp,
    )
    sensor_mask = np.asarray(state.sensor_mask, dtype=float)
    power = np.asarray(state.power, dtype=float)
    load = np.asarray(state.load, dtype=float)
    task_counts = np.asarray(state.task_counts, dtype=float)
    trend = np.asarray(state.observed_temp_trend, dtype=float)
    trend = np.where(np.isfinite(trend), trend, 0.0)
    trend = trend * sensor_mask

    candidate_one_hot = np.zeros(n, dtype=float)
    candidate_one_hot[candidate_core] = 1.0
    row, col = core_to_rc(candidate_core, cfg.grid_size)
    neighbors = neighbor_indices(candidate_core, cfg.grid_size)

    observed_valid = sensor_values[np.asarray(state.sensor_mask, dtype=bool)]
    if observed_valid.size == 0:
        max_observed = cfg.ambient_temp
        mean_observed = cfg.ambient_temp
    else:
        max_observed = float(np.max(observed_valid))
        mean_observed = float(np.mean(observed_valid))

    if neighbors:
        neighbor_power = float(np.mean(power[neighbors]))
        neighbor_load = float(np.mean(load[neighbors]))
        neighbor_temp = float(np.mean(sensor_values[neighbors]))
    else:
        neighbor_power = 0.0
        neighbor_load = 0.0
        neighbor_temp = cfg.ambient_temp

    scalars = np.array(
        [
            row / max(1, cfg.grid_size - 1),
            col / max(1, cfg.grid_size - 1),
            float(task.power),
            float(task.duration) / 100.0,
            float(np.sum(power)),
            float(np.sum(load)),
            max_observed,
            mean_observed,
            float(sensor_values[candidate_core]),
            float(power[candidate_core]),
            float(load[candidate_core]),
            neighbor_power,
            neighbor_load,
            neighbor_temp,
            float(np.sum(task_counts)),
            float(state.time_index) / max(1, cfg.episode_length),
        ],
        dtype=float,
    )

    features = np.concatenate(
        [
            sensor_values,
            sensor_mask,
            power,
            load,
            task_counts,
            trend,
            candidate_one_hot,
            scalars,
        ]
    )
    expected = len(feature_names(cfg))
    if features.shape[0] != expected:
        raise RuntimeError(f"Feature length {features.shape[0]} != expected {expected}")
    return features.astype(float)


def build_candidate_matrix(
    state: ChipState,
    task: Task,
    cfg: ThermalGuardConfig,
) -> np.ndarray:
    return np.vstack(
        [build_candidate_features(state, task, core, cfg) for core in range(cfg.num_cores)]
    )
