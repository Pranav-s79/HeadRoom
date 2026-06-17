from __future__ import annotations

import numpy as np

from .config import ThermalGuardConfig


class SensorModel:
    """Sparse noisy thermal sensor model with NaN for unavailable readings."""

    def __init__(
        self,
        cfg: ThermalGuardConfig,
        dropout_prob: float | None = None,
        noise_sigma: float | None = None,
    ) -> None:
        self.cfg = cfg
        self.sensor_indices = np.array(cfg.sensor_indices, dtype=int)
        self.dropout_prob = cfg.sensor_dropout_prob if dropout_prob is None else float(dropout_prob)
        self.noise_sigma = cfg.sensor_noise_sigma if noise_sigma is None else float(noise_sigma)

    def observe(
        self, true_temperatures: np.ndarray, rng: np.random.Generator
    ) -> tuple[np.ndarray, np.ndarray]:
        readings = np.full(self.cfg.num_cores, np.nan, dtype=float)
        mask = np.zeros(self.cfg.num_cores, dtype=bool)
        for core in self.sensor_indices:
            if rng.random() < self.dropout_prob:
                continue
            readings[core] = true_temperatures[core] + rng.normal(0.0, self.noise_sigma)
            mask[core] = True
        return readings, mask


def impute_sensor_readings(
    sensor_readings: np.ndarray,
    sensor_mask: np.ndarray,
    ambient_temp: float,
) -> np.ndarray:
    values = np.asarray(sensor_readings, dtype=float).copy()
    mask = np.asarray(sensor_mask, dtype=bool)
    values[~mask] = np.nan
    if np.any(mask):
        fill_value = float(np.nanmean(values[mask]))
    else:
        fill_value = float(ambient_temp)
    values = np.where(np.isfinite(values), values, fill_value)
    return values
