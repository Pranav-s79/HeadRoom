from __future__ import annotations

import copy

import numpy as np

from .config import ThermalGuardConfig
from .sensors import SensorModel
from .utils import ChipState, Task, neighbor_indices


class ThermalSimulator:
    """Discrete-time 4x4 many-core thermal simulator."""

    def __init__(self, cfg: ThermalGuardConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed if seed is None else seed)
        self._core_gain = np.ones(cfg.num_cores, dtype=float)
        self.reset(seed=seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.true_temperatures = np.full(self.cfg.num_cores, self.cfg.initial_temp, dtype=float)
        self.power = np.zeros(self.cfg.num_cores, dtype=float)
        self.load = np.zeros(self.cfg.num_cores, dtype=float)
        self.task_counts = np.zeros(self.cfg.num_cores, dtype=int)
        self.active_tasks: list[list[Task]] = [[] for _ in range(self.cfg.num_cores)]
        self.time_index = 0
        self.completed_tasks = 0
        self.assigned_tasks = 0
        self.dropped_tasks = 0
        self.assignment_counts = np.zeros(self.cfg.num_cores, dtype=int)
        self._last_sensor_readings: np.ndarray | None = None
        self._last_sensor_mask: np.ndarray | None = None
        self._core_gain = self.rng.normal(1.0, 0.025, size=self.cfg.num_cores)

    def clone(self) -> "ThermalSimulator":
        return copy.deepcopy(self)

    def assign_task(self, task: Task, core: int) -> bool:
        if core < 0 or core >= self.cfg.num_cores:
            raise ValueError(f"Invalid core {core}")
        active_total = int(np.sum(self.task_counts))
        if active_total >= self.cfg.task_queue_limit:
            self.dropped_tasks += 1
            return False
        assigned = task.copy_for_core(core)
        self.active_tasks[core].append(assigned)
        self.assigned_tasks += 1
        self.assignment_counts[core] += 1
        self._refresh_power_load()
        return True

    def step(self) -> None:
        old = self.true_temperatures
        laplacian = np.zeros_like(old)
        for core in range(self.cfg.num_cores):
            neighbors = neighbor_indices(core, self.cfg.grid_size)
            if neighbors:
                laplacian[core] = np.mean(old[neighbors]) - old[core]

        proposed = old.copy()
        proposed += self.cfg.heat_gain * self.power * self._core_gain
        proposed += self.cfg.diffusion_coeff * laplacian
        proposed -= self.cfg.cooling_coeff * (old - self.cfg.ambient_temp)
        if self.cfg.thermal_noise_sigma > 0:
            proposed += self.rng.normal(0.0, self.cfg.thermal_noise_sigma, size=self.cfg.num_cores)

        inertia = self.cfg.thermal_inertia
        new_temps = inertia * old + (1.0 - inertia) * proposed
        self.true_temperatures = np.clip(
            new_temps,
            self.cfg.ambient_temp - 5.0,
            self.cfg.max_reasonable_temp,
        )

        for core in range(self.cfg.num_cores):
            still_active: list[Task] = []
            for task in self.active_tasks[core]:
                task.remaining_time -= 1
                if task.remaining_time <= 0:
                    self.completed_tasks += 1
                else:
                    still_active.append(task)
            self.active_tasks[core] = still_active
        self.time_index += 1
        self._refresh_power_load()

    def _refresh_power_load(self) -> None:
        for core in range(self.cfg.num_cores):
            tasks = self.active_tasks[core]
            self.power[core] = float(sum(task.power for task in tasks))
            self.load[core] = float(sum(task.remaining_time for task in tasks))
            self.task_counts[core] = len(tasks)

    def observe_state(self, sensor_model: SensorModel) -> ChipState:
        readings, mask = sensor_model.observe(self.true_temperatures, self.rng)
        trend = np.zeros(self.cfg.num_cores, dtype=float)
        if self._last_sensor_readings is not None and self._last_sensor_mask is not None:
            valid = mask & self._last_sensor_mask
            trend[valid] = readings[valid] - self._last_sensor_readings[valid]
        self._last_sensor_readings = readings.copy()
        self._last_sensor_mask = mask.copy()
        return self.build_state(readings, mask, trend)

    def build_state(
        self,
        sensor_readings: np.ndarray,
        sensor_mask: np.ndarray,
        observed_temp_trend: np.ndarray | None = None,
    ) -> ChipState:
        trend = (
            np.zeros(self.cfg.num_cores, dtype=float)
            if observed_temp_trend is None
            else np.asarray(observed_temp_trend, dtype=float).copy()
        )
        return ChipState(
            true_temperatures=self.true_temperatures.copy(),
            power=self.power.copy(),
            load=self.load.copy(),
            task_counts=self.task_counts.astype(float).copy(),
            sensor_readings=np.asarray(sensor_readings, dtype=float).copy(),
            sensor_mask=np.asarray(sensor_mask, dtype=bool).copy(),
            time_index=int(self.time_index),
            observed_temp_trend=trend,
        )

    def future_peak_after_assignment(self, task: Task, candidate_core: int, horizon: int) -> float:
        sim = self.clone()
        sim.assign_task(task, candidate_core)
        peak = float(np.max(sim.true_temperatures))
        for _ in range(horizon):
            sim.step()
            peak = max(peak, float(np.max(sim.true_temperatures)))
        return peak
