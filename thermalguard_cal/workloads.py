from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ThermalGuardConfig
from .utils import Task


@dataclass(frozen=True)
class WorkloadProfile:
    name: str
    arrival_rate: float
    mix: tuple[float, float, float]
    burst_probability: float = 0.0
    burst_extra_rate: float = 0.0
    power_scale: float = 1.0
    duration_scale: float = 1.0


class WorkloadGenerator:
    """Stable ID and separate OOD workload generator."""

    def __init__(self, cfg: ThermalGuardConfig, profile: WorkloadProfile) -> None:
        self.cfg = cfg
        self.profile = profile

    @classmethod
    def for_split(cls, cfg: ThermalGuardConfig, split: str) -> "WorkloadGenerator":
        if split == "ood":
            profile = WorkloadProfile(
                name="ood",
                arrival_rate=cfg.ood_arrival_rate,
                mix=cfg.ood_task_mix,
                burst_probability=cfg.ood_burst_probability,
                burst_extra_rate=cfg.ood_burst_extra_rate,
                power_scale=cfg.ood_power_scale,
                duration_scale=cfg.ood_duration_scale,
            )
        else:
            profile = WorkloadProfile(
                name="id",
                arrival_rate=cfg.id_arrival_rate,
                mix=cfg.id_task_mix,
                burst_probability=cfg.id_burst_probability,
                burst_extra_rate=cfg.id_burst_extra_rate,
                power_scale=cfg.id_power_scale,
                duration_scale=cfg.id_duration_scale,
            )
        return cls(cfg, profile)

    def generate_episode(self, seed: int, episode_id: int) -> dict[int, list[Task]]:
        rng = np.random.default_rng(seed)
        arrivals: dict[int, list[Task]] = {}
        task_id = episode_id * 1_000_000

        for t in range(self.cfg.episode_length):
            rate = self.profile.arrival_rate
            if self.profile.burst_probability and rng.random() < self.profile.burst_probability:
                rate += self.profile.burst_extra_rate
            count = min(
                int(rng.poisson(rate)),
                self.cfg.max_arrivals_per_step,
            )
            if count <= 0:
                continue
            arrivals[t] = []
            for _ in range(count):
                power, duration = self._sample_task_params(rng)
                arrivals[t].append(
                    Task(
                        task_id=task_id,
                        arrival_time=t,
                        power=power,
                        duration=duration,
                        remaining_time=duration,
                    )
                )
                task_id += 1
        return arrivals

    def _sample_task_params(self, rng: np.random.Generator) -> tuple[float, int]:
        tier = rng.choice(3, p=np.array(self.profile.mix) / np.sum(self.profile.mix))
        if tier == 0:
            power = rng.uniform(0.5, 1.0)
            duration = int(rng.integers(10, 31))
        elif tier == 1:
            power = rng.uniform(1.0, 2.0)
            duration = int(rng.integers(20, 61))
        else:
            power = rng.uniform(2.0, 3.5)
            duration = int(rng.integers(30, 101))

        if self.profile.name == "ood" and tier == 2:
            power *= rng.uniform(1.05, 1.25)
            duration = int(duration * rng.uniform(1.05, 1.25))
        power *= self.profile.power_scale
        duration = int(round(duration * self.profile.duration_scale))
        return float(power), max(1, int(duration))
