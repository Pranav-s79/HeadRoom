from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ThermalGuardConfig
from .features import build_candidate_matrix
from .sensors import impute_sensor_readings
from .utils import ChipState, Task


class Scheduler:
    name = "base"

    def choose_core(self, state: ChipState, task: Task) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        pass


class RandomScheduler(Scheduler):
    name = "random"

    def __init__(self, cfg: ThermalGuardConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.random_seed if seed is None else seed)

    def choose_core(self, state: ChipState, task: Task) -> int:
        return int(self.rng.integers(0, self.cfg.num_cores))


class RoundRobinScheduler(Scheduler):
    name = "round_robin"

    def __init__(self, cfg: ThermalGuardConfig) -> None:
        self.cfg = cfg
        self.next_core = 0

    def reset(self) -> None:
        self.next_core = 0

    def choose_core(self, state: ChipState, task: Task) -> int:
        core = self.next_core
        self.next_core = (self.next_core + 1) % self.cfg.num_cores
        return int(core)


class CoolestCoreOracleScheduler(Scheduler):
    name = "coolest_core_oracle_true_temp"

    def __init__(self, cfg: ThermalGuardConfig) -> None:
        self.cfg = cfg

    def choose_core(self, state: ChipState, task: Task) -> int:
        return int(np.argmin(state.true_temperatures))


class CoolestCoreObservedScheduler(Scheduler):
    name = "coolest_core_observed"

    def __init__(self, cfg: ThermalGuardConfig) -> None:
        self.cfg = cfg

    def choose_core(self, state: ChipState, task: Task) -> int:
        estimated = impute_sensor_readings(state.sensor_readings, state.sensor_mask, self.cfg.ambient_temp)
        return int(np.lexsort((np.arange(self.cfg.num_cores), estimated))[0])


class TrendAwareScheduler(Scheduler):
    name = "trend_aware_observed"

    def __init__(self, cfg: ThermalGuardConfig, k: float | None = None) -> None:
        self.cfg = cfg
        self.k = cfg.trend_k if k is None else float(k)

    def choose_core(self, state: ChipState, task: Task) -> int:
        estimated = impute_sensor_readings(state.sensor_readings, state.sensor_mask, self.cfg.ambient_temp)
        trend = np.where(np.isfinite(state.observed_temp_trend), state.observed_temp_trend, 0.0)
        score = estimated + self.k * trend
        return int(np.lexsort((np.arange(self.cfg.num_cores), score))[0])


@dataclass
class ModelPredictionScheduler(Scheduler):
    cfg: ThermalGuardConfig
    model: object
    name: str = "model_prediction"
    method: str = "predict"

    def choose_core(self, state: ChipState, task: Task) -> int:
        scores = self.score_candidates(state, task)
        return int(np.argmin(scores))

    def score_candidates(self, state: ChipState, task: Task) -> np.ndarray:
        X = build_candidate_matrix(state, task, self.cfg)
        predict_fn = getattr(self.model, self.method)
        return np.asarray(predict_fn(X), dtype=float)


class PointPredictionScheduler(ModelPredictionScheduler):
    def __init__(self, cfg: ThermalGuardConfig, model: object) -> None:
        super().__init__(cfg=cfg, model=model, name="point_prediction_rf", method="predict_point")


class UncalibratedQuantileScheduler(ModelPredictionScheduler):
    def __init__(self, cfg: ThermalGuardConfig, model: object) -> None:
        super().__init__(cfg=cfg, model=model, name="uncalibrated_quantile", method="predict_quantile")


class ConformalUpperBoundScheduler(ModelPredictionScheduler):
    def __init__(self, cfg: ThermalGuardConfig, model: object, threshold_c: float | None = None) -> None:
        super().__init__(cfg=cfg, model=model, name="conformal_upper_bound", method="predict_conformal_upper")
        self.threshold_c = cfg.conformal_lb_threshold_c if threshold_c is None else float(threshold_c)

    def choose_core(self, state: ChipState, task: Task) -> int:
        scores = self.score_candidates(state, task)
        best = float(np.min(scores))
        candidates = np.flatnonzero(scores <= best + self.threshold_c)
        estimated = impute_sensor_readings(state.sensor_readings, state.sensor_mask, self.cfg.ambient_temp)
        order = sorted(
            candidates.tolist(),
            key=lambda core: (float(state.load[core]), float(estimated[core]), int(core)),
        )
        return int(order[0])
