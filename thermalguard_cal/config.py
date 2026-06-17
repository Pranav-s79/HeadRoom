from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ThermalGuardConfig:
    random_seed: int = 17
    grid_size: int = 4
    ambient_temp: float = 35.0
    initial_temp: float = 35.0
    thermal_limit: float = 85.0
    episode_length: int = 150
    prediction_horizon: int = 10

    heat_gain: float = 0.48
    cooling_coeff: float = 0.035
    diffusion_coeff: float = 0.045
    thermal_inertia: float = 0.88
    thermal_noise_sigma: float = 0.015
    max_reasonable_temp: float = 130.0

    train_episodes: int = 20
    calibration_episodes: int = 5
    test_episodes: int = 5
    ood_episodes: int = 5

    sensor_indices: tuple[int, ...] = (0, 3, 12, 15)
    sensor_noise_sigma: float = 0.5
    sensor_dropout_prob: float = 0.04
    ood_sensor_dropout_prob: float = 0.12

    id_arrival_rate: float = 0.38
    ood_arrival_rate: float = 0.56
    max_arrivals_per_step: int = 3
    task_queue_limit: int = 256

    quantile_alpha: float = 0.90
    conformal_target_coverage: float = 0.90
    conformal_lb_threshold_c: float = 1.5
    trend_k: float = 2.0

    max_train_rows: int = 60_000
    random_forest_estimators: int = 64
    gradient_boosting_estimators: int = 140

    output_dir: str = "outputs"
    quick: bool = True

    @property
    def num_cores(self) -> int:
        return self.grid_size * self.grid_size

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["num_cores"] = self.num_cores
        return data


def make_config(mode: str = "quick", **overrides: Any) -> ThermalGuardConfig:
    if mode not in {"quick", "full"}:
        raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")

    if mode == "full":
        cfg = ThermalGuardConfig(
            quick=False,
            episode_length=500,
            train_episodes=200,
            calibration_episodes=50,
            test_episodes=50,
            ood_episodes=50,
            max_train_rows=120_000,
            random_forest_estimators=96,
            gradient_boosting_estimators=180,
        )
    else:
        cfg = ThermalGuardConfig()

    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(f"Unknown config field {key!r}")
        setattr(cfg, key, value)

    if cfg.num_cores != 16:
        raise ValueError("This MVP is locked to a 4x4 / 16-core chip.")
    return cfg


def ensure_output_dirs(cfg: ThermalGuardConfig) -> None:
    for child in ("data", "models", "figures", "reports"):
        (cfg.output_path / child).mkdir(parents=True, exist_ok=True)
