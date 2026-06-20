from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class ThermalGuardConfig:
    random_seed: int = 17
    preset: str = "normal"
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
    id_task_mix: tuple[float, float, float] = (0.45, 0.40, 0.15)
    ood_task_mix: tuple[float, float, float] = (0.22, 0.38, 0.40)
    id_burst_probability: float = 0.0
    ood_burst_probability: float = 0.12
    id_burst_extra_rate: float = 0.0
    ood_burst_extra_rate: float = 1.4
    id_power_scale: float = 1.0
    ood_power_scale: float = 1.0
    id_duration_scale: float = 1.0
    ood_duration_scale: float = 1.0
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


PRESET_OVERRIDES: dict[str, dict[str, Any]] = {
    "easy": {
        "preset": "easy",
        "id_arrival_rate": 0.34,
        "ood_arrival_rate": 0.50,
        "sensor_dropout_prob": 0.03,
        "ood_sensor_dropout_prob": 0.10,
    },
    "normal": {
        "preset": "normal",
    },
    "challenging": {
        "preset": "challenging",
        "heat_gain": 0.49,
        "cooling_coeff": 0.034,
        "diffusion_coeff": 0.045,
        "sensor_dropout_prob": 0.06,
        "ood_sensor_dropout_prob": 0.12,
        "id_arrival_rate": 0.50,
        "ood_arrival_rate": 0.61,
        "id_task_mix": (0.32, 0.42, 0.26),
        "ood_task_mix": (0.20, 0.34, 0.46),
        "id_burst_probability": 0.11,
        "ood_burst_probability": 0.15,
        "id_burst_extra_rate": 1.0,
        "ood_burst_extra_rate": 1.5,
        "id_power_scale": 1.03,
        "ood_power_scale": 1.05,
        "id_duration_scale": 1.08,
        "ood_duration_scale": 1.12,
    },
    "stress": {
        "preset": "stress",
        "heat_gain": 0.66,
        "cooling_coeff": 0.026,
        "diffusion_coeff": 0.036,
        "sensor_dropout_prob": 0.12,
        "ood_sensor_dropout_prob": 0.22,
        "id_arrival_rate": 0.62,
        "ood_arrival_rate": 0.76,
        "id_task_mix": (0.22, 0.38, 0.40),
        "ood_task_mix": (0.12, 0.30, 0.58),
        "id_burst_probability": 0.12,
        "ood_burst_probability": 0.22,
        "id_burst_extra_rate": 1.2,
        "ood_burst_extra_rate": 2.0,
        "id_power_scale": 1.10,
        "ood_power_scale": 1.15,
        "id_duration_scale": 1.20,
        "ood_duration_scale": 1.25,
    },
}


def make_config(mode: str = "quick", preset: str = "normal", **overrides: Any) -> ThermalGuardConfig:
    if mode not in {"quick", "full"}:
        raise ValueError(f"mode must be 'quick' or 'full', got {mode!r}")
    if preset not in PRESET_OVERRIDES:
        valid = ", ".join(sorted(PRESET_OVERRIDES))
        raise ValueError(f"preset must be one of {valid}, got {preset!r}")

    if mode == "full":
        cfg = ThermalGuardConfig(
            quick=False,
            episode_length=500,
            train_episodes=200,
            calibration_episodes=50,
            test_episodes=50,
            ood_episodes=50,
            max_train_rows=120_000,
            # Full-mode hyperparameters reviewed in Work Stream 2c: the random
            # forest gains from more trees (200-500 range), and the quantile GBR
            # benefits from more low-learning-rate estimators. Depth/learning-rate
            # are set in models.py and were left at their stable defaults.
            random_forest_estimators=300,
            gradient_boosting_estimators=240,
        )
    else:
        cfg = ThermalGuardConfig()

    for key, value in PRESET_OVERRIDES[preset].items():
        setattr(cfg, key, value)

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
