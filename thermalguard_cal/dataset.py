from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ThermalGuardConfig, ensure_output_dirs
from .features import build_candidate_features, feature_names
from .sensors import SensorModel
from .simulator import ThermalSimulator
from .utils import Task
from .workloads import WorkloadGenerator


@dataclass(frozen=True)
class SplitSpec:
    name: str
    start_episode: int
    count: int
    workload_split: str

    @property
    def episode_ids(self) -> range:
        return range(self.start_episode, self.start_episode + self.count)


def split_specs(cfg: ThermalGuardConfig) -> list[SplitSpec]:
    train = SplitSpec("train", 0, cfg.train_episodes, "id")
    cal = SplitSpec("calibration", cfg.train_episodes, cfg.calibration_episodes, "id")
    test = SplitSpec("test_id", cfg.train_episodes + cfg.calibration_episodes, cfg.test_episodes, "id")
    ood = SplitSpec("test_ood", 10_000, cfg.ood_episodes, "ood")
    return [train, cal, test, ood]


def generate_datasets(cfg: ThermalGuardConfig) -> dict[str, dict[str, int]]:
    ensure_output_dirs(cfg)
    data_dir = cfg.output_path / "data"
    with (data_dir / "feature_names.json").open("w", encoding="utf-8") as f:
        json.dump(feature_names(cfg), f, indent=2)

    summaries: dict[str, dict[str, int]] = {}
    for spec in split_specs(cfg):
        X, y, meta = generate_split(cfg, spec)
        np.save(data_dir / f"X_{spec.name}.npy", X)
        np.save(data_dir / f"y_{spec.name}.npy", y)
        meta.to_csv(data_dir / f"meta_{spec.name}.csv", index=False)
        summaries[spec.name] = {
            "rows": int(X.shape[0]),
            "features": int(X.shape[1]) if X.ndim == 2 else 0,
            "episodes": int(spec.count),
        }
    with (data_dir / "dataset_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2, sort_keys=True)
    return summaries


def generate_split(
    cfg: ThermalGuardConfig,
    spec: SplitSpec,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    X_rows: list[np.ndarray] = []
    y_rows: list[float] = []
    meta_rows: list[dict[str, object]] = []
    generator = WorkloadGenerator.for_split(cfg, spec.workload_split)
    dropout = cfg.ood_sensor_dropout_prob if spec.workload_split == "ood" else cfg.sensor_dropout_prob
    sensor_model = SensorModel(cfg, dropout_prob=dropout)

    for episode_id in spec.episode_ids:
        seed = cfg.random_seed + episode_id * 9973
        sim = ThermalSimulator(cfg, seed=seed)
        arrivals = generator.generate_episode(seed=seed + 101, episode_id=episode_id)
        rng = np.random.default_rng(seed + 202)
        stratified_cursor = int(rng.integers(0, cfg.num_cores))

        for timestep in range(cfg.episode_length):
            observed_state = sim.observe_state(sensor_model)
            readings = observed_state.sensor_readings.copy()
            mask = observed_state.sensor_mask.copy()
            trend = observed_state.observed_temp_trend.copy()

            for task in arrivals.get(timestep, []):
                state = sim.build_state(readings, mask, trend)
                for candidate_core in range(cfg.num_cores):
                    X_rows.append(build_candidate_features(state, task, candidate_core, cfg))
                    y_rows.append(
                        sim.future_peak_after_assignment(
                            task,
                            candidate_core,
                            cfg.prediction_horizon,
                        )
                    )
                    meta_rows.append(
                        {
                            "episode_id": episode_id,
                            "timestep": timestep,
                            "candidate_core": candidate_core,
                            "task_id": task.task_id,
                            "task_power": task.power,
                            "task_duration": task.duration,
                            "split": spec.name,
                            "workload_type": spec.workload_split,
                        }
                    )

                core = stratified_cursor % cfg.num_cores
                stratified_cursor += 1
                sim.assign_task(task, core)

            sim.step()

    if X_rows:
        X = np.vstack(X_rows).astype(float)
        y = np.array(y_rows, dtype=float)
    else:
        X = np.empty((0, len(feature_names(cfg))), dtype=float)
        y = np.empty((0,), dtype=float)
    meta = pd.DataFrame(meta_rows)
    return X, y, meta


def load_split(output_dir: str | Path, split: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    data_dir = Path(output_dir) / "data"
    X = np.load(data_dir / f"X_{split}.npy")
    y = np.load(data_dir / f"y_{split}.npy")
    meta = pd.read_csv(data_dir / f"meta_{split}.csv")
    return X, y, meta
