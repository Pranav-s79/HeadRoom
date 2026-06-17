from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Task:
    task_id: int
    arrival_time: int
    power: float
    duration: int
    remaining_time: int
    assigned_core: int | None = None

    def copy_for_core(self, core: int) -> "Task":
        return Task(
            task_id=self.task_id,
            arrival_time=self.arrival_time,
            power=float(self.power),
            duration=int(self.duration),
            remaining_time=int(self.remaining_time),
            assigned_core=int(core),
        )


@dataclass
class ChipState:
    # INTERNAL SIMULATOR USE ONLY. Feature code must not read this field.
    true_temperatures: np.ndarray
    power: np.ndarray
    load: np.ndarray
    task_counts: np.ndarray
    sensor_readings: np.ndarray
    sensor_mask: np.ndarray
    time_index: int
    observed_temp_trend: np.ndarray


@dataclass
class RunSummary:
    scheduler: str
    split: str
    baseline_type: str
    uses_oracle: bool
    peak_temperature: float
    average_temperature: float
    hotspot_violations: int
    hotspot_timestep_pct: float
    completed_tasks: int
    assigned_tasks: int
    dropped_tasks: int
    average_waiting_time: float
    assignment_load_std: float
    average_max_temperature: float
    marginal_coverage: float | None = None
    selected_core_coverage: float | None = None
    selected_coverage_gap: float | None = None
    selected_bound_violations: int | None = None
    average_selected_bound: float | None = None
    average_selected_actual: float | None = None
    drift_mean_abs_z: float | None = None
    drift_max_ks: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def core_to_rc(core: int, grid_size: int) -> tuple[int, int]:
    return divmod(int(core), int(grid_size))


def neighbor_indices(core: int, grid_size: int) -> list[int]:
    row, col = core_to_rc(core, grid_size)
    neighbors: list[int] = []
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr, nc = row + dr, col + dc
        if 0 <= nr < grid_size and 0 <= nc < grid_size:
            neighbors.append(nr * grid_size + nc)
    return neighbors


def stable_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(int(seed))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def current_git_commit(cwd: str | Path = ".") -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
