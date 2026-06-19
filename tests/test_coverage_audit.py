import numpy as np
from numpy.typing import NDArray
from typing import Any

from thermalguard_cal.config import make_config
from thermalguard_cal.evaluate import evaluate_scheduler
from thermalguard_cal.features import feature_names


class FixedConformalScheduler:
    name = "conformal_upper_bound"

    def reset(self) -> None:
        pass

    def choose_core(self, state: Any, task: Any) -> int:
        return 0


class SplitAwareConformalModel:
    def __init__(self, names: list[str]) -> None:
        self.feature_names = names

    def predict_conformal_upper(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if X.shape[0] == 1:
            return np.zeros(1, dtype=float)
        return np.full(X.shape[0], 1_000.0, dtype=float)


def test_selected_core_coverage_uses_selected_decision_not_candidate_average() -> None:
    cfg = make_config(
        "quick",
        episode_length=8,
        prediction_horizon=2,
        test_episodes=1,
        id_arrival_rate=1.0,
        id_task_mix=(0.0, 0.0, 1.0),
        sensor_dropout_prob=0.0,
    )
    names = feature_names(cfg)
    cal_stats = {
        "mean": np.zeros(len(names), dtype=float),
        "std": np.ones(len(names), dtype=float),
        "sample": np.zeros((4, len(names)), dtype=float),
    }

    result = evaluate_scheduler(
        cfg=cfg,
        scheduler=FixedConformalScheduler(),
        bundle=SplitAwareConformalModel(names),
        split="id",
        cal_stats=cal_stats,
    )
    coverage = {row["coverage_type"]: row for row in result["coverage_rows"]}

    assert coverage["marginal_all_candidates_on_visited_states"]["empirical_coverage"] == 1.0
    assert coverage["selected_core_after_scheduler_selection"]["empirical_coverage"] == 0.0
    assert result["summary"].selected_core_coverage == 0.0
    assert result["summary"].selected_bound_violations > 0
