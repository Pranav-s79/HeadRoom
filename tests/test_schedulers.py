import numpy as np

from thermalguard_cal.config import make_config
from thermalguard_cal.schedulers import ConformalUpperBoundScheduler, RandomScheduler, RoundRobinScheduler
from thermalguard_cal.utils import ChipState, Task


class DummyConformalModel:
    def predict_conformal_upper(self, X):
        scores = np.full(X.shape[0], 100.0, dtype=float)
        scores[:3] = [10.0, 10.5, 10.8]
        return scores


def make_state(cfg):
    return ChipState(
        true_temperatures=np.full(cfg.num_cores, cfg.ambient_temp),
        power=np.zeros(cfg.num_cores),
        load=np.arange(cfg.num_cores, dtype=float)[::-1],
        task_counts=np.zeros(cfg.num_cores),
        sensor_readings=np.full(cfg.num_cores, np.nan),
        sensor_mask=np.zeros(cfg.num_cores, dtype=bool),
        time_index=0,
        observed_temp_trend=np.zeros(cfg.num_cores),
    )


def test_schedulers_return_valid_core():
    cfg = make_config("quick")
    state = make_state(cfg)
    task = Task(1, 0, 1.0, 10, 10)
    assert 0 <= RandomScheduler(cfg, seed=1).choose_core(state, task) < cfg.num_cores


def test_round_robin_cycles():
    cfg = make_config("quick")
    rr = RoundRobinScheduler(cfg)
    state = make_state(cfg)
    task = Task(1, 0, 1.0, 10, 10)
    assert [rr.choose_core(state, task) for _ in range(18)] == list(range(16)) + [0, 1]


def test_conformal_scheduler_load_balance_fallback():
    cfg = make_config("quick")
    state = make_state(cfg)
    state.load[0] = 10.0
    state.load[1] = 0.0
    task = Task(1, 0, 1.0, 10, 10)
    sched = ConformalUpperBoundScheduler(cfg, DummyConformalModel(), threshold_c=1.0)
    assert sched.choose_core(state, task) == 1
