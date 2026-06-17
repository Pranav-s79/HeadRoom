import numpy as np
from pathlib import Path

from thermalguard_cal.config import make_config
import thermalguard_cal.features as features_module
from thermalguard_cal.features import build_candidate_features
from thermalguard_cal.utils import ChipState, Task


def test_feature_vector_ignores_true_temperatures():
    cfg = make_config("quick")
    base = dict(
        power=np.ones(cfg.num_cores),
        load=np.arange(cfg.num_cores, dtype=float),
        task_counts=np.zeros(cfg.num_cores),
        sensor_readings=np.full(cfg.num_cores, np.nan),
        sensor_mask=np.zeros(cfg.num_cores, dtype=bool),
        time_index=3,
        observed_temp_trend=np.zeros(cfg.num_cores),
    )
    task = Task(1, 0, 1.5, 20, 20)
    state_a = ChipState(true_temperatures=np.zeros(cfg.num_cores), **base)
    state_b = ChipState(true_temperatures=np.full(cfg.num_cores, 9999.0), **base)
    features_a = build_candidate_features(state_a, task, 4, cfg)
    features_b = build_candidate_features(state_b, task, 4, cfg)
    np.testing.assert_allclose(features_a, features_b)


def test_features_module_does_not_reference_true_temperatures_field():
    source = Path(features_module.__file__).read_text(encoding="utf-8")
    assert "true_temperatures" not in source
