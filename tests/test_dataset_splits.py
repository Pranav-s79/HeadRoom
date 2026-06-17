from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import split_specs


def test_episode_splits_do_not_overlap():
    cfg = make_config("quick")
    specs = split_specs(cfg)
    episode_sets = {spec.name: set(spec.episode_ids) for spec in specs}
    for left_name, left in episode_sets.items():
        for right_name, right in episode_sets.items():
            if left_name >= right_name:
                continue
            assert left.isdisjoint(right)


def test_ood_split_marked_separately():
    cfg = make_config("quick")
    specs = {spec.name: spec for spec in split_specs(cfg)}
    assert specs["train"].workload_split == "id"
    assert specs["calibration"].workload_split == "id"
    assert specs["test_id"].workload_split == "id"
    assert specs["test_ood"].workload_split == "ood"
