from thermalguard_cal.config import make_config
from thermalguard_cal.workloads import WorkloadGenerator


def test_id_and_ood_workloads_are_separate_profiles():
    cfg = make_config("quick")
    id_gen = WorkloadGenerator.for_split(cfg, "id")
    ood_gen = WorkloadGenerator.for_split(cfg, "ood")
    assert id_gen.profile.name == "id"
    assert ood_gen.profile.name == "ood"
    assert ood_gen.profile.mix != id_gen.profile.mix


def test_generated_tasks_have_valid_ranges():
    cfg = make_config("quick")
    gen = WorkloadGenerator.for_split(cfg, "id")
    arrivals = gen.generate_episode(seed=123, episode_id=0)
    tasks = [task for batch in arrivals.values() for task in batch]
    assert tasks
    assert all(task.power > 0 for task in tasks)
    assert all(task.duration == task.remaining_time for task in tasks)
