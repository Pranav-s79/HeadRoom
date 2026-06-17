import numpy as np

from thermalguard_cal.config import make_config
from thermalguard_cal.simulator import ThermalSimulator
from thermalguard_cal.utils import Task


def test_zero_power_cools_toward_ambient():
    cfg = make_config("quick")
    sim = ThermalSimulator(cfg, seed=1)
    sim.true_temperatures[:] = 70.0
    before = float(np.mean(sim.true_temperatures))
    for _ in range(20):
        sim.step()
    after = float(np.mean(sim.true_temperatures))
    assert after < before
    assert after > cfg.ambient_temp


def test_sustained_power_heats_core():
    cfg = make_config("quick")
    sim = ThermalSimulator(cfg, seed=2)
    sim.assign_task(Task(1, 0, power=3.0, duration=100, remaining_time=100), 5)
    before = sim.true_temperatures[5]
    for _ in range(30):
        sim.step()
    assert sim.true_temperatures[5] > before


def test_diffusion_warms_neighbor():
    cfg = make_config("quick")
    sim = ThermalSimulator(cfg, seed=3)
    sim.true_temperatures[:] = cfg.ambient_temp
    sim.true_temperatures[5] = 90.0
    before = sim.true_temperatures[6]
    for _ in range(5):
        sim.step()
    assert sim.true_temperatures[6] > before
    assert sim.true_temperatures[0] < sim.true_temperatures[6]
