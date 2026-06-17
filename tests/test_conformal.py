import numpy as np

from thermalguard_cal.conformal import OneSidedConformalCalibrator, empirical_coverage


def test_conformal_correction_nonnegative_and_upper_monotone():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    q = np.array([1.5, 1.5, 2.5, 5.0])
    cal = OneSidedConformalCalibrator(target_coverage=0.9).fit(y, q)
    upper = cal.predict_upper(q)
    assert cal.correction >= 0
    assert np.all(upper >= q)


def test_q_level_formula_exact():
    y = np.arange(10.0)
    q = y - 1.0
    cal = OneSidedConformalCalibrator(target_coverage=0.9).fit(y, q)
    assert cal.q_level == min(1.0, np.ceil((10 + 1) * 0.9) / 10)


def test_empirical_coverage_toy():
    y = np.array([1, 2, 3, 4])
    upper = np.array([1, 3, 2, 5])
    assert empirical_coverage(y, upper) == 0.75
