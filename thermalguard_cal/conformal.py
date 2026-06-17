from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


@dataclass
class OneSidedConformalCalibrator:
    target_coverage: float = 0.90
    correction: float = 0.0
    q_level: float = 1.0
    n_calibration: int = 0

    def fit(self, y_true: np.ndarray, q_pred: np.ndarray) -> "OneSidedConformalCalibrator":
        y_true = np.asarray(y_true, dtype=float)
        q_pred = np.asarray(q_pred, dtype=float)
        if y_true.shape != q_pred.shape:
            raise ValueError("y_true and q_pred must have the same shape")
        n = int(y_true.size)
        if n <= 0:
            raise ValueError("Conformal calibration requires at least one example")
        residuals = y_true - q_pred
        # CONFORMAL CORRECTION FORMULA (use exactly this, do not approximate):
        # Let n = number of calibration examples.
        # Let target_coverage = desired coverage (default 0.90, optionally 0.95).
        # Let residuals[i] = y_true[i] - q_pred[i] for each calibration example i.
        # Compute the adjusted quantile level:
        # q_level = min(1.0, ceil((n + 1) * target_coverage) / n)
        # Then:
        # correction = quantile(residuals, q_level), using the standard (linear-interpolation) empirical quantile.
        # correction = max(0, correction)
        # calibrated_upper = q_pred + correction
        self.q_level = min(1.0, ceil((n + 1) * self.target_coverage) / n)
        try:
            correction = np.quantile(residuals, self.q_level, method="linear")
        except TypeError:
            correction = np.quantile(residuals, self.q_level, interpolation="linear")
        self.correction = float(max(0.0, correction))
        self.n_calibration = n
        return self

    def predict_upper(self, q_pred: np.ndarray) -> np.ndarray:
        return np.asarray(q_pred, dtype=float) + self.correction

    def coverage(self, y_true: np.ndarray, q_pred: np.ndarray) -> float:
        upper = self.predict_upper(q_pred)
        return empirical_coverage(y_true, upper)


def empirical_coverage(y_true: np.ndarray, upper: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    upper = np.asarray(upper, dtype=float)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(y_true <= upper))
