"""Models for ThermalGuard-Cal.

Prediction target: FUTURE PEAK CHIP TEMPERATURE over the horizon, a global
whole-chip quantity. The target is not the candidate core's own future
temperature. Selected-core coverage therefore means coverage of the global
outcome that results from a placement choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # pragma: no cover - older sklearn compatibility
    root_mean_squared_error = None
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .conformal import OneSidedConformalCalibrator, empirical_coverage
from .config import ThermalGuardConfig, ensure_output_dirs
from .dataset import load_split


@dataclass
class ModelBundle:
    linear_point: Any
    forest_point: Any
    quantile_upper: Any
    conformal: OneSidedConformalCalibrator
    feature_names: list[str]
    config: dict[str, Any]

    def predict_point(self, X: np.ndarray) -> np.ndarray:
        return self.forest_point.predict(X)

    def predict_quantile(self, X: np.ndarray) -> np.ndarray:
        return self.quantile_upper.predict(X)

    def predict_conformal_upper(self, X: np.ndarray) -> np.ndarray:
        return self.conformal.predict_upper(self.predict_quantile(X))


def train_and_save_models(cfg: ThermalGuardConfig) -> pd.DataFrame:
    ensure_output_dirs(cfg)
    X_train, y_train, _ = load_split(cfg.output_path, "train")
    X_cal, y_cal, _ = load_split(cfg.output_path, "calibration")
    X_test, y_test, _ = load_split(cfg.output_path, "test_id")
    X_ood, y_ood, _ = load_split(cfg.output_path, "test_ood")

    rng = np.random.default_rng(cfg.random_seed)
    if X_train.shape[0] > cfg.max_train_rows:
        idx = rng.choice(X_train.shape[0], size=cfg.max_train_rows, replace=False)
        X_fit = X_train[idx]
        y_fit = y_train[idx]
    else:
        X_fit = X_train
        y_fit = y_train

    linear = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    forest = RandomForestRegressor(
        n_estimators=cfg.random_forest_estimators,
        max_depth=14,
        min_samples_leaf=4,
        n_jobs=-1,
        random_state=cfg.random_seed,
    )
    quantile = GradientBoostingRegressor(
        loss="quantile",
        alpha=cfg.quantile_alpha,
        n_estimators=cfg.gradient_boosting_estimators,
        max_depth=3,
        learning_rate=0.055,
        min_samples_leaf=10,
        random_state=cfg.random_seed,
    )

    linear.fit(X_fit, y_fit)
    forest.fit(X_fit, y_fit)
    quantile.fit(X_fit, y_fit)

    q_cal = quantile.predict(X_cal)
    conformal = OneSidedConformalCalibrator(cfg.conformal_target_coverage).fit(y_cal, q_cal)

    feature_names_path = cfg.output_path / "data" / "feature_names.json"
    if feature_names_path.exists():
        import json

        with feature_names_path.open("r", encoding="utf-8") as f:
            names = json.load(f)
    else:
        names = [f"f{i}" for i in range(X_train.shape[1])]

    bundle = ModelBundle(
        linear_point=linear,
        forest_point=forest,
        quantile_upper=quantile,
        conformal=conformal,
        feature_names=names,
        config=cfg.to_dict(),
    )
    models_dir = cfg.output_path / "models"
    joblib.dump(bundle, models_dir / "model_bundle.joblib")
    joblib.dump(linear, models_dir / "linear_point.joblib")
    joblib.dump(forest, models_dir / "forest_point.joblib")
    joblib.dump(quantile, models_dir / "quantile_upper.joblib")
    joblib.dump(conformal, models_dir / "conformal_calibrator.joblib")

    rows = []
    for split, X, y in (
        ("calibration", X_cal, y_cal),
        ("test_id", X_test, y_test),
        ("test_ood", X_ood, y_ood),
    ):
        rows.extend(_metric_rows(split, "linear_point", y, linear.predict(X), kind="point"))
        rows.extend(_metric_rows(split, "forest_point", y, forest.predict(X), kind="point"))
        q_pred = quantile.predict(X)
        rows.extend(_metric_rows(split, "quantile_upper", y, q_pred, kind="upper"))
        conformal_upper = conformal.predict_upper(q_pred)
        rows.extend(_metric_rows(split, "conformal_upper", y, conformal_upper, kind="upper"))

    metrics = pd.DataFrame(rows)
    metrics.to_csv(cfg.output_path / "reports" / "model_metrics.csv", index=False)
    return metrics


def _metric_rows(split: str, model: str, y: np.ndarray, pred: np.ndarray, kind: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if y.size == 0:
        return rows
    if kind == "point":
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "mae",
                "value": float(mean_absolute_error(y, pred)),
            }
        )
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "rmse",
                "value": float(_rmse(y, pred)),
            }
        )
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "max_abs_error",
                "value": float(np.max(np.abs(y - pred))),
            }
        )
    else:
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "empirical_coverage",
                "value": empirical_coverage(y, pred),
            }
        )
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "average_bound",
                "value": float(np.mean(pred)),
            }
        )
        rows.append(
            {
                "split": split,
                "model": model,
                "metric": "average_conservatism",
                "value": float(np.mean(pred - y)),
            }
        )
    return rows


def _rmse(y: np.ndarray, pred: np.ndarray) -> float:
    if root_mean_squared_error is not None:
        return float(root_mean_squared_error(y, pred))
    return float(np.sqrt(np.mean((np.asarray(y) - np.asarray(pred)) ** 2)))


def load_model_bundle(output_dir: str | Path) -> ModelBundle:
    return joblib.load(Path(output_dir) / "models" / "model_bundle.joblib")
