"""Build the small, git-tracked ``demo_data/`` package for the hosted demo (P5).

Trains a *compact* quick-mode bundle (fewer trees/estimators so the pickle is
small) into a temp dir, then copies just what the dashboard needs to render every
page into ``demo_data/`` — keeping the whole package under 10 MB so it can ship in
git for Streamlit Community Cloud:

  models/model_bundle.joblib          compact, loadable bundle (drives the replay)
  data/{X,y}_calibration.npy          subsampled calibration features (drift panel)
  data/feature_names.json
  reports/metrics_id.csv, metrics_ood.csv
  reports/metrics_multiseed_id.csv, metrics_multiseed_ood.csv  (mean +/- std table)
  reports/coverage_metrics.csv, model_metrics.csv, conformal_diagnostics.csv
  reports/run_manifest.json
  multiseed/challenging/seed_*/reports/conformal_diagnostics.csv  (before/after story)

The before/after-conformal diagnostics are reused from the existing
``outputs/multiseed/challenging`` runs rather than recomputed, so this script is
fast and needs no challenging-preset retraining.

    python build_demo_data.py
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models

ROOT = Path(__file__).resolve().parent
DEMO_DIR = ROOT / "demo_data"
OUTPUTS = ROOT / "outputs"
CAL_SUBSAMPLE = 800   # rows kept from the calibration features (drift panel only)


def main() -> None:
    if DEMO_DIR.exists():
        shutil.rmtree(DEMO_DIR)
    for sub in ("data", "models", "reports"):
        (DEMO_DIR / sub).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_out = Path(tmp) / "outputs"
        # Compact bundle: fewer trees/estimators keep the pickle small (<10MB total).
        cfg = make_config(
            "quick", preset="normal", random_seed=17, output_dir=str(tmp_out),
            random_forest_estimators=16, gradient_boosting_estimators=60,
        )
        generate_datasets(cfg)
        train_and_save_models(cfg)
        evaluate_and_report(cfg, write_plots=False)
        write_run_manifest(cfg, "quick")

        shutil.copy2(tmp_out / "models" / "model_bundle.joblib",
                     DEMO_DIR / "models" / "model_bundle.joblib")

        # Subsample calibration features (the drift panel only needs a sample).
        X = np.load(tmp_out / "data" / "X_calibration.npy")
        y = np.load(tmp_out / "data" / "y_calibration.npy")
        if X.shape[0] > CAL_SUBSAMPLE:
            idx = np.random.default_rng(0).choice(X.shape[0], CAL_SUBSAMPLE, replace=False)
            X, y = X[idx], y[idx]
        np.save(DEMO_DIR / "data" / "X_calibration.npy", X.astype(np.float32))
        np.save(DEMO_DIR / "data" / "y_calibration.npy", y.astype(np.float32))
        shutil.copy2(tmp_out / "data" / "feature_names.json", DEMO_DIR / "data" / "feature_names.json")

        for name in ("metrics_id.csv", "metrics_ood.csv", "coverage_metrics.csv",
                     "model_metrics.csv", "conformal_diagnostics.csv", "run_manifest.json"):
            src = tmp_out / "reports" / name
            if src.exists():
                shutil.copy2(src, DEMO_DIR / "reports" / name)

    # Multi-seed mean+/-std tables (so the demo also shows aggregate numbers).
    for name in ("metrics_multiseed_id.csv", "metrics_multiseed_ood.csv"):
        src = OUTPUTS / "reports" / name
        if src.exists():
            shutil.copy2(src, DEMO_DIR / "reports" / name)

    # Reuse existing challenging diagnostics for the before/after conformal story.
    src_root = OUTPUTS / "multiseed" / "challenging"
    copied = 0
    for diag in sorted(src_root.glob("seed_*/reports/conformal_diagnostics.csv"))[:3]:
        seed_name = diag.parent.parent.name
        dst = DEMO_DIR / "multiseed" / "challenging" / seed_name / "reports"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(diag, dst / "conformal_diagnostics.csv")
        copied += 1

    files = [f for f in DEMO_DIR.rglob("*") if f.is_file()]
    total = sum(f.stat().st_size for f in files)
    print(f"demo_data/ built — {total / 1e6:.2f} MB across {len(files)} files "
          f"({copied} challenging-diagnostic files)")
    if total > 10e6:
        print("[!] WARNING: demo_data exceeds 10 MB — trim further before committing.")


if __name__ == "__main__":
    main()
