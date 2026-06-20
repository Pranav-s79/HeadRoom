"""Multi-seed evaluation WITHOUT retraining (Work Stream 2b).

After a full-mode training run, this re-runs the scheduler evaluation N times with
different workload-generation seeds. The trained model bundle, conformal
calibrator, and on-disk calibration feature statistics are held fixed; only the
random seed that drives the simulator and workload generator changes. This
isolates result variance due to workload realization from variance due to
training.

For each (split, scheduler) it reports mean +/- std across seeds of:
  - peak temperature
  - hotspot violations
  - completed tasks
  - marginal coverage          (conformal only)
  - selected-core coverage     (conformal only)

Outputs (under ``outputs/reports/``):
  - metrics_multiseed_id.csv
  - metrics_multiseed_ood.csv
  - metrics_multiseed_raw.csv  (per-seed rows, for auditing)

Usage:
    python run_eval_multiseed.py --full --preset normal --seeds 17 23 29 31 41
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import load_split
from thermalguard_cal.evaluate import (
    _calibration_feature_stats,
    evaluate_scheduler,
    make_schedulers,
)
from thermalguard_cal.models import load_model_bundle

# Columns aggregated as mean +/- std across workload seeds.
AGG_COLUMNS = [
    "peak_temperature",
    "average_max_temperature",
    "hotspot_violations",
    "hotspot_timestep_pct",
    "completed_tasks",
    "marginal_coverage",
    "selected_core_coverage",
    "selected_coverage_gap",
]


def evaluate_one_seed(mode: str, preset: str, seed: int, output_dir: Path) -> pd.DataFrame:
    """Evaluate all schedulers for a single workload seed, reusing trained models."""
    cfg = make_config(mode, preset=preset, random_seed=int(seed), output_dir=str(output_dir))
    bundle = load_model_bundle(cfg.output_path)
    X_cal, _, _ = load_split(cfg.output_path, "calibration")
    cal_stats = _calibration_feature_stats(X_cal)

    rows: list[dict[str, object]] = []
    for split in ("id", "ood"):
        for scheduler in make_schedulers(cfg, bundle):
            result = evaluate_scheduler(cfg, scheduler, bundle, split, cal_stats)
            row = result["summary"].to_dict()
            row["seed"] = int(seed)
            rows.append(row)
    return pd.DataFrame(rows)


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    grouped = per_seed.groupby(["split", "scheduler"], dropna=False)
    out: list[dict[str, object]] = []
    for (split, scheduler), group in grouped:
        row: dict[str, object] = {
            "split": split,
            "scheduler": scheduler,
            "n_seeds": int(group["seed"].nunique()),
        }
        for col in AGG_COLUMNS:
            values = pd.to_numeric(group.get(col), errors="coerce")
            has = values.notna().any()
            mean = float(values.mean()) if has else np.nan
            std = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            row[f"{col}_mean"] = mean
            row[f"{col}_std"] = std
            row[f"{col}_mean_std"] = f"{mean:.3f} +/- {std:.3f}" if has else ""
        out.append(row)
    return pd.DataFrame(out).sort_values(["split", "scheduler"]).reset_index(drop=True)


def run_eval_multiseed(mode: str, preset: str, seeds: list[int], output_dir: str = "outputs") -> pd.DataFrame:
    output_path = Path(output_dir)
    reports_dir = output_path / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        print(f"Evaluating workload seed {seed} (no retraining)")
        frames.append(evaluate_one_seed(mode, preset, seed, output_path))
    per_seed = pd.concat(frames, ignore_index=True)
    per_seed.to_csv(reports_dir / "metrics_multiseed_raw.csv", index=False)

    agg = aggregate(per_seed)
    agg[agg["split"] == "id"].to_csv(reports_dir / "metrics_multiseed_id.csv", index=False)
    agg[agg["split"] == "ood"].to_csv(reports_dir / "metrics_multiseed_ood.csv", index=False)

    _write_summary_markdown(reports_dir / "metrics_multiseed_summary.md", preset, seeds, agg)
    _inject_into_final_report(reports_dir / "final_report.md", preset, seeds, agg)
    return agg


def _inject_into_final_report(report_path: Path, preset: str, seeds: list[int], agg: pd.DataFrame) -> None:
    """Replace single-seed headline numbers with a multi-seed mean+/-std section.

    Inserts (or refreshes) a `## Multi-seed Headline Results` block just before the
    `## Limitations` section so the report's top-line numbers reflect the 5-seed
    aggregate rather than a single workload realization.
    """
    if not report_path.exists():
        return
    view = agg[
        ["split", "scheduler", "peak_temperature_mean_std", "hotspot_violations_mean_std",
         "completed_tasks_mean_std", "marginal_coverage_mean_std", "selected_core_coverage_mean_std"]
    ].rename(columns={
        "peak_temperature_mean_std": "peak_temp (C)",
        "hotspot_violations_mean_std": "violations",
        "completed_tasks_mean_std": "completed",
        "marginal_coverage_mean_std": "marginal_cov",
        "selected_core_coverage_mean_std": "selected_cov",
    })
    section = (
        "## Multi-seed Headline Results\n\n"
        f"These are the FINAL project numbers: all 8 schedulers re-evaluated on "
        f"{len(seeds)} workload seeds ({', '.join(str(s) for s in seeds)}) using the "
        f"already-trained models (no retraining). Reported as mean +/- sample std.\n\n"
        f"{view.to_markdown(index=False)}\n\n"
    )
    text = report_path.read_text(encoding="utf-8")
    marker = "## Multi-seed Headline Results"
    if marker in text:
        start = text.find(marker)
        nxt = text.find("\n## ", start + 1)
        text = text[:start] + section + (text[nxt + 1:] if nxt != -1 else "")
    else:
        anchor = text.find("## Limitations")
        if anchor != -1:
            text = text[:anchor] + section + text[anchor:]
        else:
            text = text.rstrip() + "\n\n" + section
    report_path.write_text(text, encoding="utf-8")


def _write_summary_markdown(path: Path, preset: str, seeds: list[int], agg: pd.DataFrame) -> None:
    view = agg[
        [
            "split",
            "scheduler",
            "peak_temperature_mean_std",
            "hotspot_violations_mean_std",
            "completed_tasks_mean_std",
            "marginal_coverage_mean_std",
            "selected_core_coverage_mean_std",
        ]
    ].rename(
        columns={
            "peak_temperature_mean_std": "peak_temp (C)",
            "hotspot_violations_mean_std": "violations",
            "completed_tasks_mean_std": "completed",
            "marginal_coverage_mean_std": "marginal_cov",
            "selected_core_coverage_mean_std": "selected_cov",
        }
    )
    table = view.to_markdown(index=False)
    path.write_text(
        f"""# Multi-seed Evaluation (no retraining): {preset}

Workload seeds: {", ".join(str(s) for s in seeds)}

Models, conformal calibrator, and calibration feature statistics are held fixed
across seeds; only the simulator / workload-generation seed changes. Values are
mean +/- sample standard deviation across seeds. These are the headline numbers
for the writeup and resume.

{table}
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-evaluate schedulers across workload seeds without retraining.")
    parser.add_argument("--quick", action="store_true", help="Use quick settings (must match the trained run).")
    parser.add_argument("--full", action="store_true", help="Use full settings (must match the trained run).")
    parser.add_argument("--preset", choices=("easy", "normal", "challenging", "stress"), default="normal")
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Workload seeds to evaluate.")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    agg = run_eval_multiseed(mode=mode, preset=args.preset, seeds=args.seeds, output_dir=args.output_dir)
    print(agg.to_string(index=False))


if __name__ == "__main__":
    main()
