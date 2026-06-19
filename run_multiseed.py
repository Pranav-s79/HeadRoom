from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermalguard_cal.config import make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import train_and_save_models
from thermalguard_cal.review_bundle import create_research_review_bundle


AGGREGATE_COLUMNS = [
    "peak_temperature",
    "average_max_temperature",
    "hotspot_violations",
    "hotspot_timestep_pct",
    "completed_tasks",
    "marginal_coverage",
    "selected_core_coverage",
    "selected_coverage_gap",
]


def run_multiseed(mode: str, preset: str, seeds: list[int]) -> pd.DataFrame:
    output_root = Path("outputs")
    seed_root = output_root / "multiseed" / preset
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    frames: list[pd.DataFrame] = []
    for seed in seeds:
        overrides = _multiseed_overrides(mode)
        cfg = make_config(
            mode,
            preset=preset,
            random_seed=int(seed),
            output_dir=str(seed_root / f"seed_{seed}"),
            **overrides,
        )
        print(f"Running seed {seed} with {preset} preset")
        generate_datasets(cfg)
        train_and_save_models(cfg)
        metrics, _coverage = evaluate_and_report(cfg)
        write_run_manifest(cfg, mode)
        metrics = metrics.copy()
        metrics["seed"] = int(seed)
        frames.append(metrics)

    all_metrics = pd.concat(frames, ignore_index=True)
    raw_path = reports_dir / f"multiseed_metrics_raw_{preset}.csv"
    all_metrics.to_csv(raw_path, index=False)

    aggregate = _aggregate_metrics(all_metrics)
    metrics_path = reports_dir / f"multiseed_metrics_{preset}.csv"
    aggregate.to_csv(metrics_path, index=False)
    summary_path = reports_dir / f"multiseed_summary_{preset}.md"
    summary_path.write_text(_summary_markdown(preset, seeds, aggregate), encoding="utf-8")

    _plot_multiseed_hotspots(aggregate, figures_dir / f"multiseed_hotspots_{preset}.png")
    _plot_multiseed_peak_temp(aggregate, figures_dir / f"multiseed_peak_temp_{preset}.png")
    _plot_multiseed_coverage(aggregate, figures_dir / f"multiseed_coverage_{preset}.png")
    _update_final_report_multiseed(preset, summary_path)
    create_research_review_bundle(Path.cwd(), output_root)
    return aggregate


def _multiseed_overrides(mode: str) -> dict[str, int]:
    if mode != "quick":
        return {}
    return {
        "episode_length": 90,
        "train_episodes": 5,
        "calibration_episodes": 2,
        "test_episodes": 2,
        "ood_episodes": 2,
        "random_forest_estimators": 24,
        "gradient_boosting_estimators": 50,
        "max_train_rows": 20_000,
    }


def _aggregate_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    grouped = metrics.groupby(["preset", "split", "scheduler"], dropna=False)
    rows: list[dict[str, object]] = []
    for (preset, split, scheduler), group in grouped:
        row: dict[str, object] = {
            "preset": preset,
            "split": split,
            "scheduler": scheduler,
            "n_seeds": int(group["seed"].nunique()),
        }
        for column in AGGREGATE_COLUMNS:
            values = pd.to_numeric(group[column], errors="coerce")
            row[f"{column}_mean"] = float(values.mean()) if values.notna().any() else np.nan
            row[f"{column}_std"] = float(values.std(ddof=1)) if values.notna().sum() > 1 else 0.0
            if values.notna().any():
                row[f"{column}_mean_std"] = f"{row[f'{column}_mean']:.3f} +/- {row[f'{column}_std']:.3f}"
            else:
                row[f"{column}_mean_std"] = ""
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["split", "scheduler"]).reset_index(drop=True)


def _summary_markdown(preset: str, seeds: list[int], aggregate: pd.DataFrame) -> str:
    view_cols = [
        "split",
        "scheduler",
        "peak_temperature_mean_std",
        "average_max_temperature_mean_std",
        "hotspot_violations_mean_std",
        "hotspot_timestep_pct_mean_std",
        "completed_tasks_mean_std",
        "marginal_coverage_mean_std",
        "selected_core_coverage_mean_std",
        "selected_coverage_gap_mean_std",
    ]
    table = aggregate[view_cols].to_markdown(index=False)
    return f"""# Multiseed Summary: {preset}

Seeds: {", ".join(str(seed) for seed in seeds)}

Values are reported as mean +/- sample standard deviation across seeds. Quick
multiseed uses compact per-seed settings so five independent runs complete in a
verification pass; use `run_all.py --quick --preset {preset}` for the fuller
single-seed quick result.

{table}
"""


def _plot_multiseed_hotspots(aggregate: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    _plot_metric_bars(
        aggregate,
        "hotspot_violations",
        "Mean hotspot timestep count",
        "Multiseed hotspot violations",
        path,
    )


def _plot_multiseed_peak_temp(aggregate: pd.DataFrame, path: Path) -> None:
    _plot_metric_bars(
        aggregate,
        "peak_temperature",
        "Mean peak temperature (C)",
        "Multiseed peak temperature",
        path,
    )


def _plot_multiseed_coverage(aggregate: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    conf = aggregate[aggregate["scheduler"] == "conformal_upper_bound"].copy()
    if conf.empty:
        return
    labels = [row["split"].upper() for _, row in conf.iterrows()]
    selected = conf["selected_core_coverage_mean"].to_numpy(dtype=float)
    selected_err = conf["selected_core_coverage_std"].to_numpy(dtype=float)
    marginal = conf["marginal_coverage_mean"].to_numpy(dtype=float)
    marginal_err = conf["marginal_coverage_std"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x - 0.18, marginal, width=0.36, yerr=marginal_err, label="Marginal", color="#476c9b")
    ax.bar(x + 0.18, selected, width=0.36, yerr=selected_err, label="Selected-core", color="#2b7a78")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Coverage")
    ax.set_title("Multiseed conformal coverage")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _plot_metric_bars(aggregate: pd.DataFrame, metric: str, ylabel: str, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    if aggregate.empty:
        return
    splits = list(aggregate["split"].drop_duplicates())
    schedulers = list(aggregate["scheduler"].drop_duplicates())
    x = np.arange(len(schedulers))
    width = 0.36 if len(splits) == 2 else 0.7
    fig, ax = plt.subplots(figsize=(12, 5))
    for idx, split in enumerate(splits):
        group = aggregate[aggregate["split"] == split].set_index("scheduler")
        means = [float(group.loc[scheduler, f"{metric}_mean"]) if scheduler in group.index else np.nan for scheduler in schedulers]
        stds = [float(group.loc[scheduler, f"{metric}_std"]) if scheduler in group.index else np.nan for scheduler in schedulers]
        offset = (idx - (len(splits) - 1) / 2) * width
        ax.bar(x + offset, means, width=width, yerr=stds, label=split.upper())
    ax.set_xticks(x)
    ax.set_xticklabels(schedulers, rotation=35, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def _update_final_report_multiseed(preset: str, summary_path: Path) -> None:
    report_path = Path("outputs") / "reports" / "final_report.md"
    if not report_path.exists():
        return
    summary = summary_path.read_text(encoding="utf-8")
    section = "## Multiseed Result\n\n" + summary.split("\n", 1)[1].strip() + "\n"
    text = report_path.read_text(encoding="utf-8")
    start = text.find("## Multiseed Result")
    if start != -1:
        next_start = text.find("\n## ", start + 1)
        text = text[:start].rstrip() + "\n\n" + section + ("\n" + text[next_start + 1 :].lstrip() if next_start != -1 else "")
    else:
        marker = "## Limitations"
        marker_at = text.find(marker)
        if marker_at == -1:
            text = text.rstrip() + "\n\n" + section
        else:
            text = text[:marker_at].rstrip() + "\n\n" + section + "\n" + text[marker_at:].lstrip()
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ThermalGuard-Cal across multiple seeds.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings.")
    parser.add_argument("--full", action="store_true", help="Use full research settings.")
    parser.add_argument(
        "--preset",
        choices=("easy", "normal", "challenging", "stress"),
        default="challenging",
        help="Workload/simulator stress preset.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True, help="Base random seeds.")
    args = parser.parse_args()
    mode = "full" if args.full else "quick"
    aggregate = run_multiseed(mode=mode, preset=args.preset, seeds=args.seeds)
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
