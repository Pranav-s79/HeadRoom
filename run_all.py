"""HeadRoom (thermalguard_cal) — single-command end-to-end pipeline.

    python run_all.py --quick        # dev mode (~minutes)
    python run_all.py --full         # full research settings (longer)
    python run_all.py --quick --seed 7

Runs, in order: directory setup -> dataset generation -> model training ->
ID + OOD scheduler evaluation -> figures -> final_report.md -> run_manifest.json,
then prints a summary and a block of sanity checks. Every step is wrapped so a
failure prints the step name and a full traceback instead of silently continuing.

The ``--seed`` flag (default 42) controls all randomness globally: it seeds the
simulator, workload generators, sensor dropout, model fitting, and the random
scheduler, all derived from ``cfg.random_seed``.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import pandas as pd

from thermalguard_cal.config import ThermalGuardConfig, ensure_output_dirs, make_config
from thermalguard_cal.dataset import generate_datasets
from thermalguard_cal.evaluate import evaluate_and_report, write_run_manifest
from thermalguard_cal.models import load_model_bundle, train_and_save_models
from thermalguard_cal.review_bundle import create_research_review_bundle

PRIMARY_SCHED = "conformal_upper_bound"
NOMINAL_COVERAGE = 0.90


def _step(name: str, fn):
    """Run a pipeline step, surfacing failures with the step name + traceback."""
    print(f"\n=== {name} ===", flush=True)
    try:
        return fn()
    except Exception:  # noqa: BLE001 — we re-raise after a clear message
        print(f"\n[run_all] STEP FAILED: {name}", file=sys.stderr)
        traceback.print_exc()
        raise SystemExit(f"Pipeline aborted at step: {name}")


def _print_model_parameters(cfg: ThermalGuardConfig) -> None:
    bundle = load_model_bundle(cfg.output_path)
    forest = bundle.forest_point
    quantile = bundle.quantile_upper
    print("Saved models (outputs/models/):")
    print("  linear_point            : Ridge pipeline (StandardScaler + Ridge)")
    print(f"  forest_point            : RandomForest n_estimators={getattr(forest, 'n_estimators', '?')}, "
          f"max_depth={getattr(forest, 'max_depth', '?')}")
    print(f"  quantile_upper          : GradientBoosting n_estimators={getattr(quantile, 'n_estimators', '?')}, "
          f"learning_rate={getattr(quantile, 'learning_rate', '?')}, "
          f"max_depth={getattr(quantile, 'max_depth', '?')}, alpha={getattr(quantile, 'alpha', '?')}")
    print(f"  conformal_calibrator    : correction=+{bundle.conformal.correction:.4f}°C, "
          f"q_level={bundle.conformal.q_level:.4f}, n_cal={bundle.conformal.n_calibration}")


def _conformal_diagnostics(cfg: ThermalGuardConfig) -> None:
    path = cfg.output_path / "reports" / "conformal_diagnostics.csv"
    if not path.exists():
        print("  (conformal_diagnostics.csv not found)")
        return
    diag = pd.read_csv(path)
    vals = {m: float(v) for m, v in zip(diag["metric"], diag["value"])}
    correction = vals.get("conformal_correction")
    before = vals.get("calibration_empirical_coverage_before_conformal")
    after = vals.get("calibration_empirical_coverage_after_conformal")
    print(f"  nominal target coverage           : {cfg.conformal_target_coverage:.3f}")
    print(f"  conformal correction (delta)      : +{correction:.4f}°C" if correction is not None else "  correction: n/a")
    print(f"  calibration coverage (pre)        : {before*100:.2f}%" if before is not None else "  pre: n/a")
    print(f"  calibration coverage (post)       : {after*100:.2f}%" if after is not None else "  post: n/a")
    if after is not None:
        lo, hi = NOMINAL_COVERAGE - 0.02, NOMINAL_COVERAGE + 0.02
        if lo <= after <= hi:
            print(f"  post within +/-2% of target?      : OK ({lo*100:.0f}-{hi*100:.0f}%)")
        elif after > hi:
            print(f"  post within +/-2% of target?      : over-covers "
                  f"({after*100:.1f}% >= {NOMINAL_COVERAGE*100:.0f}% — still valid; CQR guarantees >= target)")
        else:
            print(f"  post within +/-2% of target?      : UNDER target ({after*100:.1f}%)")
        if after < 0.85 or after > 0.97:
            print("  [!] NOTE: calibration coverage is far from the 90% target for this seed "
                  "(small quick-mode calibration set); see multi-seed results for the headline figure.")


def _coverage(coverage: pd.DataFrame, split: str, token: str) -> float | None:
    rows = coverage[(coverage["split"] == split)
                    & coverage["coverage_type"].str.contains(token, case=False, na=False)]
    if rows.empty or pd.isna(rows["empirical_coverage"].iloc[0]):
        return None
    return float(rows["empirical_coverage"].iloc[0])


def _sched_val(metrics: pd.DataFrame, split: str, scheduler: str, col: str):
    rows = metrics[(metrics["split"] == split) & (metrics["scheduler"] == scheduler)]
    if rows.empty or pd.isna(rows[col].iloc[0]):
        return None
    return float(rows[col].iloc[0])


def _print_topline(metrics: pd.DataFrame, coverage: pd.DataFrame) -> None:
    cols = ["scheduler", "peak_temperature", "hotspot_violations", "completed_tasks"]
    for split in ("id", "ood"):
        sub = metrics[metrics["split"] == split][cols].copy()
        sub = sub.sort_values("peak_temperature")
        print(f"\nTop-line scheduler comparison — {split.upper()} (sorted by peak temp):")
        print(sub.to_string(index=False))
    print("\nConformal coverage (calibrated upper bound vs 90% target):")
    for split in ("id", "ood"):
        m = _coverage(coverage, split, "marginal")
        s = _coverage(coverage, split, "selected")
        ms = "n/a" if m is None else f"{m*100:.1f}%"
        ss = "n/a" if s is None else f"{s*100:.1f}%"
        print(f"  {split.upper():3s}  marginal={ms:>7s}   selected-core={ss:>7s}")


def _sanity_checks(metrics: pd.DataFrame, coverage: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)
    results: list[tuple[str, bool, str]] = []

    # 1. Random should not be the best (lowest peak temp) on any split.
    rnd_peak_best = any(
        _sched_val(metrics, sp, "random", "peak_temperature")
        == metrics[metrics["split"] == sp]["peak_temperature"].min()
        for sp in ("id", "ood")
        if not metrics[metrics["split"] == sp].empty
    )
    results.append(("Random is NOT the best scheduler (peak temp)", not rnd_peak_best,
                    "random had the lowest peak temp somewhere" if rnd_peak_best else ""))

    # 2. Oracle (privileged true-temp) beats its observed-sensor counterpart on OOD.
    #    The greedy oracle minimizes *current* temperature, so it is not guaranteed
    #    to be top-1 for *future* peak under bursty load — but with perfect
    #    information it must beat the same heuristic run on noisy sparse sensors.
    oracle_peak = _sched_val(metrics, "ood", "coolest_core_oracle_true_temp", "peak_temperature")
    observed_peak = _sched_val(metrics, "ood", "coolest_core_observed", "peak_temperature")
    ood = metrics[metrics["split"] == "ood"]
    not_worst = oracle_peak is not None and not ood.empty and oracle_peak < ood["peak_temperature"].max()
    beats_observed = oracle_peak is not None and observed_peak is not None and oracle_peak <= observed_peak
    oracle_ok = bool(not_worst and beats_observed)
    results.append(("Oracle beats its observed-sensor twin and is not worst (OOD)", oracle_ok,
                    f"oracle={oracle_peak:.1f} vs observed={observed_peak:.1f}"
                    if oracle_peak is not None and observed_peak is not None else "missing"))

    # 3. Conformal fewer violations than random and round-robin on ID.
    conf_id_v = _sched_val(metrics, "id", PRIMARY_SCHED, "hotspot_violations")
    rnd_id_v = _sched_val(metrics, "id", "random", "hotspot_violations")
    rr_id_v = _sched_val(metrics, "id", "round_robin", "hotspot_violations")
    conf_ok = (conf_id_v is not None and rnd_id_v is not None and rr_id_v is not None
               and conf_id_v <= rnd_id_v and conf_id_v <= rr_id_v)
    results.append(("Conformal <= random and round-robin violations (ID)", bool(conf_ok),
                    f"conf={conf_id_v}, rnd={rnd_id_v}, rr={rr_id_v}"))

    # 4. OOD coverage meaningfully lower than ID (>10% gap).
    id_m = _coverage(coverage, "id", "marginal")
    ood_m = _coverage(coverage, "ood", "marginal")
    gap_ok = id_m is not None and ood_m is not None and (id_m - ood_m) > 0.10
    results.append(("ID coverage exceeds OOD coverage by >10pp", bool(gap_ok),
                    f"ID={id_m*100:.1f}% OOD={ood_m*100:.1f}%" if id_m and ood_m else "coverage missing"))

    # 5. Peak temps physically plausible (35-150C).
    peaks = pd.to_numeric(metrics["peak_temperature"], errors="coerce").dropna()
    plausible = bool((peaks >= 35).all() and (peaks <= 150).all())
    results.append(("Peak temps in plausible 35-150°C range", plausible,
                    f"min={peaks.min():.1f} max={peaks.max():.1f}"))

    # 6. Completed tasks similar across schedulers (within 20% on each split).
    tput_ok = True
    detail = []
    for split in ("id", "ood"):
        ct = pd.to_numeric(metrics[metrics["split"] == split]["completed_tasks"], errors="coerce").dropna()
        if ct.empty or ct.max() == 0:
            continue
        spread = (ct.max() - ct.min()) / ct.max()
        detail.append(f"{split}:{spread*100:.0f}%")
        if spread > 0.20:
            tput_ok = False
    results.append(("Completed tasks within 20% across schedulers", tput_ok, " ".join(detail)))

    allpass = True
    for label, ok, detail in results:
        mark = "PASS" if ok else "FAIL"
        if not ok:
            allpass = False
        print(f"  [{mark}] {label}" + (f"   ({detail})" if detail else ""))
    print("-" * 60)
    print("ALL SANITY CHECKS PASSED" if allpass else "SOME SANITY CHECKS FAILED — investigate above")
    return allpass


def main() -> None:
    parser = argparse.ArgumentParser(description="Run HeadRoom (thermalguard_cal) end to end.")
    parser.add_argument("--quick", action="store_true", help="Use quick development settings (~minutes).")
    parser.add_argument("--full", action="store_true", help="Use full research settings (longer).")
    parser.add_argument(
        "--preset",
        choices=("easy", "normal", "challenging", "stress"),
        default="normal",
        help="Workload/simulator stress preset.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Global base random seed (default 42).")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        help="Run a multiseed retraining aggregate instead of a single pipeline run.",
    )
    args = parser.parse_args()
    mode = "full" if args.full else "quick"

    if args.seeds:
        from run_multiseed import run_multiseed
        run_multiseed(mode=mode, preset=args.preset, seeds=args.seeds)
        return

    cfg = make_config(mode, preset=args.preset, random_seed=args.seed)
    print(f"HeadRoom pipeline — mode={mode}, preset={cfg.preset}, seed={cfg.random_seed}")

    _step("1/7 Create output directories", lambda: ensure_output_dirs(cfg))

    summaries = _step("2/7 Generate datasets", lambda: generate_datasets(cfg))
    for split, summary in summaries.items():
        print(f"  {split:12s}: {summary['rows']:>7d} rows, {summary['features']} features, "
              f"{summary['episodes']} episodes")

    model_metrics = _step("3/7 Train models (linear, RF, quantile, conformal)",
                          lambda: train_and_save_models(cfg))
    _print_model_parameters(cfg)
    print("\nConformal calibration:")
    _conformal_diagnostics(cfg)

    scheduler_metrics, coverage = _step(
        "4/7 Evaluate 8 schedulers on ID + OOD (+ figures + final_report.md)",
        lambda: evaluate_and_report(cfg),
    )
    _step("5/7 Write run manifest", lambda: write_run_manifest(cfg, mode))
    bundle_path = _step("6/7 Build research review bundle",
                        lambda: create_research_review_bundle(Path.cwd(), cfg.output_path))

    print("\n" + "=" * 60)
    print("RUN SUMMARY")
    print("=" * 60)
    print("\nModel metrics (test_id):")
    mm_id = model_metrics[model_metrics["split"] == "test_id"]
    print(mm_id.to_string(index=False))
    _print_topline(scheduler_metrics, coverage)

    passed = _sanity_checks(scheduler_metrics, coverage)

    report = cfg.output_path / "reports" / "final_report.md"
    print(f"\nFinal report : {report}")
    print(f"Review bundle: {bundle_path}")
    print(f"All outputs  : {cfg.output_path}/")
    if not passed:
        raise SystemExit("Pipeline completed but some sanity checks failed (see above).")


if __name__ == "__main__":
    main()
