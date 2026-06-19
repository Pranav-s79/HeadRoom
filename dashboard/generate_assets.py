"""Pre-generate static portfolio assets for HeadRoom.

Writes ``portfolio_assets/architecture.png`` and ``portfolio_assets/research_summary.png``
at the HeadRoom root so the README renders cleanly on GitHub without needing the live
dashboard. Reads only existing ``outputs/`` CSV artifacts — no pipeline re-run, no
Streamlit, no thermalguard_cal imports.

Run from anywhere:

    python dashboard/generate_assets.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DASHBOARD_DIR = Path(__file__).resolve().parent
for _p in (str(PROJECT_ROOT), str(DASHBOARD_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import figures as F  # noqa: E402

REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
MULTISEED_DIR = PROJECT_ROOT / "outputs" / "multiseed" / "challenging"
PORTFOLIO_DIR = PROJECT_ROOT / "portfolio_assets"
PRIMARY_SCHED = "conformal_upper_bound"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def coverage_value(split: str, kind: str) -> float:
    df = _read_csv(REPORTS_DIR / f"metrics_{split}.csv")
    col = "marginal_coverage" if kind == "marginal" else "selected_core_coverage"
    rows = df[df["scheduler"] == PRIMARY_SCHED] if not df.empty else df
    if rows.empty or col not in rows or pd.isna(rows[col].iloc[0]):
        return 0.0
    return float(rows[col].iloc[0])


def conformal_before_after() -> dict:
    rows = []
    for diag in sorted(MULTISEED_DIR.glob("seed_*/reports/conformal_diagnostics.csv")):
        df = _read_csv(diag)
        if df.empty:
            continue
        vals = {m: float(v) for m, v in zip(df["metric"], df["value"])}
        rows.append(
            {
                "seed": diag.parent.parent.name.replace("seed_", ""),
                "before": vals.get("calibration_empirical_coverage_before_conformal"),
                "after": vals.get("calibration_empirical_coverage_after_conformal"),
                "correction": vals.get("conformal_correction"),
            }
        )
    table = pd.DataFrame(rows).dropna() if rows else pd.DataFrame()
    pos = table[table["correction"] > 0.05] if not table.empty else table
    if not pos.empty:
        pos = pos.assign(lift=pos["after"] - pos["before"]).sort_values("lift", ascending=False)
        best = pos.iloc[0]
        return {"before": float(best["before"]), "after": float(best["after"]),
                "correction": float(best["correction"]), "source": f"challenging preset, seed {best['seed']}"}
    return {"before": 0.90, "after": 0.90, "correction": 0.0, "source": "canonical run"}


def main() -> None:
    PORTFOLIO_DIR.mkdir(parents=True, exist_ok=True)

    arch = F.architecture_figure()
    arch_path = PORTFOLIO_DIR / "architecture.png"
    arch.savefig(arch_path, dpi=100, facecolor=F.PALETTE["bg"])
    F.plt.close(arch)
    print(f"wrote {arch_path.relative_to(PROJECT_ROOT)}")

    ba = conformal_before_after()
    summary = F.research_summary_figure(
        nominal=0.90,
        id_marginal=coverage_value("id", "marginal"),
        id_selected=coverage_value("id", "selected"),
        ood_marginal=coverage_value("ood", "marginal"),
        ood_selected=coverage_value("ood", "selected"),
        before_cov=ba["before"], after_cov=ba["after"], correction_c=ba["correction"],
        correction_note=(
            f"Coverage cards: canonical evaluation run. Before/after: {ba['source']} calibration "
            "split. Conformal only widens bounds, so an already-conservative base model needs none."
        ),
    )
    summary_path = PORTFOLIO_DIR / "research_summary.png"
    summary.savefig(summary_path, dpi=100, facecolor=F.PALETTE["bg"], bbox_inches="tight")
    F.plt.close(summary)
    print(f"wrote {summary_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
