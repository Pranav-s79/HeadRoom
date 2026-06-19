from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


def create_research_review_bundle(project_root: Path, output_root: Path) -> Path:
    project_root = project_root.resolve()
    output_root = (project_root / output_root).resolve() if not output_root.is_absolute() else output_root.resolve()
    reports_dir = output_root / "reports"
    figures_dir = output_root / "figures"
    bundle_path = reports_dir / "research_review_bundle.zip"
    reports_dir.mkdir(parents=True, exist_ok=True)

    include_files: list[Path] = []
    include_files.extend(sorted((project_root / "thermalguard_cal").glob("*.py")))
    include_files.extend(sorted((project_root / "tests").glob("*.py")))
    include_files.extend(sorted(project_root.glob("run_*.py")))
    for rel in ("README.md", "requirements.txt"):
        path = project_root / rel
        if path.exists():
            include_files.append(path)
    for folder in (reports_dir, figures_dir):
        if folder.exists():
            include_files.extend(
                path
                for path in sorted(folder.rglob("*"))
                if path.is_file() and path.resolve() != bundle_path.resolve()
            )
    data_dir = output_root / "data"
    for name in ("dataset_summary.json", "feature_names.json"):
        path = data_dir / name
        if path.exists():
            include_files.append(path)

    with ZipFile(bundle_path, "w", compression=ZIP_DEFLATED) as archive:
        seen: set[str] = set()
        for path in include_files:
            if not path.exists() or not path.is_file():
                continue
            arcname = path.relative_to(project_root).as_posix()
            if arcname in seen:
                continue
            seen.add(arcname)
            archive.write(path, arcname)
    return bundle_path
