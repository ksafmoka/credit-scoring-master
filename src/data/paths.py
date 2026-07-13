"""Resolve input CSV path for ingestion (Lending Club preferred)."""

from __future__ import annotations

import os
from pathlib import Path


def resolve_applications_csv(
    explicit: str | None = None,
    *,
    data_dir: str | Path | None = None,
) -> Path:
    """
    Prefer real Lending Club file when present.

    Order:
      1) explicit path / LENDING_CLUB_CSV_PATH env
      2) data/lending_club.csv
      3) data/sample_applications.csv
    """
    candidates: list[Path] = []

    if explicit:
        candidates.append(Path(explicit))
    env_path = os.getenv("LENDING_CLUB_CSV_PATH", "").strip()
    if env_path:
        candidates.append(Path(env_path))

    # Container + host layouts
    roots = []
    if data_dir is not None:
        roots.append(Path(data_dir))
    roots.extend(
        [
            Path("/opt/airflow/data"),
            Path(__file__).resolve().parents[2] / "data",
            Path("data"),
        ]
    )
    for root in roots:
        candidates.append(root / "lending_club.csv")
        candidates.append(root / "sample_applications.csv")

    seen: set[str] = set()
    for path in candidates:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        if path.is_file():
            return path

    raise FileNotFoundError(
        "No applications CSV found. Place Lending Club dump at "
        "data/lending_club.csv (preferred) or data/sample_applications.csv."
    )
