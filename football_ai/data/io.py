"""Tabular I/O helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    import polars as pl
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    pl = None


def ensure_parent_directory(path: Path) -> None:
    """Create parent directories for a file path."""
    path.parent.mkdir(parents=True, exist_ok=True)


def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV, Parquet, or JSON file into a pandas DataFrame."""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(file_path)
    if suffix == ".parquet":
        return pd.read_parquet(file_path)
    if suffix == ".json":
        return pd.read_json(file_path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def read_table_fast(path: str | Path) -> pd.DataFrame:
    """Read a file with Polars when available for faster ingestion."""
    file_path = Path(path)
    if pl is None:
        return read_table(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(file_path).to_pandas()
    if suffix == ".parquet":
        return pl.read_parquet(file_path).to_pandas()
    return read_table(file_path)


def write_table(frame: pd.DataFrame, path: str | Path) -> None:
    """Persist a pandas DataFrame based on file extension."""
    file_path = Path(path)
    ensure_parent_directory(file_path)
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        frame.to_csv(file_path, index=False)
        return
    if suffix == ".parquet":
        frame.to_parquet(file_path, index=False)
        return
    if suffix == ".json":
        frame.to_json(file_path, orient="records", indent=2, date_format="iso")
        return
    raise ValueError(f"Unsupported file extension: {suffix}")
