"""FBref scraper and CSV importer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.scrapers.base import BaseDataSource


class FBrefSource(BaseDataSource):
    """Importer for FBref tables or prepared exports."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="fbref")

    def import_from_file(self, path: str | Path) -> pd.DataFrame:
        """Load prepared FBref exports from CSV or Parquet."""
        file_path = Path(path)
        if file_path.suffix.lower() == ".csv":
            frame = pd.read_csv(file_path)
        elif file_path.suffix.lower() == ".parquet":
            frame = pd.read_parquet(file_path)
        else:
            raise ValueError("FBref importer supports CSV and Parquet files only.")
        normalized = self.normalize_frame(frame)
        self.persist_frame(normalized, "fbref_import.parquet")
        return normalized

    def import_first_table_from_url(self, url: str) -> pd.DataFrame:
        """Load the first HTML table found on an FBref page."""
        frame = self.fetch_table(url)
        normalized = self.normalize_frame(frame)
        self.persist_frame(normalized, "fbref_table.parquet")
        return normalized

    def normalize_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize FBref columns where possible."""
        normalized = frame.copy()
        for column in ("Squad", "Team", "home_team_name", "away_team_name"):
            if column in normalized.columns:
                normalized[column] = normalized[column].astype(str).map(canonicalize_team_name)
        normalized["source"] = "fbref"
        return normalized


if __name__ == "__main__":
    print("Use FBrefSource.import_from_file(path) or import_first_table_from_url(url).")
