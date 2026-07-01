"""Importer for user-provided Kaggle files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.data.io import read_table_fast
from football_ai.scrapers.base import BaseDataSource


class KaggleFileImporter(BaseDataSource):
    """Importer for local Kaggle datasets provided by the user."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="kaggle")

    def import_file(self, path: str | Path) -> pd.DataFrame:
        """Load a user-provided structured file."""
        frame = read_table_fast(path)
        output_name = f"{Path(path).stem}_normalized.parquet"
        self.persist_frame(frame, output_name)
        return frame


if __name__ == "__main__":
    print("Use KaggleFileImporter.import_file(path).")
