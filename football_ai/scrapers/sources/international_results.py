"""Full-history international results importer (martj42/international_results).

CC0-licensed, ~49,500 men's full-international matches from 1872, actively
maintained (verified 2026-07-24). Fills the gap the project's own base file
has always had -- `international_results_modern_with_elo.parquet` only goes
back to 2000 -- and is the prerequisite for any historical record ("first
time since 1962") the planned `insights/` module needs.

Two output frames, deliberately not one:
- `records_frame`: full history, 1872-today. For facts/records only.
- `training_frame`: `records_frame` filtered to a recent cutoff. Pre-~1990
  football has different scoring distributions and would quietly degrade
  model training if fed in unfiltered -- never pass `records_frame` to
  `build_features`/`run_training` directly.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import argparse
import logging

import pandas as pd

from football_ai.data.enrichment import merge_advanced_stats
from football_ai.data.io import read_table, write_table
from football_ai.scrapers.base import BaseDataSource


LOGGER = logging.getLogger(__name__)

_BASE_URL = "https://raw.githubusercontent.com/martj42/international_results/master"
_DEFAULT_TRAINING_CUTOFF_YEAR = 1990


class InternationalResultsSource(BaseDataSource):
    """Fetches and normalizes martj42/international_results' full-history CSVs."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="international_results")

    def fetch_results(self) -> pd.DataFrame:
        """Fetch the raw results.csv (date, home_team, away_team, home_score, away_score, ...)."""
        return pd.read_csv(StringIO(self.fetch_text(f"{_BASE_URL}/results.csv")))

    def fetch_goalscorers(self) -> pd.DataFrame:
        """Fetch the raw goalscorers.csv (per-goal scorer/minute/own-goal/penalty)."""
        return pd.read_csv(StringIO(self.fetch_text(f"{_BASE_URL}/goalscorers.csv")))

    def fetch_shootouts(self) -> pd.DataFrame:
        """Fetch the raw shootouts.csv (penalty shootout winners)."""
        return pd.read_csv(StringIO(self.fetch_text(f"{_BASE_URL}/shootouts.csv")))

    def fetch_former_names(self) -> pd.DataFrame:
        """Fetch former_names.csv (current/former team name + validity date range)."""
        return pd.read_csv(StringIO(self.fetch_text(f"{_BASE_URL}/former_names.csv")))

    def normalize_results(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Map martj42's results.csv columns onto this project's wide-matches schema.

        Near 1:1 already (verified 2026-07-24: 25,430 of our existing
        25,434-row base file's rows match this source exactly on
        (canonicalized home, canonicalized away, date) -- team-name
        conventions already agree almost everywhere without any extra
        resolution work).
        """
        normalized = pd.DataFrame(
            {
                "match_date": pd.to_datetime(raw["date"]),
                "home_team_name": raw["home_team"],
                "away_team_name": raw["away_team"],
                "home_score": raw["home_score"],
                "away_score": raw["away_score"],
                "tournament": raw["tournament"],
                "city": raw["city"],
                "country": raw["country"],
                "neutral_venue": raw["neutral"].astype(bool),
            }
        )
        return normalized.sort_values("match_date").reset_index(drop=True)

    def build_records_frame(self, elo_source_path: str | Path | None = None) -> pd.DataFrame:
        """Build the full-history records frame: martj42 identity/score + our Elo where it overlaps.

        martj42 is the merge *base* (identity/score/date/tournament always
        wins -- it's the larger, actively-maintained source) and our
        existing Elo file is *supplemental* (`merge_advanced_stats` only
        fills columns the base doesn't already have -- martj42 has no Elo
        columns at all, so this is pure enrichment, never a real conflict).
        Pre-2000 rows simply get no Elo (`eloratings.net` coverage doesn't
        reach that far either) -- expected, not a bug; `records_frame` isn't
        meant to feed model training anyway.
        """
        results = self.normalize_results(self.fetch_results())
        if elo_source_path is None:
            return results
        elo_source = Path(elo_source_path)
        if not elo_source.exists():
            LOGGER.warning(
                "Elo source %s not found; records_frame will have no Elo columns.",
                elo_source,
            )
            return results
        elo_frame = read_table(elo_source)
        return merge_advanced_stats(base_matches=results, supplemental_matches=elo_frame)

    def build_training_frame(
        self,
        records_frame: pd.DataFrame,
        cutoff_year: int = _DEFAULT_TRAINING_CUTOFF_YEAR,
    ) -> pd.DataFrame:
        """Filter records_frame to a training-safe recent window.

        Pre-cutoff football has different scoring distributions (fewer
        matches per team, different competition structures, wartime gaps)
        that would quietly degrade model training if mixed in unfiltered --
        this is a deliberate, hard-coded scope boundary, not a knob to
        casually widen. `cutoff_year` defaults to 1990, called out in the
        project's own roadmap as "to be established empirically" -- treat as
        provisional, not settled.
        """
        return records_frame.loc[records_frame["match_date"].dt.year >= cutoff_year].reset_index(
            drop=True
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and merge the full-history international results dataset."
    )
    parser.add_argument(
        "--elo-source",
        default=None,
        help=(
            "Path to the existing Elo-enriched matches file to merge in "
            "(defaults to settings.inference_data_path if omitted)."
        ),
    )
    parser.add_argument(
        "--records-output",
        default="football_ai/data/processed/international_results_full_history.parquet",
        help="Output path for the full-history records frame.",
    )
    parser.add_argument(
        "--training-output",
        default="football_ai/data/processed/international_results_training_history.parquet",
        help="Output path for the training-safe (cutoff-filtered) frame.",
    )
    parser.add_argument(
        "--cutoff-year",
        type=int,
        default=_DEFAULT_TRAINING_CUTOFF_YEAR,
        help="Earliest match year to include in the training frame.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source = InternationalResultsSource()

    elo_path = arguments.elo_source
    if elo_path is None:
        from football_ai.config.settings import get_settings

        elo_path = get_settings().inference_data_path

    records_frame = source.build_records_frame(elo_source_path=elo_path)
    write_table(records_frame, arguments.records_output)
    print(
        f"records_frame: {len(records_frame)} matches, "
        f"{records_frame['match_date'].min().date()} -> {records_frame['match_date'].max().date()} "
        f"-> {arguments.records_output}"
    )

    training_frame = source.build_training_frame(records_frame, cutoff_year=arguments.cutoff_year)
    write_table(training_frame, arguments.training_output)
    print(
        f"training_frame (>= {arguments.cutoff_year}): {len(training_frame)} matches "
        f"-> {arguments.training_output}"
    )
