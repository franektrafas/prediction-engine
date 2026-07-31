"""Full-history international results importer (martj42/international_results).

CC0-licensed, ~49,500 men's full-international matches from 1872, actively
maintained (verified 2026-07-24). Fills the gap the project's own base file
has always had -- `international_results_modern_with_elo.parquet` only goes
back to 2000 -- and is the prerequisite for any historical record ("first
time since 1962") the planned `insights/` module needs.

Three output frames, deliberately not one:
- `records_frame`: full history, 1872-today. For facts/records only.
- `training_frame`: `records_frame` filtered to a recent cutoff. Pre-~1990
  football has different scoring distributions and would quietly degrade
  model training if fed in unfiltered -- never pass `records_frame` to
  `build_features`/`run_training` directly.
- `goalscorers_frame`: per-goal scorer/minute/own-goal/penalty events,
  linked to `records_frame`'s `match_id` -- feeds
  `insights.player_milestones` (goalscorers.csv starts 1916, later than
  results.csv's 1872, so it's necessarily a subset of `records_frame`).
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import argparse
import logging

import pandas as pd

from football_ai.data.enrichment import merge_advanced_stats
from football_ai.data.io import read_table, write_table
from football_ai.data.normalization import canonicalize_team_name
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

        `home_team_name`/`away_team_name` are canonicalized here, not left
        for a downstream caller -- CONFIRMED REAL BUG (found 2026-07-30
        auditing the engine): every `insights/` function (streaks.py,
        h2h.py, records.py, milestones.py) filters this frame by comparing
        against a `canonicalize_team_name()`-normalized query name. Left
        raw, any team whose canonical name differs from martj42's spelling
        (e.g. "Czech Republic" here vs. this project's canonical "Czechia")
        silently matched zero rows in every insights/ query -- no error,
        just empty facts -- reproduced for real: `team_matches(df,
        "Czechia")` returned 0 rows against 363 real "Czech Republic" rows
        before this fix.

        `match_date` is UTC-aware (`utc=True`), not naive -- CONFIRMED bug
        (found 2026-07-30 auditing the engine): `build_records_frame()`
        returns this frame as-is when `elo_source_path` is `None`/missing,
        but returns a UTC-aware frame (via `data.enrichment.merge_advanced_stats`,
        which itself calls `pd.to_datetime(..., utc=True)`) when Elo
        enrichment succeeds -- the exact same function's return dtype
        depended on an unrelated file's existence. `insights/`'s
        `as_of_date` filtering compares against `tz="UTC"` timestamps and
        raises `TypeError` on a naive frame. Setting `utc=True` here makes
        every `build_records_frame()` path consistent from the start.
        """
        normalized = pd.DataFrame(
            {
                "match_date": pd.to_datetime(raw["date"], utc=True),
                "home_team_name": raw["home_team"].map(canonicalize_team_name),
                "away_team_name": raw["away_team"].map(canonicalize_team_name),
                "home_score": raw["home_score"],
                "away_score": raw["away_score"],
                "tournament": raw["tournament"],
                "city": raw["city"],
                "country": raw["country"],
                "neutral_venue": raw["neutral"].astype(bool),
            }
        )
        normalized = normalized.sort_values("match_date").reset_index(drop=True)
        # Assigned here (chronological order, before any merge) rather than left
        # to a downstream caller -- `insights/` (Krok 2) needs a stable per-match
        # id for Fact provenance, and merge_advanced_stats' left-merge preserves
        # base-frame row order, so this survives build_records_frame() intact.
        # Stable across re-runs as long as martj42 only appends new matches at
        # the end and doesn't reorder history -- true in practice, not
        # contractually guaranteed by them.
        normalized.insert(0, "match_id", normalized.index + 1)
        return normalized

    def normalize_goalscorers(self, raw: pd.DataFrame, records_frame: pd.DataFrame) -> pd.DataFrame:
        """Map martj42's goalscorers.csv onto a match_id-linked, canonical-team-name schema.

        Joined to `records_frame`'s `match_id` via (match_date, canonical
        home, canonical away) -- verified for real, 2026-07-30: 100% of the
        real 47,914 raw goal events match a real `match_id` this way, no
        fuzzy-matching or manual reconciliation needed. Rows that somehow
        fail to match (not observed in practice, but `records_frame` isn't
        contractually guaranteed to be the exact same fetch) are dropped
        with a warning rather than silently kept with a null `match_id`,
        which would violate `insights.schemas.Fact`'s provenance contract
        the moment a caller tried to use one.

        `scorer` is kept as martj42's raw name string, not resolved through
        any alias table -- unlike team names, player names don't have this
        project's established `config.team_aliases` infrastructure, and a
        given player's name spelling has been observed consistent within
        this source (spot-checked for real: Robert Lewandowski appears
        under one exact spelling across all 69 of his real goal-scoring
        rows for Poland).
        """
        normalized = pd.DataFrame(
            {
                "match_date": pd.to_datetime(raw["date"], utc=True),
                "home_team_name": raw["home_team"].map(canonicalize_team_name),
                "away_team_name": raw["away_team"].map(canonicalize_team_name),
                "team_name": raw["team"].map(canonicalize_team_name),
                "scorer": raw["scorer"],
                "minute": raw["minute"],
                "own_goal": raw["own_goal"].astype(bool),
                "penalty": raw["penalty"].astype(bool),
            }
        )
        merged = normalized.merge(
            records_frame[["match_id", "match_date", "home_team_name", "away_team_name"]],
            on=["match_date", "home_team_name", "away_team_name"],
            how="left",
        )
        unmatched = int(merged["match_id"].isna().sum())
        if unmatched:
            LOGGER.warning(
                "%d of %d goalscorer rows did not match a real match_id in records_frame; dropping them.",
                unmatched,
                len(merged),
            )
        matched = merged.dropna(subset=["match_id"]).copy()
        matched["match_id"] = matched["match_id"].astype("int64")
        return matched.sort_values("match_date").reset_index(drop=True)

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
    parser.add_argument(
        "--goalscorers-output",
        default="football_ai/data/processed/international_results_goalscorers.parquet",
        help="Output path for the match_id-linked goalscorers frame.",
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

    goalscorers_frame = source.normalize_goalscorers(source.fetch_goalscorers(), records_frame)
    write_table(goalscorers_frame, arguments.goalscorers_output)
    print(f"goalscorers_frame: {len(goalscorers_frame)} goal events -> {arguments.goalscorers_output}")
