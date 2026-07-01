"""Cleaning pipeline for multi-source football data."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name


REQUIRED_MATCH_COLUMNS = [
    "match_date",
    "home_team_name",
    "away_team_name",
]


@dataclass(slots=True)
class CleaningReport:
    """Summary of cleaning operations."""

    initial_rows: int
    final_rows: int
    removed_duplicates: int
    rows_with_missing_required_values: int


def normalize_team_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply canonical national team naming to team columns."""
    cleaned = frame.copy()
    for column in ("home_team_name", "away_team_name", "team_name", "opponent_name"):
        if column in cleaned.columns:
            cleaned[column] = cleaned[column].astype(str).map(canonicalize_team_name)
    return cleaned


def drop_duplicate_matches(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated matches using date and team pairs."""
    if not set(REQUIRED_MATCH_COLUMNS).issubset(frame.columns):
        return frame.copy()
    deduplicated = frame.sort_values("match_date").drop_duplicates(
        subset=REQUIRED_MATCH_COLUMNS,
        keep="last",
    )
    return deduplicated


def validate_required_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with missing values in required columns."""
    available_required = [column for column in REQUIRED_MATCH_COLUMNS if column in frame.columns]
    if not available_required:
        return frame.copy()
    return frame.dropna(subset=available_required)


def clean_matches(frame: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """Run the end-to-end match cleaning pipeline."""
    initial_rows = len(frame)
    cleaned = normalize_team_columns(frame)
    cleaned = validate_required_columns(cleaned)
    rows_after_required = len(cleaned)
    cleaned = drop_duplicate_matches(cleaned)
    report = CleaningReport(
        initial_rows=initial_rows,
        final_rows=len(cleaned),
        removed_duplicates=rows_after_required - len(cleaned),
        rows_with_missing_required_values=initial_rows - rows_after_required,
    )
    return cleaned.reset_index(drop=True), report
