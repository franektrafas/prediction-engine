"""Merge supplemental advanced-stat sources into a base wide matches frame."""

from __future__ import annotations

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name


_MERGE_KEY_COLUMNS = ("_merge_home_team", "_merge_away_team", "match_date")
_IDENTITY_COLUMNS = ("home_team_name", "away_team_name", "home_score", "away_score")


def merge_advanced_stats(base_matches: pd.DataFrame, supplemental_matches: pd.DataFrame) -> pd.DataFrame:
    """Left-merge a supplemental wide matches frame's stat columns onto a base frame.

    The base frame is always the source of truth for match identity (date,
    teams, score) -- supplemental columns only fill in where the base frame
    doesn't already have a non-null value for that column, so layering a
    richer future source on top can never silently override a value the base
    data already provides. Matching is on (canonicalized home team,
    canonicalized away team, match date) rather than raw team-name strings,
    since different sources spell team names differently (see
    `config.team_aliases`). Not specific to any one source -- reusable for
    any future supplemental frame (StatsBomb, FBref, ...) with the same wide
    `home_*`/`away_*` column convention `features.builder` expects.
    """
    if base_matches.empty or supplemental_matches.empty:
        return base_matches.copy()

    base = base_matches.copy()
    base["match_date"] = pd.to_datetime(base["match_date"], utc=True)
    base["_merge_home_team"] = base["home_team_name"].map(canonicalize_team_name)
    base["_merge_away_team"] = base["away_team_name"].map(canonicalize_team_name)

    supplemental = supplemental_matches.copy()
    supplemental["match_date"] = pd.to_datetime(supplemental["match_date"], utc=True)
    supplemental["_merge_home_team"] = supplemental["home_team_name"].map(canonicalize_team_name)
    supplemental["_merge_away_team"] = supplemental["away_team_name"].map(canonicalize_team_name)

    supplemental_stat_columns = [
        column
        for column in supplemental.columns
        if column not in _IDENTITY_COLUMNS and column not in _MERGE_KEY_COLUMNS
    ]
    supplemental_slice = supplemental[[*_MERGE_KEY_COLUMNS, *supplemental_stat_columns]].drop_duplicates(
        subset=list(_MERGE_KEY_COLUMNS), keep="last"
    )

    merged = base.merge(
        supplemental_slice,
        on=list(_MERGE_KEY_COLUMNS),
        how="left",
        suffixes=("", "_supplemental"),
    )
    for column in supplemental_stat_columns:
        supplemental_column = f"{column}_supplemental"
        if supplemental_column not in merged.columns:
            continue  # no name collision -- the supplemental column already landed under its own name
        if column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[supplemental_column])
        else:
            merged[column] = merged[supplemental_column]
        merged = merged.drop(columns=[supplemental_column])

    return merged.drop(columns=list(_MERGE_KEY_COLUMNS))
