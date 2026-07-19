"""Player enrichment for live lineup rows."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.data.normalization import normalize_text


DEFAULT_ENRICHMENT_COLUMNS = (
    "age",
    "market_value_eur",
    "national_team_caps",
    "national_team_minutes",
    "club_name",
    "league_name",
    "club_form_score",
    "injury_risk_score",
    "rest_days",
    "fatigue_index",
    "is_captain",
    "started_previous_match",
)


@dataclass(slots=True)
class LivePlayerEnrichmentResult:
    """Output of lineup enrichment."""

    frame: pd.DataFrame
    matched_rows: int
    unmatched_rows: int
    enrichment_columns: list[str]


def _prepare_lookup_key(frame: pd.DataFrame) -> pd.DataFrame:
    """Add normalized lookup keys used for fuzzy-safe joins."""
    prepared = frame.copy()
    prepared["team_name_key"] = prepared["team_name"].map(normalize_text)
    prepared["player_name_key"] = prepared["player_name"].map(normalize_text)
    return prepared


def enrich_lineups_with_player_data(
    lineups: pd.DataFrame,
    player_reference: pd.DataFrame,
    *,
    enrichment_columns: tuple[str, ...] = DEFAULT_ENRICHMENT_COLUMNS,
) -> LivePlayerEnrichmentResult:
    """Join live lineups with player-level metadata.

    The reference frame can come from a database export, manual CSV, or a
    future Transfermarkt/API integration. Matching prefers `player_external_id`
    when present and falls back to normalized player + team names.
    """
    if lineups.empty:
        return LivePlayerEnrichmentResult(
            frame=lineups.copy(),
            matched_rows=0,
            unmatched_rows=0,
            enrichment_columns=list(enrichment_columns),
        )

    required_lineup_columns = {"team_name", "player_name"}
    missing_lineup_columns = required_lineup_columns.difference(lineups.columns)
    if missing_lineup_columns:
        missing_list = ", ".join(sorted(missing_lineup_columns))
        raise ValueError(f"Lineups are missing required columns: {missing_list}")

    required_reference_columns = {"team_name", "player_name"}
    missing_reference_columns = required_reference_columns.difference(player_reference.columns)
    if missing_reference_columns:
        missing_list = ", ".join(sorted(missing_reference_columns))
        raise ValueError(f"Player reference is missing required columns: {missing_list}")

    lineup_frame = _prepare_lookup_key(lineups)
    reference_frame = _prepare_lookup_key(player_reference)

    reference_columns = [
        column
        for column in enrichment_columns
        if column in reference_frame.columns
    ]

    if "player_external_id" in lineup_frame.columns and "player_external_id" in reference_frame.columns:
        external_id_merge = lineup_frame.merge(
            reference_frame[
                ["player_external_id", "team_name_key", "player_name_key", *reference_columns]
            ].drop_duplicates(subset=["player_external_id"], keep="last"),
            on="player_external_id",
            how="left",
            suffixes=("", "_reference"),
        )
    else:
        external_id_merge = lineup_frame.copy()

    merged = external_id_merge.merge(
        reference_frame[
            ["team_name_key", "player_name_key", *reference_columns]
        ].drop_duplicates(subset=["team_name_key", "player_name_key"], keep="last"),
        on=["team_name_key", "player_name_key"],
        how="left",
        suffixes=("", "_fallback"),
    )

    for column in reference_columns:
        fallback_column = f"{column}_fallback"
        if column not in merged.columns and fallback_column in merged.columns:
            merged[column] = merged[fallback_column]
        elif fallback_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[fallback_column])
            merged = merged.drop(columns=[fallback_column])

    enrichment_presence = pd.Series(False, index=merged.index)
    for column in reference_columns:
        enrichment_presence = enrichment_presence | merged[column].notna()

    cleanup_columns = ["team_name_key", "player_name_key"]
    merged = merged.drop(columns=[column for column in cleanup_columns if column in merged.columns])
    return LivePlayerEnrichmentResult(
        frame=merged,
        matched_rows=int(enrichment_presence.sum()),
        unmatched_rows=int((~enrichment_presence).sum()),
        enrichment_columns=reference_columns,
    )
