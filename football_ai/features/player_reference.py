"""Helpers for maintaining a player reference dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.data.normalization import normalize_text


REFERENCE_BASE_COLUMNS = (
    "team_name",
    "player_name",
    "player_external_id",
    "team_external_id",
    "position",
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
    "last_updated_source",
)


@dataclass(slots=True)
class PlayerReferenceUpdateResult:
    """Result of merging new player observations into the reference table."""

    frame: pd.DataFrame
    added_players: int
    updated_players: int


def _ensure_reference_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Ensure the player reference exposes all expected columns."""
    reference = frame.copy()
    for column in REFERENCE_BASE_COLUMNS:
        if column not in reference.columns:
            reference[column] = pd.NA
    return reference


def _with_lookup_keys(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach stable lookup keys for merging players."""
    keyed = frame.copy()
    keyed["team_name_key"] = keyed["team_name"].map(normalize_text)
    keyed["player_name_key"] = keyed["player_name"].map(normalize_text)
    return keyed


def update_player_reference(
    existing_reference: pd.DataFrame | None,
    observed_players: pd.DataFrame,
    *,
    source_name: str,
) -> PlayerReferenceUpdateResult:
    """Merge observed player rows into a persistent reference dataset.

    New players are appended automatically, while existing rows are updated only
    for columns where the new observation provides a non-null value.
    """
    if observed_players.empty:
        empty_reference = _ensure_reference_columns(existing_reference or pd.DataFrame())
        return PlayerReferenceUpdateResult(
            frame=empty_reference,
            added_players=0,
            updated_players=0,
        )

    required_columns = {"team_name", "player_name"}
    missing_columns = required_columns.difference(observed_players.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Observed players are missing required columns: {missing_list}")

    existing = _ensure_reference_columns(
        existing_reference if existing_reference is not None else pd.DataFrame()
    )
    observed = _ensure_reference_columns(observed_players)

    observed["last_updated_source"] = source_name
    existing_keyed = _with_lookup_keys(existing)
    observed_keyed = _with_lookup_keys(observed)

    if "player_external_id" in observed_keyed.columns:
        observed_keyed["player_external_id"] = observed_keyed["player_external_id"].astype("object")
    if "player_external_id" in existing_keyed.columns:
        existing_keyed["player_external_id"] = existing_keyed["player_external_id"].astype("object")

    merged = existing_keyed.copy()
    added_players = 0
    updated_players = 0

    for observed_row in observed_keyed.to_dict(orient="records"):
        player_external_id = observed_row.get("player_external_id")
        if pd.notna(player_external_id) and player_external_id != "":
            match_mask = merged["player_external_id"].astype("object") == player_external_id
        else:
            match_mask = (
                (merged["team_name_key"] == observed_row["team_name_key"])
                & (merged["player_name_key"] == observed_row["player_name_key"])
            )

        if not bool(match_mask.any()):
            merged = pd.concat([merged, pd.DataFrame([observed_row])], ignore_index=True)
            added_players += 1
            continue

        match_index = merged.index[match_mask][0]
        changed = False
        for column, value in observed_row.items():
            if column in {"team_name_key", "player_name_key"}:
                continue
            if pd.notna(value) and value != "":
                if pd.isna(merged.at[match_index, column]) or merged.at[match_index, column] != value:
                    merged.at[match_index, column] = value
                    changed = True
        if changed:
            updated_players += 1

    cleaned = merged.drop(columns=["team_name_key", "player_name_key"], errors="ignore")
    cleaned = cleaned.drop_duplicates(
        subset=["player_external_id", "team_name", "player_name"],
        keep="last",
    ).reset_index(drop=True)
    return PlayerReferenceUpdateResult(
        frame=cleaned,
        added_players=added_players,
        updated_players=updated_players,
    )
