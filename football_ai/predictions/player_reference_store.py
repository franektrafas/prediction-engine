"""Persistence helpers for the player reference dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from football_ai.data.io import read_table, write_table
from football_ai.features.player_reference import PlayerReferenceUpdateResult, update_player_reference


def load_player_reference(path: str | Path) -> pd.DataFrame:
    """Load a player reference table or return an empty frame when absent."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return read_table(file_path)


def update_player_reference_store(
    path: str | Path,
    observed_players: pd.DataFrame,
    *,
    source_name: str,
) -> PlayerReferenceUpdateResult:
    """Load, update, and persist the player reference dataset."""
    file_path = Path(path)
    existing = load_player_reference(file_path)
    result = update_player_reference(
        existing_reference=existing,
        observed_players=observed_players,
        source_name=source_name,
    )
    write_table(result.frame, file_path)
    return result
