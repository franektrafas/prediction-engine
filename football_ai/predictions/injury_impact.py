"""Value-weighted injury impact for live prediction payloads."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from football_ai.config.settings import get_settings
from football_ai.data.io import read_table
from football_ai.data.normalization import normalize_text
from football_ai.predictions.player_reference_store import load_player_reference
from football_ai.predictions.schemas import MatchContextPayload


LOGGER = logging.getLogger(__name__)


def _load_injuries(path: str | Path) -> pd.DataFrame:
    """Load the injuries store, or an empty frame if it doesn't exist yet."""
    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    return read_table(file_path)


def compute_injured_market_value(
    team_name: str,
    injuries: pd.DataFrame,
    player_reference: pd.DataFrame,
) -> float:
    """Sum the market values of a team's currently-injured players.

    Matches `injuries` to `player_reference` by normalized team_name +
    player_name -- both stores come from the same `TransfermarktSource` with
    the same naming conventions, so no cross-source id reconciliation is
    needed (unlike matching api-football.com ids against Transfermarkt ids).
    An injured player absent from `player_reference` (not yet refreshed,
    name mismatch) contributes `0.0` rather than being dropped or erroring.
    """
    if injuries.empty or player_reference.empty:
        return 0.0
    team_key = normalize_text(team_name)
    team_injuries = injuries.loc[injuries["team_name"].map(normalize_text) == team_key]
    if team_injuries.empty:
        return 0.0

    reference = player_reference.copy()
    reference["team_name_key"] = reference["team_name"].map(normalize_text)
    reference["player_name_key"] = reference["player_name"].map(normalize_text)
    reference = reference.loc[reference["team_name_key"] == team_key]
    if reference.empty:
        return 0.0

    injured_keys = set(team_injuries["player_name"].map(normalize_text))
    matched = reference.loc[reference["player_name_key"].isin(injured_keys)]
    if matched.empty or "market_value_eur" not in matched.columns:
        return 0.0
    return float(pd.to_numeric(matched["market_value_eur"], errors="coerce").fillna(0.0).sum())


def enrich_payload_with_injury_impact(
    payload: MatchContextPayload,
    injuries: pd.DataFrame | None = None,
    player_reference: pd.DataFrame | None = None,
) -> MatchContextPayload:
    """Fill value-weighted injury-impact payload fields for both sides.

    Only fields the caller left unset (`None`) are overwritten -- any value
    already supplied explicitly on the payload always wins. Never raises:
    a missing/stale injuries store, a missing player reference, or any
    loader failure all degrade to returning `payload` unchanged, the same
    resilience contract as `live_prediction.enrich_payload_with_live_lineups`.
    Deliberately independent of `payload.fixture_id` -- this only needs the
    team names, which are always present, unlike lineup enrichment.
    """
    try:
        settings = get_settings()
        resolved_injuries = injuries if injuries is not None else _load_injuries(settings.injuries_path)
        resolved_reference = (
            player_reference
            if player_reference is not None
            else load_player_reference(settings.player_reference_path)
        )
        if resolved_injuries.empty:
            return payload
        home_value = compute_injured_market_value(
            payload.home_team_name, resolved_injuries, resolved_reference
        )
        away_value = compute_injured_market_value(
            payload.away_team_name, resolved_injuries, resolved_reference
        )
    except Exception:
        LOGGER.warning("Failed to compute injury impact", exc_info=True)
        return payload

    current = payload.model_dump()
    if current.get("home_injured_market_value") is None:
        current["home_injured_market_value"] = home_value
    if current.get("away_injured_market_value") is None:
        current["away_injured_market_value"] = away_value
    return MatchContextPayload(**current)
