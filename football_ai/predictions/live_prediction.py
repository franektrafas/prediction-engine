"""Wire live api-football.com lineups into prediction payloads."""

from __future__ import annotations

import logging
from typing import Protocol

import pandas as pd

from football_ai.config.settings import get_settings
from football_ai.data.normalization import normalize_text
from football_ai.predictions.live_match_context import build_live_match_context
from football_ai.predictions.player_reference_store import load_player_reference
from football_ai.predictions.schemas import MatchContextPayload


LOGGER = logging.getLogger(__name__)


class LineupSource(Protocol):
    """Anything that can fetch lineups for a fixture id, e.g. `FootballApiSource`."""

    def fetch_fixture_lineups(self, fixture_id: int) -> pd.DataFrame:
        """Fetch lineup rows for a fixture."""
        ...


def _swap_home_away(updates: dict[str, float]) -> dict[str, float]:
    """Swap `home_`/`away_` keys when the fetched lineup order is reversed."""
    swapped: dict[str, float] = {}
    for key, value in updates.items():
        if key.startswith("home_"):
            swapped[f"away_{key[len('home_'):]}"] = value
        elif key.startswith("away_"):
            swapped[f"home_{key[len('away_'):]}"] = value
        else:
            swapped[key] = value
    return swapped


def enrich_payload_with_live_lineups(
    payload: MatchContextPayload,
    football_api_source: LineupSource | None = None,
    player_reference: pd.DataFrame | None = None,
) -> MatchContextPayload:
    """Fill lineup-derived payload fields from a fixture's live lineups.

    Only fields the caller left unset (`None`) are overwritten -- any value
    already supplied explicitly on the payload always wins. Every failure mode
    (no `fixture_id`, no API key, network/API error, no reference match, team
    names that don't line up) falls back to returning `payload` unchanged, so
    `/predict` never fails a request because live enrichment didn't work.
    """
    if payload.fixture_id is None:
        return payload

    if football_api_source is None:
        from football_ai.scrapers.sources.football_api import FootballApiSource

        football_api_source = FootballApiSource()

    try:
        lineups = football_api_source.fetch_fixture_lineups(payload.fixture_id)
    except Exception:
        LOGGER.warning(
            "Failed to fetch live lineups for fixture %s", payload.fixture_id, exc_info=True
        )
        return payload
    if lineups is None or lineups.empty:
        return payload

    reference = (
        player_reference
        if player_reference is not None
        else load_player_reference(get_settings().player_reference_path)
    )

    try:
        context = build_live_match_context(lineups, reference)
    except Exception:
        LOGGER.warning(
            "Failed to build live match context for fixture %s", payload.fixture_id, exc_info=True
        )
        return payload

    if context.matched_rows == 0 or context.lineup_feature_frame.empty or not context.payload_updates:
        return payload

    frame = context.lineup_feature_frame
    lineup_home = normalize_text(frame["home_team_name"].iloc[0])
    lineup_away = normalize_text(frame["away_team_name"].iloc[0])
    payload_home = normalize_text(payload.home_team_name)
    payload_away = normalize_text(payload.away_team_name)

    updates = dict(context.payload_updates)
    if lineup_home == payload_away and lineup_away == payload_home:
        updates = _swap_home_away(updates)
    elif not (lineup_home == payload_home and lineup_away == payload_away):
        LOGGER.warning(
            "Live lineup teams (%s vs %s) do not match payload teams (%s vs %s) for "
            "fixture %s; skipping live enrichment.",
            frame["home_team_name"].iloc[0],
            frame["away_team_name"].iloc[0],
            payload.home_team_name,
            payload.away_team_name,
            payload.fixture_id,
        )
        return payload

    current = payload.model_dump()
    for field_name, value in updates.items():
        if field_name in current and current[field_name] is None:
            current[field_name] = value
    return MatchContextPayload(**current)
