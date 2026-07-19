"""Helpers for turning live lineup data into prediction payload updates."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.features.live_lineup_features import build_live_lineup_features
from football_ai.features.live_player_enrichment import (
    LivePlayerEnrichmentResult,
    enrich_lineups_with_player_data,
)


@dataclass(slots=True)
class LiveMatchContextResult:
    """Output of live lineup enrichment for prediction payloads."""

    enriched_lineups: pd.DataFrame
    payload_updates: dict[str, float]
    matched_rows: int
    unmatched_rows: int
    lineup_feature_frame: pd.DataFrame


def _build_payload_updates_from_lineup_features(lineup_feature_frame: pd.DataFrame) -> dict[str, float]:
    """Map lineup-derived features into the current prediction payload schema."""
    if lineup_feature_frame.empty:
        return {}
    row = lineup_feature_frame.iloc[0]
    return {
        "home_lineup_average_age": float(row.get("home_avg_age_starting_xi", 0.0)),
        "away_lineup_average_age": float(row.get("away_avg_age_starting_xi", 0.0)),
        "home_lineup_market_value": float(row.get("home_total_market_value_starting_xi", 0.0)),
        "away_lineup_market_value": float(row.get("away_total_market_value_starting_xi", 0.0)),
        "home_goalkeeper_form": float(row.get("home_goalkeeper_continuity", 0.0)),
        "away_goalkeeper_form": float(row.get("away_goalkeeper_continuity", 0.0)),
        "home_forward_form": float(row.get("home_club_form_mean_starting_xi", 0.0)),
        "away_forward_form": float(row.get("away_club_form_mean_starting_xi", 0.0)),
    }


def build_live_match_context(
    lineups: pd.DataFrame,
    player_reference: pd.DataFrame,
) -> LiveMatchContextResult:
    """Enrich lineups with player data and derive payload-compatible features."""
    enrichment_result: LivePlayerEnrichmentResult = enrich_lineups_with_player_data(
        lineups=lineups,
        player_reference=player_reference,
    )
    lineup_feature_result = build_live_lineup_features(enrichment_result.frame)
    payload_updates = _build_payload_updates_from_lineup_features(lineup_feature_result.frame)
    return LiveMatchContextResult(
        enriched_lineups=enrichment_result.frame,
        payload_updates=payload_updates,
        matched_rows=enrichment_result.matched_rows,
        unmatched_rows=enrichment_result.unmatched_rows,
        lineup_feature_frame=lineup_feature_result.frame,
    )
