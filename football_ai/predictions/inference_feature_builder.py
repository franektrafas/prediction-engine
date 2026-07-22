"""Feature preparation helpers for prediction-time inference."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_ai.features.builder import build_features
from football_ai.predictions.schemas import MatchContextPayload


_INFERENCE_RATIO_EXCLUDE_TOKENS = (
    "xg_",
    "xga_",
    "shots_",
    "shots_on_target_",
    "corners_",
    "cards_",
    "possession_",
    "pressing_intensity_",
)


@dataclass(slots=True)
class InferenceFeatureBuildResult:
    """Single-row feature output used by prediction-time models."""

    frame: pd.DataFrame
    dropped_columns: list[str]


def payload_to_match_row(payload: MatchContextPayload) -> dict[str, object]:
    """Convert API payload into a match-like row compatible with feature generation."""
    home_elo = payload.home_elo or 1500.0
    away_elo = payload.away_elo or 1500.0
    home_recent_xg = payload.home_recent_xg or 1.5
    away_recent_xg = payload.away_recent_xg or 1.2
    home_recent_xga = payload.home_recent_xga or 1.1
    away_recent_xga = payload.away_recent_xga or 1.3
    return {
        "match_date": pd.to_datetime(payload.match_date, utc=True),
        "home_team_name": payload.home_team_name,
        "away_team_name": payload.away_team_name,
        "home_score": np.nan,
        "away_score": np.nan,
        "tournament": payload.tournament or "Unknown",
        "stadium_name": payload.stadium_name or "Unknown",
        "country": payload.country or "Unknown",
        "neutral_venue": bool(payload.neutral_venue),
        "home_elo": home_elo,
        "away_elo": away_elo,
        "home_elo_rank": np.nan,
        "away_elo_rank": np.nan,
        "elo_diff": home_elo - away_elo,
        "home_xg": home_recent_xg,
        "away_xg": away_recent_xg,
        "home_shots": np.nan,
        "away_shots": np.nan,
        "home_shots_on_target": np.nan,
        "away_shots_on_target": np.nan,
        "home_corners": np.nan,
        "away_corners": np.nan,
        "home_yellow_cards": np.nan,
        "away_yellow_cards": np.nan,
        "home_red_cards": 0.0,
        "away_red_cards": 0.0,
        "home_possession": np.nan,
        "away_possession": np.nan,
        "home_pressing_intensity": np.nan,
        "away_pressing_intensity": np.nan,
        "home_lineup_average_age": payload.home_lineup_average_age or 27.0,
        "away_lineup_average_age": payload.away_lineup_average_age or 27.0,
        "home_lineup_market_value": payload.home_lineup_market_value or 100000000.0,
        "away_lineup_market_value": payload.away_lineup_market_value or 100000000.0,
        "home_goalkeeper_form": payload.home_goalkeeper_form or 0.0,
        "away_goalkeeper_form": payload.away_goalkeeper_form or 0.0,
        "home_forward_form": payload.home_forward_form or 0.0,
        "away_forward_form": payload.away_forward_form or 0.0,
        "home_travel_distance_km": payload.home_travel_distance_km or 0.0,
        "away_travel_distance_km": payload.away_travel_distance_km or 0.0,
        "home_injured_market_value": payload.home_injured_market_value or 0.0,
        "away_injured_market_value": payload.away_injured_market_value or 0.0,
        "home_recent_xga": home_recent_xga,
        "away_recent_xga": away_recent_xga,
        "weather_temperature_c": payload.weather_temperature_c or 18.0,
        "weather_precipitation_mm": payload.weather_precipitation_mm or 0.0,
    }


def _should_drop_inference_column(column: str) -> bool:
    """Return whether a feature column is too sparse for live inference."""
    if "_ratio" not in column:
        return False
    return any(token in column for token in _INFERENCE_RATIO_EXCLUDE_TOKENS)


def build_inference_feature_frame(
    payload: MatchContextPayload,
    historical_matches: pd.DataFrame | None,
) -> InferenceFeatureBuildResult:
    """Build a single-row inference frame from history and current match context."""
    upcoming_match = pd.DataFrame([payload_to_match_row(payload)])
    if historical_matches is None or historical_matches.empty:
        return InferenceFeatureBuildResult(frame=upcoming_match, dropped_columns=[])

    history = historical_matches.copy()
    if "match_date" in history.columns:
        history["match_date"] = pd.to_datetime(history["match_date"], utc=True)
    payload_match_date = pd.to_datetime(payload.match_date, utc=True)
    history = history.loc[history["match_date"] < payload_match_date].copy()
    combined = pd.concat([history, upcoming_match], ignore_index=True)
    combined["match_date"] = pd.to_datetime(combined["match_date"], utc=True)
    combined["match_id"] = np.arange(1, len(combined) + 1)
    feature_result = build_features(combined)
    target_match_id = int(combined["match_id"].iloc[-1])
    feature_frame = feature_result.frame.loc[feature_result.frame["match_id"] == target_match_id].copy()
    if feature_frame.empty:
        return InferenceFeatureBuildResult(frame=upcoming_match, dropped_columns=[])
    dropped_columns = [
        column for column in feature_frame.columns
        if _should_drop_inference_column(column)
    ]
    cleaned_frame = feature_frame.drop(columns=dropped_columns, errors="ignore").reset_index(drop=True)
    return InferenceFeatureBuildResult(frame=cleaned_frame, dropped_columns=dropped_columns)
