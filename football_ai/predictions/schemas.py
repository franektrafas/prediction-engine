"""Prediction payload schemas."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class MatchContextPayload(BaseModel):
    """Incoming match context for predictions."""

    match_date: datetime
    home_team_name: str
    away_team_name: str
    tournament: str | None = None
    stadium_name: str | None = None
    country: str | None = None
    neutral_venue: bool = False
    home_elo: float | None = None
    away_elo: float | None = None
    home_lineup_average_age: float | None = None
    away_lineup_average_age: float | None = None
    home_lineup_market_value: float | None = None
    away_lineup_market_value: float | None = None
    home_goalkeeper_form: float | None = None
    away_goalkeeper_form: float | None = None
    home_forward_form: float | None = None
    away_forward_form: float | None = None
    home_travel_distance_km: float | None = None
    away_travel_distance_km: float | None = None
    home_recent_xg: float | None = None
    away_recent_xg: float | None = None
    home_recent_xga: float | None = None
    away_recent_xga: float | None = None
    weather_temperature_c: float | None = None
    weather_precipitation_mm: float | None = None


class ProbabilityPayload(BaseModel):
    """Three-way outcome probabilities."""

    home_win: float
    draw: float
    away_win: float


class PredictionResponse(BaseModel):
    """Prediction response for the REST API."""

    model_version: str = Field(..., description="Resolved model version or heuristic version.")
    outcome_probabilities: ProbabilityPayload
    exact_score: dict[str, int]
    goals: dict[str, float]
    btts: float
    over_under: dict[str, float]
    corners: dict[str, float]
    cards: dict[str, float]
    shots: dict[str, float]
    shots_on_target: dict[str, float]
    xg: dict[str, float]
    possession: dict[str, float]
