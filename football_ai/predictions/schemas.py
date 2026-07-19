"""Prediction payload schemas."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("home_team_name", "away_team_name")
    @classmethod
    def validate_team_name(cls, value: str) -> str:
        """Reject empty team names."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("Team name cannot be empty.")
        return normalized


class ProbabilityPayload(BaseModel):
    """Three-way outcome probabilities."""

    home_win: float
    draw: float
    away_win: float

    @field_validator("home_win", "draw", "away_win")
    @classmethod
    def validate_probability_range(cls, value: float) -> float:
        """Ensure probabilities are bounded."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Probability must be between 0 and 1.")
        return value


class PredictionResponse(BaseModel):
    """Prediction response for the REST API."""

    model_version: str = Field(..., description="Resolved model version or heuristic version.")
    model_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Resolved model version per predicted target.",
    )
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

    @field_validator(
        "exact_score",
        "goals",
        "over_under",
        "corners",
        "cards",
        "shots",
        "shots_on_target",
        "xg",
        "possession",
    )
    @classmethod
    def validate_non_empty_mapping(cls, value: dict[str, Any]) -> dict[str, Any]:
        """Ensure response mappings are populated."""
        if not value:
            raise ValueError("Response mapping cannot be empty.")
        return value
