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
    fixture_id: int | None = Field(
        default=None,
        description=(
            "Optional api-football.com fixture id. When set, /predict enriches "
            "lineup-derived fields (form, market value, age, ...) from live "
            "lineups before running inference, without overriding any value "
            "already supplied explicitly."
        ),
    )
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
    home_injured_market_value: float | None = None
    away_injured_market_value: float | None = None
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


class MatchContext(BaseModel):
    """Live, non-learned context surfaced alongside a prediction.

    These fields (injuries, squad market value, goalkeeper/forward form) are
    always zero-filled in historical training data -- there's no point-in-time
    historical record for them, only current snapshots -- so no trained model
    can actually learn a relationship from them (confirmed via feature
    ablation, 2026-07-24: exactly zero importance). They're excluded from
    `feature_columns` for that reason. Surfaced here instead as informational
    context a caller (UI, explainability, media use) can present alongside
    the prediction -- e.g. "model favors the home team, but note they're
    missing EUR50m of injured value" -- without implying the model itself
    weighed it.
    """

    home_injured_market_value: float | None = None
    away_injured_market_value: float | None = None
    home_lineup_market_value: float | None = None
    away_lineup_market_value: float | None = None
    home_goalkeeper_form: float | None = None
    away_goalkeeper_form: float | None = None
    home_forward_form: float | None = None
    away_forward_form: float | None = None


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
    # None (not a flat-default dict) when no trained model is registered for
    # this target -- e.g. corners/cards/shots/shots_on_target/possession
    # were withdrawn 2026-08-20 (ESPN-sourced training data confirmed
    # FORBIDDEN under Disney's Terms of Use, see
    # database/import_records.py). A constant number that never varies by
    # team or match reads as a real prediction to an API consumer; omitting
    # the field is the honest signal that there isn't one.
    corners: dict[str, float] | None = None
    cards: dict[str, float] | None = None
    shots: dict[str, float] | None = None
    shots_on_target: dict[str, float] | None = None
    xg: dict[str, float]
    possession: dict[str, float] | None = None
    match_context: MatchContext = Field(default_factory=MatchContext)

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
    def validate_non_empty_mapping(
        cls, value: dict[str, Any] | None
    ) -> dict[str, Any] | None:
        """Ensure a provided response mapping is populated -- None itself is allowed."""
        if value is not None and not value:
            raise ValueError("Response mapping cannot be empty.")
        return value
