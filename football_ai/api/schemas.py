"""Response schemas for the Krok 5 /v1 route surface."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel

from football_ai.predictions.schemas import PredictionResponse


class FactPayload(BaseModel):
    """JSON-serializable view of an `insights.Fact`, with its rendered Polish sentence attached.

    `provenance` is the real `match_id` list the fact was derived from --
    surfaced so a caller (or a journalist double-checking a claim) can trace
    every sentence back to the exact matches that produced it, per
    PLAN_SILNIKA.md's non-negotiable provenance requirement.
    """

    fact_type: str
    subject_team: str
    sentence: str
    weight: float
    provenance: list[int]
    related_teams: list[str] = []


class MatchPreviewResponse(BaseModel):
    """`GET /v1/matches/preview` -- pre-match facts + prediction, both reproducible from real data."""

    home_team: str
    away_team: str
    match_date: date
    tournament: str | None
    prediction: PredictionResponse
    facts: list[FactPayload]


class MatchRecapResponse(BaseModel):
    """`GET /v1/matches/recap` -- the real final result + facts as of right after the match."""

    home_team: str
    away_team: str
    match_date: date
    tournament: str | None
    home_score: int
    away_score: int
    facts: list[FactPayload]


class GroupStandingPayload(BaseModel):
    """One team's row in a group table, computed from real played results only."""

    team: str
    points: int
    goal_difference: int
    goals_for: int
    goals_against: int
    wins: int
    draws: int
    losses: int


class CompetitionOutlookResponse(BaseModel):
    """`GET /v1/competitions/outlook` -- Monte Carlo qualification odds for one group."""

    division: str
    group_number: int
    season: str
    teams: list[str]
    standings_so_far: list[GroupStandingPayload]
    qualification_probabilities: dict[str, float]
    n_qualifying_spots: int
    n_simulations: int
    remaining_matches: int
