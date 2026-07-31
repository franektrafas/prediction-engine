"""Krok 5 of PLAN_SILNIKA.md: /v1/matches/{preview,recap}.

Matches are identified by team names + a date, not a database row --
matching the convention `predictions.persistence.persist_prediction`
already established for `/predict` (see its own docstring): this
product's real fixture source is Wikipedia/martj42 team-name-keyed data,
not a populated `matches` table, so a numeric row id has no natural
meaning here yet.
"""

from __future__ import annotations

from datetime import date as date_type

import pandas as pd
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from football_ai.api.routes.auth import require_api_key
from football_ai.api.schemas import FactPayload, MatchPreviewResponse, MatchRecapResponse
from football_ai.config.settings import get_settings
from football_ai.data.io import read_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.detectors import facts_for_match
from football_ai.insights.schemas import Fact
from football_ai.narrative.renderer import render_fact
from football_ai.predictions.persistence import persist_prediction
from football_ai.predictions.schemas import MatchContextPayload
from football_ai.predictions.service import PredictionService


router = APIRouter(prefix="/v1/matches", tags=["matches"], dependencies=[Depends(require_api_key)])

_prediction_service = PredictionService.create()

_RECORDS_FRAME_COLUMNS = ["match_id", "match_date", "home_team_name", "away_team_name", "home_score", "away_score", "tournament"]


def _load_records_frame() -> pd.DataFrame:
    """Load the full-history records frame once at import time, not per request.

    Degrades to an empty (but correctly-shaped) frame if the file hasn't
    been built yet in this environment -- `insights/` functions handle an
    empty frame gracefully (no matches found, not a crash), matching
    `PredictionService.create()`'s own "missing file -> None/empty, never
    raise at import time" contract.
    """
    path = get_settings().records_frame_path
    if not path.exists():
        return pd.DataFrame(columns=_RECORDS_FRAME_COLUMNS)
    return read_table(path)


_RECORDS_FRAME = _load_records_frame()


def _facts_payload(facts: list[Fact]) -> list[FactPayload]:
    """Render every Fact to its Polish sentence and pack it with its provenance."""
    return [
        FactPayload(
            fact_type=fact.fact_type,
            subject_team=fact.subject_team,
            sentence=render_fact(fact),
            weight=fact.weight,
            provenance=list(fact.provenance),
            related_teams=list(fact.related_teams),
        )
        for fact in facts
    ]


@router.get("/preview", response_model=MatchPreviewResponse)
def match_preview(
    home: str,
    away: str,
    match_date: date_type,
    background_tasks: BackgroundTasks,
    tournament: str = "UEFA Nations League",
) -> MatchPreviewResponse:
    """Pre-match bundle: real model prediction + up to 10 ranked, provenance-backed historical facts.

    The prediction request is persisted as a background task via the same
    `persist_prediction` `/predict` already uses -- this endpoint's
    prediction is a real, first-class served prediction, not a side effect.
    """
    payload = MatchContextPayload(
        match_date=match_date,
        home_team_name=home,
        away_team_name=away,
        tournament=tournament,
    )
    prediction = _prediction_service.predict(payload)
    background_tasks.add_task(persist_prediction, payload, prediction)

    as_of = pd.Timestamp(match_date, tz="UTC")
    facts = facts_for_match(_RECORDS_FRAME, home, away, as_of_date=as_of, top_n=10)

    return MatchPreviewResponse(
        home_team=canonicalize_team_name(home),
        away_team=canonicalize_team_name(away),
        match_date=match_date,
        tournament=tournament,
        prediction=prediction,
        facts=_facts_payload(facts),
    )


@router.get("/recap", response_model=MatchRecapResponse)
def match_recap(home: str, away: str, match_date: date_type) -> MatchRecapResponse:
    """Post-match bundle: the real final result + facts as of right after the match.

    404s when this exact (home, away, date) isn't in the real historical
    record -- a recap can only exist for a match that's actually been
    played and made it into martj42's data, never a guess.
    """
    canonical_home = canonicalize_team_name(home)
    canonical_away = canonicalize_team_name(away)
    match_rows = _RECORDS_FRAME.loc[
        (_RECORDS_FRAME["home_team_name"] == canonical_home)
        & (_RECORDS_FRAME["away_team_name"] == canonical_away)
        & (_RECORDS_FRAME["match_date"].dt.date == match_date)
    ]
    if match_rows.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No recorded match found for {home} vs {away} on {match_date}.",
        )
    result_row = match_rows.iloc[0]

    # Exclusive as_of_date (see insights/verification.py's same +1s pattern) --
    # this match's own date can't satisfy a strict `<` cutoff against itself,
    # so any streak/milestone/H2H fact this match itself created would
    # otherwise be silently excluded from its own recap.
    as_of = pd.Timestamp(match_date, tz="UTC") + pd.Timedelta(seconds=1)
    facts = facts_for_match(_RECORDS_FRAME, home, away, as_of_date=as_of, top_n=10)

    return MatchRecapResponse(
        home_team=canonical_home,
        away_team=canonical_away,
        match_date=match_date,
        tournament=str(result_row.get("tournament") or "") or None,
        home_score=int(result_row["home_score"]),
        away_score=int(result_row["away_score"]),
        facts=_facts_payload(facts),
    )
