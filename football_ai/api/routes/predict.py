"""Prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks

from football_ai.predictions.live_prediction import enrich_payload_with_live_lineups
from football_ai.predictions.persistence import persist_prediction
from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse
from football_ai.predictions.service import PredictionService


router = APIRouter(tags=["predictions"])
prediction_service = PredictionService.create()


@router.post("/predict", response_model=PredictionResponse)
def predict_match(payload: MatchContextPayload, background_tasks: BackgroundTasks) -> PredictionResponse:
    """Predict outcome and match statistics for a national team match.

    When `payload.fixture_id` is set, lineup-derived fields are enriched from
    live api-football.com lineups before inference; any value already present
    on the payload is left untouched. The request/response is recorded to
    Postgres as a background task after the response is built, so a slow or
    unreachable database never adds latency to `/predict` -- persistence is
    always best-effort (see `predictions.persistence.persist_prediction`).
    """
    enriched_payload = enrich_payload_with_live_lineups(payload)
    response = prediction_service.predict(enriched_payload)
    background_tasks.add_task(persist_prediction, enriched_payload, response)
    return response
