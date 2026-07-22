"""Prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from football_ai.predictions.live_prediction import enrich_payload_with_live_lineups
from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse
from football_ai.predictions.service import PredictionService


router = APIRouter(tags=["predictions"])
prediction_service = PredictionService.create()


@router.post("/predict", response_model=PredictionResponse)
def predict_match(payload: MatchContextPayload) -> PredictionResponse:
    """Predict outcome and match statistics for a national team match.

    When `payload.fixture_id` is set, lineup-derived fields are enriched from
    live api-football.com lineups before inference; any value already present
    on the payload is left untouched.
    """
    enriched_payload = enrich_payload_with_live_lineups(payload)
    return prediction_service.predict(enriched_payload)
