"""Prediction endpoint."""

from __future__ import annotations

from fastapi import APIRouter

from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse
from football_ai.predictions.service import PredictionService


router = APIRouter(tags=["predictions"])
prediction_service = PredictionService.create()


@router.post("/predict", response_model=PredictionResponse)
def predict_match(payload: MatchContextPayload) -> PredictionResponse:
    """Predict outcome and match statistics for a national team match."""
    return prediction_service.predict(payload)
