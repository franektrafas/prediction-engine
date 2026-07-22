"""Best-effort persistence of served predictions to Postgres."""

from __future__ import annotations

import logging
from typing import Callable

from sqlalchemy.orm import Session

from football_ai.database.models import Prediction
from football_ai.database.repositories import ModelVersionRepository, PredictionRepository
from football_ai.database.session import SessionLocal
from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse


LOGGER = logging.getLogger(__name__)

_HEURISTIC_VERSION = "heuristic_baseline_v0"
_PRIMARY_TARGET = "match_outcome"


def persist_prediction(
    payload: MatchContextPayload,
    response: PredictionResponse,
    session_factory: Callable[[], Session] = SessionLocal,
) -> Prediction | None:
    """Record a served prediction request/response for audit and analytics.

    Never raises: a DB outage, missing credentials, or schema drift must not
    break `/predict` -- the same resilience contract as
    `predictions.live_prediction.enrich_payload_with_live_lineups`. Returns
    the persisted row on success, or `None` when persistence was skipped.

    `Prediction.model_version_id` is a single required FK, but one response
    is actually composed from up to 18 separately-registered per-target
    models (`response.model_versions`); this links to the `match_outcome`
    model specifically as the "primary" version, while the full per-target
    breakdown is preserved in `raw_prediction`. `Prediction.match_id` is left
    `NULL` -- `/predict` requests identify a match by team names + date, not
    a `matches` row, and that table is a separate, mostly-unpopulated data
    path (see the roadmap plan's Context).
    """
    try:
        with session_factory() as session:
            is_heuristic = response.model_version == _HEURISTIC_VERSION
            model_version = ModelVersionRepository(session).get_or_create(
                version=response.model_versions.get(_PRIMARY_TARGET, response.model_version),
                model_name="heuristic" if is_heuristic else "selected_model",
                target_name=_PRIMARY_TARGET,
                task_type="multiclass",
            )
            prediction = Prediction(
                match_id=None,
                model_version_id=model_version.id,
                request_payload=payload.model_dump(mode="json"),
                outcome_probabilities=response.outcome_probabilities.model_dump(),
                predicted_scores={"exact_score": response.exact_score, "goals": response.goals},
                expected_statistics={
                    "corners": response.corners,
                    "cards": response.cards,
                    "shots": response.shots,
                    "shots_on_target": response.shots_on_target,
                    "xg": response.xg,
                    "possession": response.possession,
                    "btts": response.btts,
                    "over_under": response.over_under,
                },
                raw_prediction=response.model_dump(mode="json"),
            )
            PredictionRepository(session).save(prediction)
            session.commit()
            return prediction
    except Exception:
        LOGGER.warning("Failed to persist prediction", exc_info=True)
        return None
