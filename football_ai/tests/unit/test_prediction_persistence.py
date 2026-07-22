"""Tests for best-effort prediction persistence to Postgres."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

try:
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import sessionmaker

    from football_ai.database.base import Base
    from football_ai.database.models import ModelVersion, Prediction
    from football_ai.predictions.persistence import persist_prediction
    from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse, ProbabilityPayload
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    create_engine = None  # type: ignore[assignment]
    select = None  # type: ignore[assignment]
    sessionmaker = None  # type: ignore[assignment]
    Base = None  # type: ignore[assignment]
    ModelVersion = None  # type: ignore[assignment]
    Prediction = None  # type: ignore[assignment]
    persist_prediction = None  # type: ignore[assignment]
    MatchContextPayload = None  # type: ignore[assignment]
    PredictionResponse = None  # type: ignore[assignment]
    ProbabilityPayload = None  # type: ignore[assignment]


def _payload(**overrides: object) -> "MatchContextPayload":
    """Build a minimal payload, applying any field overrides."""
    fields = {
        "match_date": datetime(2026, 7, 1, tzinfo=timezone.utc),
        "home_team_name": "Spain",
        "away_team_name": "Portugal",
    }
    fields.update(overrides)
    return MatchContextPayload(**fields)


def _response(**overrides: object) -> "PredictionResponse":
    """Build a full prediction response, applying any field overrides."""
    fields = {
        "model_version": "heuristic_baseline_v0",
        "model_versions": {"match_outcome": "heuristic_baseline_v0"},
        "outcome_probabilities": ProbabilityPayload(home_win=0.5, draw=0.3, away_win=0.2),
        "exact_score": {"home": 2, "away": 1},
        "goals": {"home": 1.8, "away": 1.1},
        "btts": 0.55,
        "over_under": {"over_1_5": 0.7, "over_2_5": 0.5, "over_3_5": 0.3},
        "corners": {"home": 5.2, "away": 4.5},
        "cards": {"home": 1.8, "away": 2.0},
        "shots": {"home": 13.0, "away": 10.2},
        "shots_on_target": {"home": 4.7, "away": 3.6},
        "xg": {"home": 1.9, "away": 1.2},
        "possession": {"home": 53.0, "away": 47.0},
    }
    fields.update(overrides)
    return PredictionResponse(**fields)


def _sqlite_session_factory():
    """Build a sessionmaker backed by a fresh in-memory SQLite database."""
    engine = create_engine("sqlite:///:memory:", future=True)
    Base.metadata.create_all(bind=engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@unittest.skipIf(persist_prediction is None, "Database dependencies are not installed.")
class PredictionPersistenceTestCase(unittest.TestCase):
    """Validate best-effort prediction persistence."""

    def test_persists_heuristic_response_with_heuristic_model_version(self) -> None:
        """A heuristic-only response should link to a 'heuristic' model version row."""
        session_factory = _sqlite_session_factory()
        result = persist_prediction(_payload(), _response(), session_factory=session_factory)
        self.assertIsNotNone(result)

        with session_factory() as session:
            model_versions = session.execute(select(ModelVersion)).scalars().all()
            predictions = session.execute(select(Prediction)).scalars().all()
        self.assertEqual(len(model_versions), 1)
        self.assertEqual(model_versions[0].model_name, "heuristic")
        self.assertEqual(model_versions[0].version, "heuristic_baseline_v0")
        self.assertEqual(len(predictions), 1)
        self.assertEqual(predictions[0].model_version_id, model_versions[0].id)
        self.assertIsNone(predictions[0].match_id)
        self.assertAlmostEqual(predictions[0].outcome_probabilities["home_win"], 0.5)

    def test_persists_real_model_response_with_selected_model_version(self) -> None:
        """A registry-backed response should link to a 'selected_model' model version row."""
        session_factory = _sqlite_session_factory()
        response = _response(model_version="v1", model_versions={"match_outcome": "v1", "btts": "v1"})
        persist_prediction(_payload(), response, session_factory=session_factory)

        with session_factory() as session:
            model_versions = session.execute(select(ModelVersion)).scalars().all()
        self.assertEqual(len(model_versions), 1)
        self.assertEqual(model_versions[0].model_name, "selected_model")
        self.assertEqual(model_versions[0].version, "v1")

    def test_reuses_existing_model_version_row_across_calls(self) -> None:
        """Repeated predictions from the same model version shouldn't duplicate the row."""
        session_factory = _sqlite_session_factory()
        persist_prediction(_payload(), _response(), session_factory=session_factory)
        persist_prediction(_payload(home_team_name="France"), _response(), session_factory=session_factory)

        with session_factory() as session:
            model_versions = session.execute(select(ModelVersion)).scalars().all()
            predictions = session.execute(select(Prediction)).scalars().all()
        self.assertEqual(len(model_versions), 1)
        self.assertEqual(len(predictions), 2)

    def test_returns_none_and_does_not_raise_when_session_factory_fails(self) -> None:
        """A DB outage must degrade gracefully instead of breaking the caller."""

        def broken_session_factory():
            raise RuntimeError("no database configured")

        result = persist_prediction(_payload(), _response(), session_factory=broken_session_factory)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
