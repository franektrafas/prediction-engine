"""Tests for the prediction service."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import tempfile
import unittest

try:
    import joblib
    import pandas as pd

    from football_ai.models.base import TrainedArtifact
    from football_ai.models.registry import ModelRegistry
    from football_ai.models.registry import RegisteredModel
    from football_ai.predictions.schemas import MatchContextPayload
    from football_ai.predictions.service import PredictionService
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    joblib = None  # type: ignore[assignment]
    pd = None  # type: ignore[assignment]
    TrainedArtifact = None  # type: ignore[assignment]
    ModelRegistry = None  # type: ignore[assignment]
    RegisteredModel = None  # type: ignore[assignment]
    MatchContextPayload = None  # type: ignore[assignment]
    PredictionService = None  # type: ignore[assignment]


class DummyClassifier:
    """Small estimator used to test registry-backed inference."""

    classes_ = ["away_win", "draw", "home_win"]

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Return a deterministic probability vector."""
        _ = features
        return [[0.2, 0.3, 0.5]]


class DummyNumericClassifier:
    """Classifier variant exposing numeric XGBoost-like classes."""

    classes_ = [0, 1, 2]

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Return a deterministic probability vector."""
        _ = features
        return [[0.25, 0.15, 0.60]]


class DummyFlatClassifier:
    """Classifier returning nearly uniform probabilities."""

    classes_ = [0, 1, 2]

    def predict_proba(self, features: pd.DataFrame) -> list[list[float]]:
        """Return a flat probability vector."""
        _ = features
        return [[0.33, 0.34, 0.33]]


@unittest.skipIf(PredictionService is None, "Prediction dependencies are not installed.")
class PredictionServiceTestCase(unittest.TestCase):
    """Validate heuristic inference behavior."""

    def test_heuristic_prediction_returns_probabilities(self) -> None:
        """Fallback predictions should produce valid probability outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = PredictionService(registry=ModelRegistry(Path(temp_dir) / "registry.json"))
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
                home_team_name="Spain",
                away_team_name="Portugal",
                home_elo=1900,
                away_elo=1850,
            )
            prediction = service.predict(payload)
            probability_sum = (
                prediction.outcome_probabilities.home_win
                + prediction.outcome_probabilities.draw
                + prediction.outcome_probabilities.away_win
            )
            self.assertAlmostEqual(probability_sum, 1.0, places=5)
            self.assertEqual(prediction.model_version, "heuristic_baseline_v0")

    def test_heuristic_prediction_surfaces_match_context_from_payload(self) -> None:
        """Live-only fields (injury impact, etc.) reach the response as context, not as model input.

        Confirmed at exactly zero importance in real ablation (2026-07-24):
        no model can learn from these (always zero in historical training
        data). They're excluded from feature_columns, but a caller-supplied
        value should still surface on the response so a UI/media consumer
        can present it alongside the prediction.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            service = PredictionService(registry=ModelRegistry(Path(temp_dir) / "registry.json"))
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
                home_team_name="Spain",
                away_team_name="Portugal",
                home_elo=1900,
                away_elo=1850,
                home_injured_market_value=42_000_000.0,
                away_injured_market_value=0.0,
            )
            prediction = service.predict(payload)
            self.assertEqual(prediction.match_context.home_injured_market_value, 42_000_000.0)
            self.assertEqual(prediction.match_context.away_injured_market_value, 0.0)
            self.assertIsNone(prediction.match_context.home_goalkeeper_form)

    def test_registry_backed_prediction_uses_registered_artifact(self) -> None:
        """A registered classifier artifact should drive API probabilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry = ModelRegistry(temp_path / "registry.json")
            artifact_path = temp_path / "match_outcome_xgboost_test_v1.joblib"
            artifact = TrainedArtifact(
                model_name="xgboost",
                target_name="match_outcome",
                task_type="multiclass",
                version="test_v1",
                feature_columns=["home_elo", "away_elo"],
                categorical_columns=[],
                hyperparameters={},
                estimator=DummyClassifier(),
                metrics={},
            )
            joblib.dump(artifact, artifact_path)
            registry.register(
                RegisteredModel(
                    model_name="xgboost",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="test_v1",
                    artifact_path=str(artifact_path),
                    metrics={},
                )
            )
            service = PredictionService(registry=registry)
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
                home_team_name="Spain",
                away_team_name="Portugal",
                home_elo=1900,
                away_elo=1850,
            )
            prediction = service.predict(payload)
            self.assertEqual(prediction.model_version, "test_v1")
            self.assertAlmostEqual(prediction.outcome_probabilities.home_win, 0.5, places=5)
            self.assertAlmostEqual(prediction.outcome_probabilities.draw, 0.3, places=5)
            self.assertAlmostEqual(prediction.outcome_probabilities.away_win, 0.2, places=5)

    def test_registry_backed_prediction_maps_numeric_classes(self) -> None:
        """Numeric multiclass labels should map to away/draw/home probabilities."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry = ModelRegistry(temp_path / "registry.json")
            artifact_path = temp_path / "match_outcome_xgboost_numeric_v1.joblib"
            artifact = TrainedArtifact(
                model_name="xgboost",
                target_name="match_outcome",
                task_type="multiclass",
                version="numeric_v1",
                feature_columns=["home_elo", "away_elo"],
                categorical_columns=[],
                hyperparameters={},
                estimator=DummyNumericClassifier(),
                metrics={},
            )
            joblib.dump(artifact, artifact_path)
            registry.register(
                RegisteredModel(
                    model_name="xgboost",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="numeric_v1",
                    artifact_path=str(artifact_path),
                    metrics={},
                )
            )
            service = PredictionService(registry=registry)
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
                home_team_name="Spain",
                away_team_name="Portugal",
                home_elo=1900,
                away_elo=1850,
            )
            prediction = service.predict(payload)
            self.assertEqual(prediction.model_version, "numeric_v1")
            self.assertAlmostEqual(prediction.outcome_probabilities.home_win, 0.60, places=5)
            self.assertAlmostEqual(prediction.outcome_probabilities.draw, 0.15, places=5)
            self.assertAlmostEqual(prediction.outcome_probabilities.away_win, 0.25, places=5)

    def test_flat_multiclass_prediction_falls_back_to_heuristic_blend(self) -> None:
        """Flat 1X2 probabilities should be replaced by a more informative fallback."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            registry = ModelRegistry(temp_path / "registry.json")
            artifact_path = temp_path / "match_outcome_xgboost_flat_v1.joblib"
            artifact = TrainedArtifact(
                model_name="xgboost",
                target_name="match_outcome",
                task_type="multiclass",
                version="flat_v1",
                feature_columns=["home_elo", "away_elo"],
                categorical_columns=[],
                hyperparameters={},
                estimator=DummyFlatClassifier(),
                metrics={},
            )
            joblib.dump(artifact, artifact_path)
            registry.register(
                RegisteredModel(
                    model_name="xgboost",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="flat_v1",
                    artifact_path=str(artifact_path),
                    metrics={},
                )
            )
            service = PredictionService(registry=registry)
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=timezone.utc),
                home_team_name="Spain",
                away_team_name="Portugal",
                home_elo=1900,
                away_elo=1850,
            )
            prediction = service.predict(payload)
            probability_sum = (
                prediction.outcome_probabilities.home_win
                + prediction.outcome_probabilities.draw
                + prediction.outcome_probabilities.away_win
            )
            self.assertAlmostEqual(probability_sum, 1.0, places=5)
            self.assertNotAlmostEqual(prediction.outcome_probabilities.home_win, 0.33, places=2)


if __name__ == "__main__":
    unittest.main()
