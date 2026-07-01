"""Tests for the prediction service."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
import tempfile
import unittest

try:
    from football_ai.models.registry import ModelRegistry
    from football_ai.predictions.schemas import MatchContextPayload
    from football_ai.predictions.service import PredictionService
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    ModelRegistry = None  # type: ignore[assignment]
    MatchContextPayload = None  # type: ignore[assignment]
    PredictionService = None  # type: ignore[assignment]


@unittest.skipIf(PredictionService is None, "Prediction dependencies are not installed.")
class PredictionServiceTestCase(unittest.TestCase):
    """Validate heuristic inference behavior."""

    def test_heuristic_prediction_returns_probabilities(self) -> None:
        """Fallback predictions should produce valid probability outputs."""
        with tempfile.TemporaryDirectory() as temp_dir:
            service = PredictionService(registry=ModelRegistry(Path(temp_dir) / "registry.json"))
            payload = MatchContextPayload(
                match_date=datetime(2026, 7, 1, tzinfo=UTC),
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


if __name__ == "__main__":
    unittest.main()
