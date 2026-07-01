"""Integration tests for the FastAPI app."""

from __future__ import annotations

import unittest

try:
    from fastapi.testclient import TestClient
    from football_ai.api.main import app
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    TestClient = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]


@unittest.skipIf(TestClient is None or app is None, "FastAPI dependencies are not installed.")
class ApiIntegrationTestCase(unittest.TestCase):
    """Validate the REST API contract."""

    def test_predict_endpoint_returns_prediction_payload(self) -> None:
        """The predict endpoint should return a full prediction response."""
        client = TestClient(app)
        response = client.post(
            "/predict",
            json={
                "match_date": "2026-07-01T18:00:00Z",
                "home_team_name": "Spain",
                "away_team_name": "Portugal",
                "home_elo": 1900,
                "away_elo": 1850
            },
        )
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertIn("outcome_probabilities", payload)
        self.assertIn("model_version", payload)


if __name__ == "__main__":
    unittest.main()
