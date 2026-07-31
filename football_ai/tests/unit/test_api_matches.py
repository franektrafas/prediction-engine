"""Tests for football_ai/api/routes/matches.py, including real-data integration."""

from __future__ import annotations

import os
import unittest
from pathlib import Path
from unittest import mock

try:
    from fastapi.testclient import TestClient

    from football_ai.api.main import app
    from football_ai.api.routes.auth import require_api_key
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    TestClient = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]
    require_api_key = None  # type: ignore[assignment]


_FULL_HISTORY_PATH = Path("football_ai/data/processed/international_results_full_history.parquet")


@unittest.skipIf(TestClient is None or app is None, "FastAPI dependencies are not installed.")
class MatchPreviewAuthTestCase(unittest.TestCase):
    """The /v1/matches/* surface must actually be gated by require_api_key."""

    def test_preview_without_a_configured_server_key_returns_503(self) -> None:
        with mock.patch.dict(os.environ, {"FOOTBALL_AI_API_KEY": ""}):
            client = TestClient(app)
            response = client.get(
                "/v1/matches/preview",
                params={"home": "Poland", "away": "Sweden", "match_date": "2026-11-17"},
            )
        self.assertEqual(response.status_code, 503)


@unittest.skipIf(TestClient is None or app is None, "FastAPI dependencies are not installed.")
class MatchPreviewTestCase(unittest.TestCase):
    """Auth bypassed via dependency override -- these tests target the route logic, not auth."""

    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_preview_returns_a_prediction_and_facts(self) -> None:
        client = TestClient(app)
        response = client.get(
            "/v1/matches/preview",
            params={"home": "Poland", "away": "Sweden", "match_date": "2026-11-17"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["home_team"], "Poland")
        self.assertEqual(body["away_team"], "Sweden")
        self.assertIn("goals", body["prediction"])
        self.assertIsInstance(body["facts"], list)
        for fact in body["facts"]:
            self.assertIn("sentence", fact)
            self.assertIn("provenance", fact)


@unittest.skipIf(TestClient is None or app is None, "FastAPI dependencies are not installed.")
@unittest.skipUnless(
    _FULL_HISTORY_PATH.exists(),
    "Needs the real full-history parquet -- run "
    "`python -m football_ai.scrapers.sources.international_results` first.",
)
class MatchRecapRealDataTestCase(unittest.TestCase):
    """Poland's real 800th international: 2018-06-12, a 4-0 win over Lithuania."""

    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_recap_returns_the_real_score_and_the_real_milestone_fact(self) -> None:
        client = TestClient(app)
        response = client.get(
            "/v1/matches/recap",
            params={"home": "Poland", "away": "Lithuania", "match_date": "2018-06-12"},
        )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["home_score"], 4)
        self.assertEqual(body["away_score"], 0)
        sentences = [fact["sentence"] for fact in body["facts"]]
        self.assertIn(
            "Polska rozegrała swój 800. mecz w historii reprezentacji, przeciwko Litwie.",
            sentences,
        )

    def test_recap_404s_for_a_match_not_in_the_real_record(self) -> None:
        client = TestClient(app)
        response = client.get(
            "/v1/matches/recap",
            params={"home": "Poland", "away": "Lithuania", "match_date": "1899-01-01"},
        )
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
