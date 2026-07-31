"""Tests for football_ai/api/routes/competitions.py.

Wikipedia is mocked here (offline, deterministic) -- the same convention
`test_uefa_nations_league_source.py` established; real-network verification
of this endpoint was done interactively, not as a committed automated test.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

try:
    from fastapi.testclient import TestClient

    from football_ai.api import routes as api_routes  # noqa: F401  (ensures package import works)
    from football_ai.api.main import app
    from football_ai.api.routes import competitions as competitions_module
    from football_ai.api.routes.auth import require_api_key
except ModuleNotFoundError:  # pragma: no cover - optional dependency guard
    TestClient = None  # type: ignore[assignment]
    app = None  # type: ignore[assignment]
    competitions_module = None  # type: ignore[assignment]
    require_api_key = None  # type: ignore[assignment]


# Real, 2026-07-30, from UefaNationsLeagueSource.fetch_group_matches("B", 4) -- Poland's actual group.
_REAL_UNPLAYED_GROUP = pd.DataFrame(
    [
        {
            "match_date": pd.Timestamp("2026-09-25"),
            "home_team_name": "Poland",
            "away_team_name": "Bosnia and Herzegovina",
            "home_score": pd.NA,
            "away_score": pd.NA,
            "stadium": "Stadion Narodowy, Warsaw",
        },
        {
            "match_date": pd.Timestamp("2026-09-25"),
            "home_team_name": "Sweden",
            "away_team_name": "Romania",
            "home_score": pd.NA,
            "away_score": pd.NA,
            "stadium": "Nationalarenan, Solna",
        },
    ]
)


@unittest.skipIf(TestClient is None or app is None, "FastAPI dependencies are not installed.")
class CompetitionOutlookTestCase(unittest.TestCase):
    """Auth bypassed via dependency override; Wikipedia fetch mocked -- targets the route logic only."""

    def setUp(self) -> None:
        app.dependency_overrides[require_api_key] = lambda: None

    def tearDown(self) -> None:
        app.dependency_overrides.pop(require_api_key, None)

    def test_outlook_returns_probabilities_summing_to_the_qualifying_spot_count(self) -> None:
        with mock.patch.object(
            competitions_module._uefa_source, "fetch_group_matches", return_value=_REAL_UNPLAYED_GROUP
        ):
            client = TestClient(app)
            response = client.get(
                "/v1/competitions/outlook",
                params={"division": "B", "group": 4, "n_simulations": 500, "n_qualifying_spots": 1},
            )
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(sorted(body["teams"]), ["Bosnia and Herzegovina", "Poland", "Romania", "Sweden"])
        self.assertAlmostEqual(sum(body["qualification_probabilities"].values()), 1.0, places=5)
        self.assertEqual(body["remaining_matches"], 2)

    def test_outlook_404s_when_the_source_returns_no_fixtures(self) -> None:
        with mock.patch.object(
            competitions_module._uefa_source, "fetch_group_matches", return_value=pd.DataFrame()
        ):
            client = TestClient(app)
            response = client.get("/v1/competitions/outlook", params={"division": "B", "group": 4})
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
