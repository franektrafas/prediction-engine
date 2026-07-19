"""Tests for the live Football API connector."""

from __future__ import annotations

import unittest

from football_ai.scrapers.sources.football_api import FootballApiSource


class FootballApiSourceTestCase(unittest.TestCase):
    """Validate Football API normalization."""

    def setUp(self) -> None:
        """Create a fresh source instance for each test."""
        self.source = FootballApiSource()

    def test_normalize_fixtures_maps_match_payload(self) -> None:
        """Fixture payloads should be converted into project-level match rows."""
        payload = {
            "response": [
                {
                    "fixture": {
                        "id": 12345,
                        "date": "2026-07-10T18:45:00+00:00",
                        "neutral": False,
                        "status": {"short": "NS"},
                        "venue": {"name": "National Stadium", "city": "Warsaw"},
                    },
                    "league": {
                        "id": 1,
                        "name": "UEFA Nations League",
                        "country": "Europe",
                        "season": 2026,
                        "round": "League A - Round 1",
                    },
                    "teams": {
                        "home": {"id": 10, "name": "Poland"},
                        "away": {"id": 20, "name": "Portugal"},
                    },
                    "goals": {"home": None, "away": None},
                }
            ]
        }
        frame = self.source.normalize_fixtures(payload)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["external_id"], "12345")
        self.assertEqual(frame.iloc[0]["home_team_name"], "Poland")
        self.assertEqual(frame.iloc[0]["away_team_name"], "Portugal")
        self.assertEqual(frame.iloc[0]["status"], "NS")
        self.assertEqual(frame.iloc[0]["stadium_name"], "National Stadium")

    def test_normalize_lineups_extracts_starters_and_substitutes(self) -> None:
        """Lineup payloads should preserve starter flags."""
        payload = {
            "response": [
                {
                    "team": {"id": 10, "name": "Poland"},
                    "startXI": [
                        {"player": {"id": 100, "name": "Player One", "number": 9, "pos": "F", "grid": "1:1"}}
                    ],
                    "substitutes": [
                        {"player": {"id": 101, "name": "Player Two", "number": 18, "pos": "M", "grid": None}}
                    ],
                }
            ]
        }
        frame = self.source.normalize_lineups(payload, fixture_id=54321)
        self.assertEqual(len(frame), 2)
        self.assertTrue(bool(frame.iloc[0]["started"]))
        self.assertFalse(bool(frame.iloc[1]["started"]))
        self.assertEqual(frame.iloc[0]["fixture_id"], 54321)


if __name__ == "__main__":
    unittest.main()
