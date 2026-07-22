"""Tests for the live Football API connector."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

import pandas as pd

from football_ai.scrapers.sources.football_api import FootballApiSource


def _squad_frame(team_external_id: int, team_name: str, players: list[tuple[str, int]]) -> pd.DataFrame:
    """Build a squad DataFrame matching `normalize_team_squad`'s output shape."""
    return pd.DataFrame(
        [
            {
                "team_external_id": team_external_id,
                "team_name": team_name,
                "player_external_id": external_id,
                "player_name": player_name,
                "age": 25.0,
                "position": "Midfielder",
                "number": index + 1,
            }
            for index, (player_name, external_id) in enumerate(players)
        ]
    )


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

    def test_refresh_player_reference_merges_multiple_team_squads(self) -> None:
        """Squads from every requested team should land in one merged store."""
        squads = {
            10: _squad_frame(10, "Poland", [("Player One", 1), ("Player Two", 2)]),
            20: _squad_frame(20, "Portugal", [("Player Three", 3)]),
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "player_reference.csv"
            with mock.patch.object(self.source, "fetch_team_squad", side_effect=lambda team_id: squads[team_id]):
                result = self.source.refresh_player_reference([10, 20], player_reference_path=reference_path)
        self.assertEqual(result.added_players, 3)
        self.assertEqual(len(result.frame), 3)
        self.assertEqual(
            set(result.frame["player_name"]),
            {"Player One", "Player Two", "Player Three"},
        )

    def test_refresh_player_reference_skips_a_failing_team(self) -> None:
        """One team's fetch failing should not stop the others from being stored."""
        good_squad = _squad_frame(10, "Poland", [("Player One", 1)])

        def fetch_side_effect(team_id: int) -> pd.DataFrame:
            if team_id == 20:
                raise RuntimeError("rate limited")
            return good_squad

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "player_reference.csv"
            with mock.patch.object(self.source, "fetch_team_squad", side_effect=fetch_side_effect):
                result = self.source.refresh_player_reference([10, 20], player_reference_path=reference_path)
        self.assertEqual(result.added_players, 1)
        self.assertEqual(list(result.frame["player_name"]), ["Player One"])

    def test_refresh_player_reference_handles_every_team_failing(self) -> None:
        """A refresh where every team fails should degrade gracefully, not raise."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "player_reference.csv"
            with mock.patch.object(
                self.source, "fetch_team_squad", side_effect=RuntimeError("boom")
            ):
                result = self.source.refresh_player_reference([10, 20], player_reference_path=reference_path)
        self.assertEqual(result.added_players, 0)
        self.assertEqual(result.updated_players, 0)
        self.assertEqual(len(result.frame), 0)


if __name__ == "__main__":
    unittest.main()
