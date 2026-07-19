"""Tests for live match context assembly."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.predictions.live_match_context import build_live_match_context


class LiveMatchContextTestCase(unittest.TestCase):
    """Validate lineup-to-payload live context conversion."""

    def test_build_live_match_context_returns_payload_updates(self) -> None:
        """Enriched lineups should produce payload-compatible updates."""
        lineups = pd.DataFrame(
            [
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "player_external_id": 1485,
                    "position": "M",
                    "grid": "3:1",
                    "started": True,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Diogo Costa",
                    "player_external_id": 369,
                    "position": "G",
                    "grid": "1:1",
                    "started": True,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Luka Modric",
                    "player_external_id": 1088,
                    "position": "M",
                    "grid": "3:1",
                    "started": True,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Dominik Livakovic",
                    "player_external_id": 1305,
                    "position": "G",
                    "grid": "1:1",
                    "started": True,
                },
            ]
        )
        reference = pd.DataFrame(
            [
                {
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "player_external_id": 1485,
                    "age": 31,
                    "market_value_eur": 55000000,
                    "club_form_score": 8.1,
                    "started_previous_match": 1,
                },
                {
                    "team_name": "Portugal",
                    "player_name": "Diogo Costa",
                    "player_external_id": 369,
                    "age": 25,
                    "market_value_eur": 40000000,
                    "club_form_score": 7.4,
                    "started_previous_match": 1,
                },
                {
                    "team_name": "Croatia",
                    "player_name": "Luka Modric",
                    "player_external_id": 1088,
                    "age": 40,
                    "market_value_eur": 8000000,
                    "club_form_score": 7.6,
                    "started_previous_match": 1,
                },
                {
                    "team_name": "Croatia",
                    "player_name": "Dominik Livakovic",
                    "player_external_id": 1305,
                    "age": 31,
                    "market_value_eur": 10000000,
                    "club_form_score": 7.1,
                    "started_previous_match": 1,
                },
            ]
        )
        result = build_live_match_context(lineups, reference)
        self.assertEqual(result.matched_rows, 4)
        self.assertEqual(result.unmatched_rows, 0)
        self.assertIn("home_lineup_average_age", result.payload_updates)
        self.assertIn("away_lineup_market_value", result.payload_updates)
        self.assertGreater(result.payload_updates["home_lineup_market_value"], 0.0)


if __name__ == "__main__":
    unittest.main()
