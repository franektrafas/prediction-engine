"""Tests for player reference maintenance."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.player_reference import update_player_reference


class PlayerReferenceTestCase(unittest.TestCase):
    """Validate automatic player reference updates."""

    def test_update_player_reference_adds_and_updates_players(self) -> None:
        """Observed players should extend and enrich the reference table."""
        existing = pd.DataFrame(
            [
                {
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "player_external_id": 1485,
                    "age": 30,
                    "club_name": "Manchester United",
                }
            ]
        )
        observed = pd.DataFrame(
            [
                {
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "player_external_id": 1485,
                    "age": 31,
                    "market_value_eur": 55000000,
                },
                {
                    "team_name": "Portugal",
                    "player_name": "Diogo Costa",
                    "player_external_id": 369,
                    "age": 25,
                    "position": "Goalkeeper",
                },
            ]
        )
        result = update_player_reference(existing, observed, source_name="football_api")
        self.assertEqual(result.added_players, 1)
        self.assertGreaterEqual(result.updated_players, 1)
        self.assertEqual(len(result.frame), 2)
        bruno_row = result.frame.loc[result.frame["player_name"] == "Bruno Fernandes"].iloc[0]
        self.assertEqual(float(bruno_row["age"]), 31.0)
        self.assertEqual(float(bruno_row["market_value_eur"]), 55000000.0)


if __name__ == "__main__":
    unittest.main()
