"""Tests for live player enrichment."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.live_player_enrichment import enrich_lineups_with_player_data


class LivePlayerEnrichmentTestCase(unittest.TestCase):
    """Validate lineup enrichment with player metadata."""

    def test_enrich_lineups_with_player_data_matches_by_external_id(self) -> None:
        """Matching should enrich lineup rows when external ids are available."""
        lineups = pd.DataFrame(
            [
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "player_external_id": 1485,
                    "position": "M",
                    "started": True,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Luka Modric",
                    "player_external_id": 1088,
                    "position": "M",
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
                    "national_team_caps": 80,
                    "club_name": "Manchester United",
                    "league_name": "Premier League",
                    "started_previous_match": 1,
                },
                {
                    "team_name": "Croatia",
                    "player_name": "Luka Modric",
                    "player_external_id": 1088,
                    "age": 40,
                    "market_value_eur": 8000000,
                    "national_team_caps": 195,
                    "club_name": "Real Madrid",
                    "league_name": "La Liga",
                    "started_previous_match": 1,
                },
            ]
        )
        result = enrich_lineups_with_player_data(lineups, reference)
        self.assertEqual(result.matched_rows, 2)
        self.assertEqual(result.unmatched_rows, 0)
        self.assertIn("market_value_eur", result.frame.columns)
        self.assertEqual(int(result.frame.iloc[0]["age"]), 31)
        self.assertEqual(str(result.frame.iloc[1]["club_name"]), "Real Madrid")


if __name__ == "__main__":
    unittest.main()
