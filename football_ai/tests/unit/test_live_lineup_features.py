"""Tests for live lineup feature extraction."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.live_lineup_features import build_live_lineup_features


class LiveLineupFeaturesTestCase(unittest.TestCase):
    """Validate live lineup feature generation."""

    def test_build_live_lineup_features_returns_single_match_row(self) -> None:
        """Two-team lineup input should become one feature row."""
        lineups = pd.DataFrame(
            [
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Diogo Costa",
                    "position": "G",
                    "grid": "1:1",
                    "started": True,
                    "age": 25,
                    "market_value_eur": 40000000,
                    "national_team_caps": 30,
                    "national_team_minutes": 2700,
                    "club_form_score": 7.4,
                    "injury_risk_score": 0.1,
                    "rest_days": 5,
                    "fatigue_index": 0.2,
                    "club_name": "Porto",
                    "league_name": "Primeira Liga",
                    "is_captain": 0,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Ruben Dias",
                    "position": "D",
                    "grid": "2:1",
                    "started": True,
                    "age": 29,
                    "market_value_eur": 70000000,
                    "national_team_caps": 70,
                    "national_team_minutes": 5600,
                    "club_form_score": 7.8,
                    "injury_risk_score": 0.2,
                    "rest_days": 4,
                    "fatigue_index": 0.4,
                    "club_name": "Manchester City",
                    "league_name": "Premier League",
                    "is_captain": 1,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Bruno Fernandes",
                    "position": "M",
                    "grid": "3:1",
                    "started": True,
                    "age": 31,
                    "market_value_eur": 55000000,
                    "national_team_caps": 80,
                    "national_team_minutes": 6000,
                    "club_form_score": 8.1,
                    "injury_risk_score": 0.3,
                    "rest_days": 3,
                    "fatigue_index": 0.5,
                    "club_name": "Manchester United",
                    "league_name": "Premier League",
                    "is_captain": 0,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Cristiano Ronaldo",
                    "position": "F",
                    "grid": "4:1",
                    "started": True,
                    "age": 41,
                    "market_value_eur": 15000000,
                    "national_team_caps": 220,
                    "national_team_minutes": 18000,
                    "club_form_score": 7.0,
                    "injury_risk_score": 0.4,
                    "rest_days": 6,
                    "fatigue_index": 0.3,
                    "club_name": "Al Nassr",
                    "league_name": "Saudi Pro League",
                    "is_captain": 0,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Portugal",
                    "player_name": "Bernardo Silva",
                    "position": "M",
                    "grid": None,
                    "started": False,
                    "club_name": "Manchester City",
                    "league_name": "Premier League",
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Dominik Livakovic",
                    "position": "G",
                    "grid": "1:1",
                    "started": True,
                    "age": 31,
                    "market_value_eur": 10000000,
                    "national_team_caps": 65,
                    "national_team_minutes": 5850,
                    "club_form_score": 7.1,
                    "injury_risk_score": 0.2,
                    "rest_days": 4,
                    "fatigue_index": 0.2,
                    "club_name": "Fenerbahce",
                    "league_name": "Super Lig",
                    "is_captain": 0,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Josip Sutalo",
                    "position": "D",
                    "grid": "2:1",
                    "started": True,
                    "age": 26,
                    "market_value_eur": 22000000,
                    "national_team_caps": 25,
                    "national_team_minutes": 1800,
                    "club_form_score": 7.0,
                    "injury_risk_score": 0.2,
                    "rest_days": 4,
                    "fatigue_index": 0.2,
                    "club_name": "Ajax",
                    "league_name": "Eredivisie",
                    "is_captain": 0,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Luka Modric",
                    "position": "M",
                    "grid": "3:1",
                    "started": True,
                    "age": 40,
                    "market_value_eur": 8000000,
                    "national_team_caps": 195,
                    "national_team_minutes": 15000,
                    "club_form_score": 7.6,
                    "injury_risk_score": 0.3,
                    "rest_days": 5,
                    "fatigue_index": 0.3,
                    "club_name": "Real Madrid",
                    "league_name": "La Liga",
                    "is_captain": 1,
                    "started_previous_match": 1,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Andrej Kramaric",
                    "position": "F",
                    "grid": "4:1",
                    "started": True,
                    "age": 35,
                    "market_value_eur": 6000000,
                    "national_team_caps": 100,
                    "national_team_minutes": 7000,
                    "club_form_score": 7.2,
                    "injury_risk_score": 0.3,
                    "rest_days": 6,
                    "fatigue_index": 0.4,
                    "club_name": "Hoffenheim",
                    "league_name": "Bundesliga",
                    "is_captain": 0,
                    "started_previous_match": 0,
                },
                {
                    "fixture_id": 1567309,
                    "team_name": "Croatia",
                    "player_name": "Mario Pasalic",
                    "position": "M",
                    "grid": None,
                    "started": False,
                    "club_name": "Atalanta",
                    "league_name": "Serie A",
                },
            ]
        )
        result = build_live_lineup_features(lineups)
        self.assertEqual(result.fixture_id, 1567309)
        self.assertEqual(len(result.frame), 1)
        self.assertIn("home_starting_xi_count", result.frame.columns)
        self.assertIn("away_starting_xi_count", result.frame.columns)
        self.assertEqual(int(result.frame.iloc[0]["home_starting_xi_count"]), 4)
        self.assertEqual(int(result.frame.iloc[0]["away_starting_xi_count"]), 4)
        self.assertEqual(int(result.frame.iloc[0]["home_goalkeeper_started"]), 1)
        self.assertEqual(int(result.frame.iloc[0]["away_goalkeeper_started"]), 1)
        self.assertGreater(float(result.frame.iloc[0]["home_total_market_value_starting_xi"]), 0.0)
        self.assertGreater(float(result.frame.iloc[0]["away_total_market_value_starting_xi"]), 0.0)
        self.assertEqual(int(result.frame.iloc[0]["home_captain_started"]), 1)
        self.assertEqual(int(result.frame.iloc[0]["away_captain_started"]), 1)
        self.assertGreater(float(result.frame.iloc[0]["home_starter_continuity_ratio"]), 0.0)
        self.assertGreater(float(result.frame.iloc[0]["away_starter_continuity_ratio"]), 0.0)


if __name__ == "__main__":
    unittest.main()
