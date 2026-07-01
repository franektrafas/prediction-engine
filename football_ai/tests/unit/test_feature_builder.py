"""Tests for automated feature generation."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.builder import build_features


class FeatureBuilderTestCase(unittest.TestCase):
    """Test the feature engineering pipeline."""

    def test_build_features_generates_large_feature_space_and_targets(self) -> None:
        """Feature generation should return a wide feature matrix and target set."""
        matches = pd.DataFrame(
            [
                {
                    "match_date": f"2024-01-{day:02d}",
                    "home_team_name": "England",
                    "away_team_name": "France",
                    "home_score": 2,
                    "away_score": 1,
                    "home_xg": 1.8,
                    "away_xg": 1.2,
                    "home_shots": 14,
                    "away_shots": 10,
                    "home_shots_on_target": 6,
                    "away_shots_on_target": 4,
                    "home_corners": 5,
                    "away_corners": 4,
                    "home_yellow_cards": 2,
                    "away_yellow_cards": 1,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                    "home_possession": 56,
                    "away_possession": 44,
                    "home_pressing_intensity": 9,
                    "away_pressing_intensity": 7,
                    "home_elo": 1820,
                    "away_elo": 1850,
                }
                for day in range(1, 13)
            ]
        )
        feature_result = build_features(matches)
        self.assertGreater(len(feature_result.feature_columns), 200)
        self.assertIn("match_outcome", feature_result.targets)
        self.assertIn("btts", feature_result.targets)
        self.assertEqual(len(feature_result.frame), len(matches))


if __name__ == "__main__":
    unittest.main()
