"""Tests for prediction-time inference feature building."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from football_ai.predictions.inference_feature_builder import build_inference_feature_frame
from football_ai.predictions.schemas import MatchContextPayload


class InferenceFeatureBuilderTestCase(unittest.TestCase):
    """Validate prediction-time feature preparation."""

    def test_build_inference_feature_frame_drops_sparse_ratio_columns(self) -> None:
        """Inference-time feature preparation should drop fragile ratio features."""
        historical_matches = pd.DataFrame(
            [
                {
                    "match_date": "2026-07-01T18:00:00Z",
                    "home_team_name": "Spain",
                    "away_team_name": "France",
                    "home_score": 2,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "country": "Spain",
                    "neutral_venue": False,
                    "home_elo": 1980,
                    "away_elo": 1970,
                    "home_xg": 1.8,
                    "away_xg": 1.1,
                    "home_shots": 13,
                    "away_shots": 10,
                    "home_shots_on_target": 5,
                    "away_shots_on_target": 4,
                    "home_corners": 6,
                    "away_corners": 3,
                    "home_yellow_cards": 1,
                    "away_yellow_cards": 2,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                    "home_possession": 55,
                    "away_possession": 45,
                    "home_pressing_intensity": 10,
                    "away_pressing_intensity": 8,
                },
                {
                    "match_date": "2026-07-05T18:00:00Z",
                    "home_team_name": "Portugal",
                    "away_team_name": "Germany",
                    "home_score": 1,
                    "away_score": 1,
                    "tournament": "Friendly",
                    "country": "Portugal",
                    "neutral_venue": False,
                    "home_elo": 1940,
                    "away_elo": 1950,
                    "home_xg": 1.3,
                    "away_xg": 1.2,
                    "home_shots": 11,
                    "away_shots": 12,
                    "home_shots_on_target": 4,
                    "away_shots_on_target": 4,
                    "home_corners": 4,
                    "away_corners": 5,
                    "home_yellow_cards": 2,
                    "away_yellow_cards": 2,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                    "home_possession": 51,
                    "away_possession": 49,
                    "home_pressing_intensity": 9,
                    "away_pressing_intensity": 9,
                },
            ]
        )
        payload = MatchContextPayload(
            match_date=datetime(2026, 7, 20, tzinfo=timezone.utc),
            home_team_name="Spain",
            away_team_name="Portugal",
            tournament="Friendly",
            country="Spain",
            home_elo=1985,
            away_elo=1940,
            home_recent_xg=1.8,
            away_recent_xg=1.5,
            home_recent_xga=0.9,
            away_recent_xga=1.0,
        )
        result = build_inference_feature_frame(payload, historical_matches)
        self.assertEqual(len(result.frame), 1)
        self.assertGreater(len(result.dropped_columns), 0)
        self.assertFalse(any(column.endswith("_ratio") and "xg_" in column for column in result.frame.columns))


if __name__ == "__main__":
    unittest.main()
