"""Tests for automated feature generation."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.builder import build_features
from football_ai.features.schemas import FeatureBuilderConfig


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

    def test_raw_current_match_score_never_becomes_a_feature(self) -> None:
        """Regression test for a real, severe leak found training on real data for the first time.

        `RAW_MATCH_OUTCOME_FIELDS` used to list "goals" while
        `build_team_match_frame` names the raw per-team columns "goals_for"/
        "goals_against" -- the mismatch meant `home_goals_for`/
        `home_goals_against`/`away_goals_for`/`away_goals_against` (this
        match's own final score) and the base file's own `total_goals`
        column all survived into the feature set unguarded. Invisible in
        every prior test because the existing fixtures use an identical
        score on every row (constant columns can't leak anything -- there's
        no variation to correlate with the target); this fixture uses
        distinct scores per match specifically so a reintroduced leak here
        would need real variation, the same condition that let a real
        LightGBM/CatBoost/XGBoost run hit 100% test accuracy on
        match_outcome before this was fixed.
        """
        matches = pd.DataFrame(
            [
                {"match_date": "2024-01-01", "home_team_name": "England", "away_team_name": "France",
                 "home_score": 2, "away_score": 1},
                {"match_date": "2024-01-05", "home_team_name": "England", "away_team_name": "Spain",
                 "home_score": 0, "away_score": 3},
                {"match_date": "2024-01-09", "home_team_name": "England", "away_team_name": "Italy",
                 "home_score": 1, "away_score": 1},
                {"match_date": "2024-01-13", "home_team_name": "England", "away_team_name": "Germany",
                 "home_score": 4, "away_score": 0},
            ]
        )
        feature_result = build_features(matches)
        leaked = {
            "home_goals_for", "home_goals_against", "away_goals_for", "away_goals_against", "total_goals",
        } & set(feature_result.feature_columns)
        self.assertFalse(leaked, f"Raw current-match score columns leaked as features: {leaked}")

    def test_raw_current_match_cards_never_becomes_a_feature(self) -> None:
        """Regression test for a real, severe leak found auditing for more of the goals-leak's class.

        `build_team_match_frame` builds a "cards" column = this match's own
        `home_yellow_cards + home_red_cards`, wide-merged in as `home_cards`/
        `away_cards`. `RAW_MATCH_OUTCOME_FIELDS` excluded "yellow_cards"/
        "red_cards" individually (the raw source column names) but never
        "cards" (the differently-named engineered column) -- so home_cards/
        away_cards survived unguarded as a near-exact restatement of the
        home_yellow_cards/away_yellow_cards regression targets. Confirmed for
        real, 2026-07-30: corr(home_cards, home_yellow_cards) = 0.977 on the
        real StatsBomb-enriched training file before this fix. Distinct
        yellow-card counts per row, same reasoning as the goals-leak test
        above -- a constant column can't leak anything to correlate against.
        """
        matches = pd.DataFrame(
            [
                {"match_date": "2024-01-01", "home_team_name": "England", "away_team_name": "France",
                 "home_score": 2, "away_score": 1, "home_yellow_cards": 1, "away_yellow_cards": 3,
                 "home_red_cards": 0, "away_red_cards": 0},
                {"match_date": "2024-01-05", "home_team_name": "England", "away_team_name": "Spain",
                 "home_score": 0, "away_score": 3, "home_yellow_cards": 4, "away_yellow_cards": 0,
                 "home_red_cards": 1, "away_red_cards": 0},
                {"match_date": "2024-01-09", "home_team_name": "England", "away_team_name": "Italy",
                 "home_score": 1, "away_score": 1, "home_yellow_cards": 2, "away_yellow_cards": 2,
                 "home_red_cards": 0, "away_red_cards": 1},
                {"match_date": "2024-01-13", "home_team_name": "England", "away_team_name": "Germany",
                 "home_score": 4, "away_score": 0, "home_yellow_cards": 0, "away_yellow_cards": 5,
                 "home_red_cards": 0, "away_red_cards": 1},
            ]
        )
        feature_result = build_features(matches)
        leaked = {"home_cards", "away_cards"} & set(feature_result.feature_columns)
        self.assertFalse(leaked, f"Raw current-match cards columns leaked as features: {leaked}")

    def test_missing_xg_rows_do_not_drag_down_a_teams_rolling_xg_mean(self) -> None:
        """Regression test: `_safe_series` used to default missing xg to 0.0, corrupting rolling means.

        A rolling mean over [2.0, missing] should reflect the one real
        observation (2.0), not treat the missing row as a real 0.0 and
        average to 1.0 -- exactly the sparse-StatsBomb-coverage scenario
        `FeatureBuilderConfig.rolling_metrics`'s own docstring describes
        ("re-add once StatsBomb... covers enough of the dataset"). xg must
        be explicitly requested here since it's not in the default
        rolling_metrics (dropped 2026-07-24 for low importance at 1.1%
        coverage) -- that's a separate, deliberate default-config choice,
        not what this test is about.
        """
        matches = pd.DataFrame(
            [
                {"match_date": "2024-01-01", "home_team_name": "England", "away_team_name": "France",
                 "home_score": 2, "away_score": 1, "home_xg": 2.0},
                {"match_date": "2024-01-05", "home_team_name": "England", "away_team_name": "Spain",
                 "home_score": 1, "away_score": 1},  # no home_xg for this row -- StatsBomb gap
                {"match_date": "2024-01-09", "home_team_name": "England", "away_team_name": "Italy",
                 "home_score": 1, "away_score": 0, "home_xg": 1.4},
            ]
        )
        config = FeatureBuilderConfig(rolling_metrics=FeatureBuilderConfig().rolling_metrics + ("xg",))
        feature_result = build_features(matches, config=config)
        third_match = feature_result.frame.iloc[2]
        self.assertEqual(third_match["home_xg_mean_3"], 2.0)


if __name__ == "__main__":
    unittest.main()
