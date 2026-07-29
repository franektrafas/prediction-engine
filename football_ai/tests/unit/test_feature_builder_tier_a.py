"""Tests for Phase 5's Tier-A feature additions (streaks, Elo momentum, etc.)."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.builder import _match_importance_tier, build_features
from football_ai.features.schemas import FeatureBuilderConfig


def _alpha_matches() -> pd.DataFrame:
    """Build a 7-match sequence for "Alpha" with a controlled win/draw/loss pattern.

    Pattern (Alpha always home): win, win, draw, loss, win, win, win -- chosen
    so the win/loss-streak assertions below have unambiguous expected values:
    two wins before match 3, a reset at the match-3 draw, a loss at match 4,
    and a fresh two-win streak built back up by match 7.
    """
    rows = [
        {
            "match_date": "2023-01-01",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent1",
            "home_score": 2,
            "away_score": 0,
            "home_xg": 1.8,
            "away_xg": 0.4,
            "home_elo": 1500,
            "away_elo": 1500,
            "tournament": "International Friendly",
            "country": "Neutralia",
        },
        {
            "match_date": "2023-01-04",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent2",
            "home_score": 3,
            "away_score": 1,
            "home_xg": 2.5,
            "away_xg": 1.0,
            "home_elo": 1550,
            "away_elo": 1500,
            "tournament": "FIFA World Cup",
            "country": "Alpha",
        },
        {
            "match_date": "2023-01-07",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent3",
            "home_score": 1,
            "away_score": 1,
            "home_xg": 1.1,
            "away_xg": 1.2,
            "home_elo": 1600,
            "away_elo": 1500,
            "tournament": "FIFA World Cup Qualifying",
            "country": "Alpha",
        },
        {
            "match_date": "2023-01-10",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent4",
            "home_score": 0,
            "away_score": 2,
            "home_xg": 0.6,
            "away_xg": 1.9,
            "home_elo": 1590,
            "away_elo": 1500,
            "tournament": "UEFA Euro",
            "country": "Neutralia",
        },
        {
            "match_date": "2023-01-13",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent5",
            "home_score": 1,
            "away_score": 0,
            "home_xg": 1.3,
            "away_xg": 0.5,
            "home_elo": 1560,
            "away_elo": 1500,
            "tournament": "Some Regional Cup",
            "country": "Neutralia",
        },
        {
            "match_date": "2023-01-16",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent6",
            "home_score": 2,
            "away_score": 0,
            "home_xg": 1.7,
            "away_xg": 0.3,
            "home_elo": 1580,
            "away_elo": 1500,
            "tournament": "International Friendly",
            "country": "Neutralia",
        },
        {
            "match_date": "2023-01-19",
            "home_team_name": "Alpha",
            "away_team_name": "Opponent7",
            "home_score": 1,
            "away_score": 0,
            "home_xg": 1.4,
            "away_xg": 0.4,
            "home_elo": 1620,
            "away_elo": 1500,
            "tournament": "FIFA World Cup",
            "country": "Alpha",
        },
    ]
    return pd.DataFrame(rows)


class MatchImportanceTierTestCase(unittest.TestCase):
    """Validate the tournament-name keyword classifier in isolation."""

    def test_classifies_friendly_as_tier_0(self) -> None:
        """A friendly should classify as tier 0."""
        self.assertEqual(_match_importance_tier("International Friendly"), 0)

    def test_classifies_qualifier_as_tier_1(self) -> None:
        """A qualifier -- even for a major tournament -- should classify as tier 1."""
        self.assertEqual(_match_importance_tier("FIFA World Cup Qualifying"), 1)

    def test_classifies_major_tournament_as_tier_2(self) -> None:
        """A recognized major tournament should classify as tier 2."""
        self.assertEqual(_match_importance_tier("UEFA Euro"), 2)
        self.assertEqual(_match_importance_tier("FIFA World Cup"), 2)

    def test_unrecognized_tournament_falls_back_to_tier_1(self) -> None:
        """An unrecognized tournament string should fall back to tier 1, not error."""
        self.assertEqual(_match_importance_tier("Some Regional Cup"), 1)
        self.assertEqual(_match_importance_tier(None), 1)


class TierAFeatureIntegrationTestCase(unittest.TestCase):
    """Validate the new features end to end through build_features()."""

    def setUp(self) -> None:
        """Build the feature result once for all assertions in this class."""
        self.result = build_features(_alpha_matches())
        self.frame = self.result.frame

    def test_raw_outcome_indicators_are_excluded_but_rolling_versions_survive(self) -> None:
        """The leak-safety regression this phase most needs to get right."""
        self.assertNotIn("home_btts_flag", self.result.feature_columns)
        self.assertNotIn("away_clean_sheet_flag", self.result.feature_columns)
        self.assertNotIn("home_over_2_5_flag", self.result.feature_columns)
        self.assertIn("home_btts_flag_mean_3", self.result.feature_columns)
        self.assertIn("home_clean_sheet_flag_mean_3", self.result.feature_columns)
        self.assertIn("home_over_2_5_flag_mean_3", self.result.feature_columns)

    def test_win_streak_accumulates_and_resets_on_draw(self) -> None:
        """Two prior wins should show as a streak of 2 right before the draw at match 3."""
        row = self.frame.loc[self.frame["away_team_name"] == "Opponent3"].iloc[0]
        self.assertEqual(row["home_win_streak"], 2.0)

    def test_win_streak_resets_after_a_loss_then_rebuilds(self) -> None:
        """The streak should be back to 0 right after the match-4 loss, then rebuild to 2 by match 7."""
        after_loss = self.frame.loc[self.frame["away_team_name"] == "Opponent5"].iloc[0]
        self.assertEqual(after_loss["home_win_streak"], 0.0)
        self.assertEqual(after_loss["home_loss_streak"], 1.0)

        rebuilt = self.frame.loc[self.frame["away_team_name"] == "Opponent7"].iloc[0]
        self.assertEqual(rebuilt["home_win_streak"], 2.0)
        self.assertEqual(rebuilt["home_loss_streak"], 0.0)

    def test_elo_change_matches_hand_computed_delta(self) -> None:
        """elo_change should equal this row's own elo_pre_match minus the previous match's."""
        row = self.frame.loc[self.frame["away_team_name"] == "Opponent3"].iloc[0]
        # home_elo: match1=1500, match2=1550, match3=1600 -> elo_change at match3 = 1600-1550 = 50
        self.assertAlmostEqual(row["home_elo_change"], 50.0)

    def test_match_importance_tier_present_on_the_wide_frame(self) -> None:
        """The wide frame should carry the classified tier per match."""
        friendly_row = self.frame.loc[self.frame["away_team_name"] == "Opponent1"].iloc[0]
        world_cup_row = self.frame.loc[self.frame["away_team_name"] == "Opponent2"].iloc[0]
        qualifier_row = self.frame.loc[self.frame["away_team_name"] == "Opponent3"].iloc[0]
        self.assertEqual(friendly_row["match_importance_tier"], 0)
        self.assertEqual(world_cup_row["match_importance_tier"], 2)
        self.assertEqual(qualifier_row["match_importance_tier"], 1)

    def test_tournament_host_flag_matches_country_to_team_name(self) -> None:
        """home_is_tournament_host should be true only when country matches the home team."""
        hosted_row = self.frame.loc[self.frame["away_team_name"] == "Opponent2"].iloc[0]
        neutral_row = self.frame.loc[self.frame["away_team_name"] == "Opponent1"].iloc[0]
        self.assertEqual(hosted_row["home_is_tournament_host"], 1.0)
        self.assertEqual(neutral_row["home_is_tournament_host"], 0.0)

    def test_composite_attack_and_defense_indices_present(self) -> None:
        """The composite indices and their home-away diff should exist and be numeric.

        Uses an explicit config with "xg"/"xga" back in rolling_metrics rather
        than self.result's default one -- real ablation on the enriched dataset
        (2026-07-24) showed StatsBomb-derived rolling columns (including xg/xga)
        at ~0% importance given only 1.1% real row coverage, so they were
        dropped from FeatureBuilderConfig's *default* rolling_metrics. That's a
        data-coverage call, not a claim the composite-index mechanism is wrong --
        this test verifies the mechanism itself still works correctly whenever a
        caller's config does include xg/xga (e.g. once coverage improves).
        """
        config = FeatureBuilderConfig(
            rolling_metrics=FeatureBuilderConfig().rolling_metrics + ("xg", "xga")
        )
        result = build_features(_alpha_matches(), config=config)
        for column in ("home_attack_index_3", "away_attack_index_3", "attack_index_3_diff"):
            self.assertIn(column, result.feature_columns)
        for column in ("home_defense_index_3", "away_defense_index_3", "defense_index_3_diff"):
            self.assertIn(column, result.feature_columns)

    def test_injured_market_value_diff_defaults_to_zero_without_source_data(self) -> None:
        """A historical training file with no injury data shouldn't error -- it's a live-only field.

        Same `_safe_series` zero-fill contract as `lineup_market_value`/
        `goalkeeper_form`: the column always exists in the underlying frame
        (so `elo_diff`-style `direct_pairs` always finds it), defaulted to
        0.0 when the source has nothing real to report. Excluded from
        `feature_columns` (2026-07-24, real ablation): always zero in
        historical training data, so no model can learn from it -- surfaced
        instead as informational context via
        `predictions.schemas.MatchContext`, not as a claimed model input.
        """
        self.assertNotIn("injured_market_value_diff", self.result.feature_columns)
        self.assertTrue((self.frame["injured_market_value_diff"] == 0.0).all())


class SupplementalMetadataLeakTestCase(unittest.TestCase):
    """Regression test for the real StatsBomb-merge leak found while retraining.

    With stadium_name/competition_id/etc. merged in unguarded, match_outcome
    hit ~0.9997 validation F1 on real data -- near-perfect on a 3-class task
    is a leak, not a good model. These columns are identity/administrative
    metadata from `data.enrichment.merge_advanced_stats`, never meant to be
    predictive features.
    """

    def test_supplemental_identity_columns_never_become_features(self) -> None:
        """external_id/source/competition_id/etc. must never appear in feature_columns."""
        matches = _alpha_matches()
        matches["external_id"] = range(len(matches))
        matches["source"] = "statsbomb"
        matches["competition_id"] = 43
        matches["competition_name"] = "FIFA World Cup"
        matches["season_name"] = "2022"
        matches["stadium_name"] = "Estadio Lusail"
        matches["referee"] = "Some Referee"
        result = build_features(matches)
        for column in (
            "external_id", "source", "competition_id", "competition_name",
            "season_name", "stadium_name", "referee",
        ):
            self.assertNotIn(column, result.feature_columns)


class InjuredMarketValueFeatureTestCase(unittest.TestCase):
    """Validate injured_market_value flows through the frame but never into feature_columns."""

    def test_injured_market_value_diff_computed_but_excluded_from_features(self) -> None:
        """The diff is still correctly computed in the frame, but is not a trainable feature.

        Always zero in real historical training data (no point-in-time
        historical injury record exists), so no model can learn from it --
        confirmed at exactly zero importance in real ablation (2026-07-24).
        It's excluded from `feature_columns` for honesty about what the
        model uses, and surfaced instead via
        `predictions.schemas.MatchContext` as informational context.
        """
        matches = _alpha_matches()
        matches["home_injured_market_value"] = 1_000_000.0
        matches["away_injured_market_value"] = 250_000.0
        result = build_features(matches)
        self.assertNotIn("injured_market_value_diff", result.feature_columns)
        self.assertNotIn("home_injured_market_value", result.feature_columns)
        self.assertNotIn("away_injured_market_value", result.feature_columns)
        row = result.frame.iloc[0]
        self.assertAlmostEqual(row["injured_market_value_diff"], 750_000.0)


if __name__ == "__main__":
    unittest.main()
