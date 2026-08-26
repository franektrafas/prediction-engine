"""Tests for automated feature generation."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.features.builder import build_features, merge_squad_indices
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

    def test_neutral_venue_does_not_contaminate_true_home_rolling(self) -> None:
        """Regression test for the neutral-venue home-advantage bug (fixed 2026-08-25).

        Before the fix, the venue-split rolling bucket grouped purely on
        `is_home` -- a team listed as home on a neutral-venue match (a World
        Cup final, a Nations League tie on neutral ground) still fed that
        match into its "true home" rolling average, exactly as if it had
        real home-crowd advantage. Poland's third match here is a neutral
        "home" match with an extreme, distinct goals_for=99 -- if it leaked
        into the true-home bucket, the 4th match's `home_goals_for_mean_3_venue`
        would be pulled far above 3 (mean of 1, 3, 99); with the fix, it must
        reflect only the two real home matches (1, 3), i.e. exactly 2.0.
        """
        matches = pd.DataFrame(
            [
                {"match_date": "2024-01-01", "home_team_name": "Poland", "away_team_name": "England",
                 "home_score": 1, "away_score": 0, "neutral_venue": False},
                {"match_date": "2024-01-05", "home_team_name": "Poland", "away_team_name": "France",
                 "home_score": 3, "away_score": 1, "neutral_venue": False},
                {"match_date": "2024-01-09", "home_team_name": "Poland", "away_team_name": "Brazil",
                 "home_score": 99, "away_score": 0, "neutral_venue": True},
                {"match_date": "2024-01-13", "home_team_name": "Poland", "away_team_name": "Germany",
                 "home_score": 5, "away_score": 2, "neutral_venue": False},
            ]
        )
        feature_result = build_features(matches)
        fourth_match = feature_result.frame.iloc[3]
        self.assertEqual(fourth_match["home_goals_for_mean_3_venue"], 2.0)


class NeutralVenueFullPipelineSymmetryTestCase(unittest.TestCase):
    """Full build_features()-level regression test for PROMPT_NAPRAWA_PRZECIEKU_HOME.md.

    `test_neutral_venue_does_not_contaminate_true_home_rolling` passed the
    whole time the real leak was live in production -- it only exercises
    `_add_rolling_features`'s venue split, not the full feature set. The
    real bug (found 2026-08-26 via a live-service diagnostic) was `is_home`
    itself surviving `_wide_merge` unexcluded: a purely positional sheet
    flag, reaching the model as a direct, ungated "am I the home-labeled
    side" signal even on a neutral match. This test instead builds the
    *complete* numeric feature set for the same neutral match under both
    home/away labelings and asserts every column mirrors -- so any future
    leak of this general shape (a new column that forgets to respect
    neutral-venue symmetry) fails loudly here, not just in production.
    """

    _PRIOR_MATCHES = [
        {
            "match_date": "2024-01-01",
            "home_team_name": "TeamA",
            "away_team_name": "TeamC",
            "home_score": 2,
            "away_score": 0,
            "neutral_venue": False,
        },
        {
            "match_date": "2024-01-05",
            "home_team_name": "TeamB",
            "away_team_name": "TeamD",
            "home_score": 1,
            "away_score": 1,
            "neutral_venue": False,
        },
    ]

    def _neutral_match_feature_row(
        self, home: str, away: str, home_score: int, away_score: int
    ):
        matches = pd.DataFrame(
            [
                *self._PRIOR_MATCHES,
                {
                    "match_date": "2024-02-01",
                    "home_team_name": home,
                    "away_team_name": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "neutral_venue": True,
                },
            ]
        )
        feature_result = build_features(matches)
        return feature_result.frame.iloc[-1], feature_result.numeric_columns

    def test_every_numeric_feature_mirrors_across_a_home_away_relabel(self) -> None:
        """Same real neutral match, relabelled -- every home_X must equal away_X."""
        original_row, numeric_columns = self._neutral_match_feature_row(
            "TeamA", "TeamB", 2, 1
        )
        # Same real event, TeamB now the sheet-labelled home side -- score swapped
        # to match (TeamB scored 1, TeamA scored 2, same as above).
        swapped_row, _ = self._neutral_match_feature_row("TeamB", "TeamA", 1, 2)

        mismatches = []
        for column in numeric_columns:
            if not (column.startswith("home_") or column.startswith("away_")):
                continue
            mirror_column = (
                "away_" + column[len("home_"):] if column.startswith("home_")
                else "home_" + column[len("away_"):]
            )
            if mirror_column not in numeric_columns:
                continue
            value, mirror_value = original_row[column], swapped_row[mirror_column]
            if pd.isna(value) and pd.isna(mirror_value):
                continue
            both_known = not pd.isna(value) and not pd.isna(mirror_value)
            difference = abs(float(value) - float(mirror_value)) if both_known else None
            close_enough = difference is not None and difference <= 1e-9
            if not close_enough:
                mismatches.append((column, value, mirror_column, mirror_value))

        self.assertEqual(
            mismatches,
            [],
            f"{len(mismatches)} numeric feature column(s) break home/away symmetry "
            f"on a neutral-venue match: {mismatches}",
        )


class MergeSquadIndicesTestCase(unittest.TestCase):
    """Regression tests for PROMPT_ZAMKNIECIE_WATKOW.md item 2's real gap.

    The squad age/height/continuity/cohesion indices were computed by
    `footballsquads.py` since 2026-08-20 but never merged into `matches` --
    no model ever saw them. These tests cover the fix: `merge_squad_indices`
    (and `build_features`'s optional `squad_summary` parameter that calls
    it).
    """

    _SUMMARY = pd.DataFrame(
        [
            {
                "competition": "worldcup",
                "team_name": "Poland",
                "tournament_start_date": pd.Timestamp("2022-11-20", tz="UTC"),
                "average_age_years": 28.5,
                "average_height_m": 1.85,
                "squad_continuity_ratio": 0.6,
                "largest_club_cluster_size": 2,
                "largest_club_cluster_pairs": 1,
                "total_intra_club_pairs": 2,
            },
            {
                "competition": "worldcup",
                "team_name": "France",
                "tournament_start_date": pd.Timestamp("2022-11-20", tz="UTC"),
                "average_age_years": 26.1,
                "average_height_m": 1.83,
                "squad_continuity_ratio": 0.8,
                "largest_club_cluster_size": 3,
                "largest_club_cluster_pairs": 3,
                "total_intra_club_pairs": 3,
            },
        ]
    )

    def test_covered_tournament_match_gets_both_sides_squad_indices(self) -> None:
        """A real World Cup match near the squad's start date picks up both sides."""
        matches = pd.DataFrame(
            [
                {
                    "match_date": "2022-11-26",
                    "home_team_name": "Poland",
                    "away_team_name": "France",
                    "tournament": "FIFA World Cup",
                    "home_score": 0,
                    "away_score": 0,
                }
            ]
        )
        merged = merge_squad_indices(matches, self._SUMMARY)
        self.assertEqual(merged["home_squad_average_age_years"].iloc[0], 28.5)
        self.assertEqual(merged["away_squad_average_age_years"].iloc[0], 26.1)
        self.assertEqual(merged["home_squad_largest_club_cluster_size"].iloc[0], 2)

    def test_uncovered_tournament_stays_none_not_fabricated(self) -> None:
        """A friendly (not one of the five covered competitions) must get NaN, not 0."""
        matches = pd.DataFrame(
            [
                {
                    "match_date": "2022-11-26",
                    "home_team_name": "Poland",
                    "away_team_name": "France",
                    "tournament": "Friendly",
                    "home_score": 1,
                    "away_score": 1,
                }
            ]
        )
        merged = merge_squad_indices(matches, self._SUMMARY)
        self.assertTrue(pd.isna(merged["home_squad_average_age_years"].iloc[0]))

    def test_match_far_outside_tolerance_does_not_pick_up_a_stale_squad(self) -> None:
        """A match years after the squad's tournament must not inherit that old squad.

        Real scenario this guards against: without a bounded `tolerance`,
        `merge_asof(direction="backward")` would keep matching a team's
        *next* World Cup qualifier -- years later -- to its old squad
        snapshot, since "the nearest earlier row" has no natural cutoff on
        its own.
        """
        matches = pd.DataFrame(
            [
                {
                    "match_date": "2025-03-15",
                    "home_team_name": "Poland",
                    "away_team_name": "France",
                    "tournament": "FIFA World Cup",
                    "home_score": 2,
                    "away_score": 0,
                }
            ]
        )
        merged = merge_squad_indices(matches, self._SUMMARY)
        self.assertTrue(pd.isna(merged["home_squad_average_age_years"].iloc[0]))

    def test_never_overwrites_an_existing_column(self) -> None:
        """If `matches` already has a same-named column, the real one must win."""
        matches = pd.DataFrame(
            [
                {
                    "match_date": "2022-11-26",
                    "home_team_name": "Poland",
                    "away_team_name": "France",
                    "tournament": "FIFA World Cup",
                    "home_score": 0,
                    "away_score": 0,
                    "home_squad_average_age_years": 99.9,
                }
            ]
        )
        merged = merge_squad_indices(matches, self._SUMMARY)
        self.assertEqual(merged["home_squad_average_age_years"].iloc[0], 99.9)

    def test_build_features_surfaces_squad_indices_as_numeric_features(self) -> None:
        """End-to-end: passing squad_summary makes the indices real, usable features."""
        matches = pd.DataFrame(
            [
                {
                    "match_date": f"2022-11-{day:02d}",
                    "home_team_name": "Poland",
                    "away_team_name": "France",
                    "tournament": "FIFA World Cup",
                    "home_score": 1,
                    "away_score": 0,
                }
                for day in (20, 26)
            ]
        )
        feature_result = build_features(matches, squad_summary=self._SUMMARY)
        numeric_columns = feature_result.numeric_columns
        self.assertIn("home_squad_average_age_years", numeric_columns)
        self.assertIn("away_squad_largest_club_cluster_size", numeric_columns)


if __name__ == "__main__":
    unittest.main()
