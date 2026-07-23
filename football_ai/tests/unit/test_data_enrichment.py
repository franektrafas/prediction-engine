"""Tests for merging supplemental advanced-stat sources into a base matches frame."""

from __future__ import annotations

import pandas as pd
import unittest

from football_ai.data.enrichment import merge_advanced_stats


class MergeAdvancedStatsTestCase(unittest.TestCase):
    """Validate the base-wins, gap-filling merge contract."""

    def test_enriches_a_match_present_in_both_frames(self) -> None:
        """A match found in the supplemental frame should gain its stat columns."""
        base = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3, "home_xg": 2.1, "away_xg": 2.8}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(len(merged), 1)
        self.assertAlmostEqual(merged.iloc[0]["home_xg"], 2.1)
        self.assertAlmostEqual(merged.iloc[0]["away_xg"], 2.8)

    def test_leaves_a_match_only_in_base_untouched_not_dropped(self) -> None:
        """A match absent from the supplemental frame must stay in the result."""
        base = pd.DataFrame(
            [{"match_date": "2019-06-01", "home_team_name": "Spain", "away_team_name": "Italy",
              "home_score": 1, "away_score": 0}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3, "home_xg": 2.1, "away_xg": 2.8}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["home_team_name"], "Spain")
        self.assertTrue(pd.isna(merged.iloc[0]["home_xg"]))

    def test_matches_across_team_name_aliases(self) -> None:
        """Differently-spelled team names for the same team must still match."""
        base = pd.DataFrame(
            [{"match_date": "2021-06-14", "home_team_name": "Czech Republic", "away_team_name": "Croatia",
              "home_score": 1, "away_score": 1}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2021-06-14", "home_team_name": "Czechia", "away_team_name": "Croatia",
              "home_score": 1, "away_score": 1, "home_shots": 12, "away_shots": 9}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["home_shots"], 12)
        # Base's own spelling is preserved, not overwritten by the supplemental source's.
        self.assertEqual(merged.iloc[0]["home_team_name"], "Czech Republic")

    def test_base_score_wins_over_a_conflicting_supplemental_value(self) -> None:
        """Identity columns (score included) always come from the base frame."""
        base = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 99, "away_score": 99, "home_xg": 2.1}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(merged.iloc[0]["home_score"], 3)
        self.assertEqual(merged.iloc[0]["away_score"], 3)

    def test_matches_across_accented_team_names(self) -> None:
        """A diacritic-only spelling difference must still resolve to the same alias."""
        base = pd.DataFrame(
            [{"match_date": "2024-01-13", "home_team_name": "Ivory Coast", "away_team_name": "Guinea-Bissau",
              "home_score": 2, "away_score": 0}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2024-01-13", "home_team_name": "Côte d'Ivoire", "away_team_name": "Guinea-Bissau",
              "home_score": 2, "away_score": 0, "home_shots": 15}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["home_shots"], 15)

    def test_matches_with_home_and_away_swapped_between_sources(self) -> None:
        """A neutral-venue match where sources disagree on which side is 'home' must still merge."""
        base = pd.DataFrame(
            [{"match_date": "2018-06-25", "home_team_name": "Russia", "away_team_name": "Uruguay",
              "home_score": 0, "away_score": 3}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2018-06-25", "home_team_name": "Uruguay", "away_team_name": "Russia",
              "home_score": 3, "away_score": 0, "home_xg": 2.4, "away_xg": 0.6}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertEqual(len(merged), 1)
        # Base's own home/away orientation (Russia home) wins -- the supplemental
        # source's home_xg (Uruguay's, since it called Uruguay "home") lands on
        # away_xg here instead, matching the base's team-to-side assignment.
        self.assertAlmostEqual(merged.iloc[0]["away_xg"], 2.4)
        self.assertAlmostEqual(merged.iloc[0]["home_xg"], 0.6)

    def test_match_date_survives_in_the_merged_result(self) -> None:
        """match_date is a merge key AND a real output column -- it must not get dropped.

        Regression test: the merge-key cleanup at the end of the function used
        to drop the whole key tuple (including match_date), silently stripping
        it from every caller's result -- invisible here since no earlier test
        asserted on it, but fatal downstream since build_features requires it.
        """
        base = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3}]
        )
        supplemental = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3, "home_xg": 2.1}]
        )
        merged = merge_advanced_stats(base, supplemental)
        self.assertIn("match_date", merged.columns)
        self.assertFalse(pd.isna(merged.iloc[0]["match_date"]))

    def test_empty_supplemental_frame_returns_base_unchanged(self) -> None:
        """An empty supplemental frame should be a safe no-op."""
        base = pd.DataFrame(
            [{"match_date": "2022-12-18", "home_team_name": "Argentina", "away_team_name": "France",
              "home_score": 3, "away_score": 3}]
        )
        merged = merge_advanced_stats(base, pd.DataFrame())
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged.iloc[0]["home_team_name"], "Argentina")


if __name__ == "__main__":
    unittest.main()
