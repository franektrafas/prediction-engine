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
