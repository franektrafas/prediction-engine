"""Tests for football_ai/insights/milestones.py.

`match_count_milestone` is golden-tested against real data (Poland's real
800th international match); confirming an *exact* cumulative-goals crossing
by hand across 892 real matches isn't practical, so `goal_count_milestone`
uses a small, explicitly-controlled fixture instead -- same reasoning
records.py's longest-run tests already used for "not hand-verifiable against
real history at this scale".
"""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from football_ai.insights.milestones import goal_count_milestone, match_count_milestone


_FULL_HISTORY_PATH = Path("football_ai/data/processed/international_results_full_history.parquet")


class MatchCountMilestoneTestCase(unittest.TestCase):
    """Validate against Poland's real, verified 800th match."""

    @unittest.skipUnless(_FULL_HISTORY_PATH.exists(), "requires the regenerated full-history parquet")
    def test_polands_real_800th_match_is_detected(self) -> None:
        """Poland's 800th international was 2018-06-12 vs Lithuania -- verified directly against the real data."""
        records_frame = pd.read_parquet(_FULL_HISTORY_PATH)
        fact = match_count_milestone(
            records_frame, "Poland", as_of_date=pd.Timestamp("2018-06-13", tz="UTC")
        )
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["milestone_number"], 800)
        self.assertEqual(fact.data["opponent"], "Lithuania")

    @unittest.skipUnless(_FULL_HISTORY_PATH.exists(), "requires the regenerated full-history parquet")
    def test_the_801st_match_is_not_a_milestone(self) -> None:
        """One match later, the count is no longer a round number -- must return None, not stale-report 800."""
        records_frame = pd.read_parquet(_FULL_HISTORY_PATH)
        fact = match_count_milestone(
            records_frame, "Poland", as_of_date=pd.Timestamp("2019-01-01", tz="UTC")
        )
        self.assertIsNone(fact)


class GoalCountMilestoneTestCase(unittest.TestCase):
    """Validate against a small, hand-controlled fixture."""

    def setUp(self) -> None:
        """Build a fixture where the team's cumulative goals cross 100 exactly on the last match."""
        rows = []
        # 9 matches scoring 11 each = 99 cumulative, then a 10th scoring 1 more -> exactly 100.
        for i in range(9):
            rows.append(
                {
                    "match_id": i + 1,
                    "match_date": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i * 10),
                    "home_team_name": "Testland",
                    "away_team_name": "Opponent",
                    "home_score": 11,
                    "away_score": 0,
                }
            )
        rows.append(
            {
                "match_id": 10,
                "match_date": pd.Timestamp("2020-03-01", tz="UTC"),
                "home_team_name": "Testland",
                "away_team_name": "Opponent",
                "home_score": 1,
                "away_score": 0,
            }
        )
        self.records_frame = pd.DataFrame(rows)

    def test_100th_career_goal_detected_on_the_exact_match(self) -> None:
        """The 10th match is where cumulative goals hit exactly 100 -- must be flagged there, nowhere else."""
        fact = goal_count_milestone(self.records_frame, "Testland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["milestone_number"], 100)
        self.assertEqual(len(fact.provenance), 10)  # every match up to and including the milestone

    def test_no_milestone_one_match_earlier(self) -> None:
        """As of the 9th match (99 cumulative goals), no round number has been crossed yet."""
        cutoff = pd.Timestamp("2020-03-01", tz="UTC")  # excludes the 10th match (>= cutoff is dropped)
        fact = goal_count_milestone(self.records_frame, "Testland", as_of_date=cutoff)
        self.assertIsNone(fact)


if __name__ == "__main__":
    unittest.main()
