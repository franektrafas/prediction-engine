"""Tests for football_ai/insights/player_milestones.py.

`player_goal_milestone` is golden-tested against real data (Robert
Lewandowski's real 50th international goal for Poland); own-goal exclusion
and the "not yet a milestone" case use a small, hand-controlled fixture --
same reasoning `test_insights_milestones.py`'s `GoalCountMilestoneTestCase`
already used.
"""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from football_ai.insights.player_milestones import (
    player_goal_milestone,
    player_goal_milestones_for_team,
    player_goals,
)


_GOALSCORERS_PATH = Path("football_ai/data/processed/international_results_goalscorers.parquet")


class PlayerGoalMilestoneRealDataTestCase(unittest.TestCase):
    """Validate against Robert Lewandowski's real 50th international goal for Poland."""

    @unittest.skipUnless(_GOALSCORERS_PATH.exists(), "requires the regenerated goalscorers parquet")
    def test_lewandowskis_real_50th_goal_is_detected(self) -> None:
        """His real 50th goal for Poland was 2021-06-23 against Sweden."""
        goalscorers_frame = pd.read_parquet(_GOALSCORERS_PATH)
        fact = player_goal_milestone(
            goalscorers_frame,
            "Poland",
            "Robert Lewandowski",
            as_of_date=pd.Timestamp("2021-06-24", tz="UTC"),
        )
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["milestone_number"], 50)
        self.assertEqual(fact.data["opponent"], "Sweden")
        self.assertEqual(len(fact.provenance), 50)  # every one of his goals up to and including this one

    @unittest.skipUnless(_GOALSCORERS_PATH.exists(), "requires the regenerated goalscorers parquet")
    def test_no_milestone_one_goal_earlier(self) -> None:
        """Before the match containing his 50th goal, 50 hasn't been crossed yet -- must return None.

        Real data quirk worth noting: his 49th and 50th goals were both
        scored in the *same* match (a brace vs. Sweden) -- so the cutoff
        must exclude that entire match's date, not just "one goal back" by
        index, or it would still include both goals.
        """
        goalscorers_frame = pd.read_parquet(_GOALSCORERS_PATH)
        goals = player_goals(goalscorers_frame, "Poland", "Robert Lewandowski")
        fiftieth_goal_date = goals.iloc[49]["match_date"]
        prior_dates = goals.loc[goals["match_date"] < fiftieth_goal_date, "match_date"]
        cutoff = prior_dates.max() + pd.Timedelta(seconds=1)
        fact = player_goal_milestone(goalscorers_frame, "Poland", "Robert Lewandowski", as_of_date=cutoff)
        self.assertIsNone(fact)

    @unittest.skipUnless(_GOALSCORERS_PATH.exists(), "requires the regenerated goalscorers parquet")
    def test_scanning_the_whole_team_finds_multiple_players_real_milestones(self) -> None:
        """The same real cutoff also catches Arkadiusz Milik's real 10th goal for Poland."""
        goalscorers_frame = pd.read_parquet(_GOALSCORERS_PATH)
        facts = player_goal_milestones_for_team(
            goalscorers_frame, "Poland", as_of_date=pd.Timestamp("2021-06-24", tz="UTC")
        )
        players = {fact.data["player_name"]: fact.data["milestone_number"] for fact in facts}
        self.assertEqual(players.get("Robert Lewandowski"), 50)
        self.assertEqual(players.get("Arkadiusz Milik"), 10)


class PlayerGoalMilestoneFixtureTestCase(unittest.TestCase):
    """Own-goal exclusion and cross-team isolation on a small, hand-controlled fixture."""

    def setUp(self) -> None:
        """9 real goals for Testland's Jan Kowalski, one own goal in between that must not count."""
        rows = []
        for i in range(9):
            rows.append(
                {
                    "match_id": i + 1,
                    "match_date": pd.Timestamp("2020-01-01", tz="UTC") + pd.Timedelta(days=i * 10),
                    "home_team_name": "Testland",
                    "away_team_name": "Opponent",
                    "team_name": "Testland",
                    "scorer": "Jan Kowalski",
                    "own_goal": False,
                }
            )
        # An own goal, credited to Testland's side but never Kowalski's own tally.
        rows.append(
            {
                "match_id": 10,
                "match_date": pd.Timestamp("2020-04-01", tz="UTC"),
                "home_team_name": "Testland",
                "away_team_name": "Opponent",
                "team_name": "Testland",
                "scorer": "Piotr Nowak",
                "own_goal": True,
            }
        )
        rows.append(
            {
                "match_id": 11,
                "match_date": pd.Timestamp("2020-05-01", tz="UTC"),
                "home_team_name": "Testland",
                "away_team_name": "Opponent",
                "team_name": "Testland",
                "scorer": "Jan Kowalski",
                "own_goal": False,
            }
        )
        self.goalscorers_frame = pd.DataFrame(rows)

    def test_own_goal_never_counts_toward_the_scorers_own_tally(self) -> None:
        """10 rows total, but only 10 real (non-own) goals for Kowalski -- his 10th, not counting the own goal."""
        fact = player_goal_milestone(self.goalscorers_frame, "Testland", "Jan Kowalski", round_numbers=(10,))
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["milestone_number"], 10)
        self.assertEqual(len(fact.provenance), 10)

    def test_no_milestone_for_a_round_number_not_reached(self) -> None:
        fact = player_goal_milestone(self.goalscorers_frame, "Testland", "Jan Kowalski", round_numbers=(20,))
        self.assertIsNone(fact)

    def test_own_goal_scorer_has_no_real_goals_of_their_own(self) -> None:
        """Piotr Nowak's only row is his own goal -- he has zero real goals, so no Fact should exist."""
        fact = player_goal_milestone(self.goalscorers_frame, "Testland", "Piotr Nowak", round_numbers=(1,))
        self.assertIsNone(fact)


if __name__ == "__main__":
    unittest.main()
