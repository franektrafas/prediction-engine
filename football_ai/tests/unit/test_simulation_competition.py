"""Tests for football_ai/simulation/competition.py."""

from __future__ import annotations

import time
import unittest

from football_ai.simulation.competition import (
    PlayedMatch,
    compute_standings,
    simulate_final_positions,
    simulate_group_qualification,
)
from football_ai.simulation.monte_carlo import MatchExpectation


class ComputeStandingsTestCase(unittest.TestCase):
    """Golden test: a hand-tallied 4-team group table."""

    def test_hand_verified_four_team_table(self) -> None:
        """A: 4pts/GD+3, C: 4pts/GD+1, B: 3pts/GD-2, D: 0pts/GD-2 -- worked out by hand alongside this test."""
        played = [
            PlayedMatch("A", "B", 3, 0),
            PlayedMatch("C", "D", 1, 0),
            PlayedMatch("A", "C", 1, 1),
            PlayedMatch("B", "D", 2, 1),
        ]
        standings = compute_standings(["A", "B", "C", "D"], played, remaining_results=[])
        self.assertEqual([s.team for s in standings], ["A", "C", "B", "D"])

        team_a = standings[0]
        self.assertEqual(team_a.points, 4)
        self.assertEqual(team_a.goal_difference, 3)
        self.assertEqual(team_a.goals_for, 4)
        self.assertEqual(team_a.goals_against, 1)
        self.assertEqual((team_a.wins, team_a.draws, team_a.losses), (1, 1, 0))

        team_d = standings[3]
        self.assertEqual(team_d.points, 0)
        self.assertEqual(team_d.goal_difference, -2)
        self.assertEqual((team_d.wins, team_d.draws, team_d.losses), (0, 0, 2))

    def test_points_beat_goal_difference_when_not_tied(self) -> None:
        """A team with more points always outranks one with fewer, regardless of GD."""
        played = [PlayedMatch("A", "B", 10, 0)]  # A: 3pts/GD+10, B: 0pts/GD-10
        remaining = [("C", "D", 1, 0)]  # C: 3pts/GD+1
        standings = compute_standings(["A", "B", "C", "D"], played, remaining)
        # A (3pts, GD+10) and C (3pts, GD+1) are tied on points -- A wins on GD.
        self.assertEqual(standings[0].team, "A")
        self.assertEqual(standings[1].team, "C")


class SimulateGroupQualificationTestCase(unittest.TestCase):
    """Validate the plan's own explicit completion criteria."""

    def setUp(self) -> None:
        """A realistic 4-team group: 2 matches already played, 4 remaining."""
        self.teams = ["Poland", "Germany", "Netherlands", "Finland"]
        self.played = [
            PlayedMatch("Poland", "Finland", 2, 0),
            PlayedMatch("Germany", "Netherlands", 1, 1),
        ]
        self.remaining = [
            MatchExpectation("Poland", "Germany", 1.3, 1.5),
            MatchExpectation("Netherlands", "Finland", 2.0, 0.7),
            MatchExpectation("Poland", "Netherlands", 1.4, 1.4),
            MatchExpectation("Germany", "Finland", 2.1, 0.6),
        ]

    def test_probabilities_sum_to_the_number_of_qualifying_spots(self) -> None:
        """PLAN_SILNIKA.md's own explicit criterion: sum across the group == number of qualifying spots."""
        probabilities = simulate_group_qualification(
            self.teams, self.played, self.remaining, n_qualifying_spots=2, n_simulations=5_000, seed=3
        )
        self.assertAlmostEqual(sum(probabilities.values()), 2.0, places=6)
        self.assertEqual(set(probabilities.keys()), set(self.teams))

    def test_ten_thousand_simulations_complete_in_under_five_seconds(self) -> None:
        """PLAN_SILNIKA.md's own explicit performance bar."""
        start = time.monotonic()
        simulate_group_qualification(
            self.teams, self.played, self.remaining, n_qualifying_spots=2, n_simulations=10_000, seed=1
        )
        elapsed = time.monotonic() - start
        self.assertLess(elapsed, 5.0, f"10,000 simulations took {elapsed:.2f}s, plan's bar is under 5s")

    def test_a_team_that_cannot_mathematically_qualify_gets_zero_probability(self) -> None:
        """A team already eliminated on points, regardless of remaining results, must show 0.0, not a rounding artifact."""
        played = [
            PlayedMatch("Poland", "Doomed", 10, 0),
            PlayedMatch("Germany", "Doomed", 10, 0),
            PlayedMatch("Netherlands", "Doomed", 10, 0),
        ]
        # "Doomed" has played everyone and lost every match (0 points, can never catch up);
        # the only match left is between two teams that both already have more points than Doomed can ever reach.
        remaining = [MatchExpectation("Poland", "Germany", 1.0, 1.0)]
        probabilities = simulate_group_qualification(
            ["Poland", "Germany", "Netherlands", "Doomed"], played, remaining, n_qualifying_spots=2,
            n_simulations=2_000, seed=5,
        )
        self.assertEqual(probabilities["Doomed"], 0.0)


class SimulateFinalPositionsTestCase(unittest.TestCase):
    """The full finishing distribution, not just "did they qualify"."""

    def setUp(self) -> None:
        """Build a small group with one played match and three still to come."""
        self.teams = ["Poland", "Germany", "Netherlands", "Wales"]
        self.played = [PlayedMatch("Poland", "Wales", 2, 0)]
        self.remaining = [
            MatchExpectation("Germany", "Netherlands", 1.4, 1.3),
            MatchExpectation("Poland", "Germany", 1.2, 1.6),
            MatchExpectation("Netherlands", "Wales", 1.8, 0.9),
        ]

    def test_every_team_row_sums_to_one(self) -> None:
        """Each simulation places a team in exactly one place, so its row is a distribution."""
        positions = simulate_final_positions(
            self.teams,
            self.played,
            self.remaining,
            n_simulations=2_000,
            seed=7,
        )
        self.assertEqual(sorted(positions), sorted(self.teams))
        for team, distribution in positions.items():
            self.assertEqual(len(distribution), len(self.teams))
            self.assertAlmostEqual(
                sum(distribution), 1.0, places=9, msg=f"{team} row must sum to 1.0"
            )

    def test_each_position_is_filled_exactly_once_per_simulation(self) -> None:
        """A *column* across teams must also sum to 1.0 -- one team per place."""
        positions = simulate_final_positions(
            self.teams,
            self.played,
            self.remaining,
            n_simulations=2_000,
            seed=11,
        )
        for place in range(len(self.teams)):
            column_total = sum(
                distribution[place] for distribution in positions.values()
            )
            self.assertAlmostEqual(
                column_total, 1.0, places=9, msg=f"place {place + 1}"
            )

    def test_a_dominant_team_wins_the_group_most_often(self) -> None:
        """Positions must track strength, not merely be well-formed numbers."""
        teams = ["Strong", "Weak"]
        remaining = [MatchExpectation("Strong", "Weak", 4.0, 0.2)]
        positions = simulate_final_positions(
            teams, [], remaining, n_simulations=2_000, seed=3
        )
        self.assertGreater(positions["Strong"][0], positions["Weak"][0])


if __name__ == "__main__":
    unittest.main()
