"""Tests for football_ai/simulation/scenarios.py."""

from __future__ import annotations

import unittest

from football_ai.simulation.competition import PlayedMatch
from football_ai.simulation.monte_carlo import MatchExpectation
from football_ai.simulation.scenarios import what_if_fixed_result


class WhatIfFixedResultTestCase(unittest.TestCase):
    """PLAN_SILNIKA.md's own criterion: a what-if scenario must move the result in the expected direction."""

    def setUp(self) -> None:
        """A tight, evenly-matched 4-team group -- baseline odds should be far from 0/1, leaving room to move."""
        self.teams = ["Poland", "Germany", "Netherlands", "Finland"]
        self.played = [PlayedMatch("Poland", "Finland", 1, 1), PlayedMatch("Germany", "Netherlands", 1, 1)]
        self.remaining = [
            MatchExpectation("Poland", "Germany", 1.3, 1.3),
            MatchExpectation("Netherlands", "Finland", 1.3, 1.3),
            MatchExpectation("Poland", "Netherlands", 1.3, 1.3),
            MatchExpectation("Germany", "Finland", 1.3, 1.3),
        ]

    def test_a_forced_big_win_improves_that_teams_odds(self) -> None:
        """Forcing Poland to beat Germany 4-0 (instead of a roughly 50/50 Poisson draw) must raise Poland's qualification chance."""
        baseline, scenario = what_if_fixed_result(
            self.teams, self.played, self.remaining, n_qualifying_spots=2,
            fixed_match_index=0, fixed_home_goals=4, fixed_away_goals=0,
            n_simulations=5_000, seed=11,
        )
        self.assertGreater(scenario["Poland"], baseline["Poland"])

    def test_a_forced_big_loss_worsens_that_teams_odds(self) -> None:
        """The mirror case -- forcing Poland to lose big must lower Poland's qualification chance."""
        baseline, scenario = what_if_fixed_result(
            self.teams, self.played, self.remaining, n_qualifying_spots=2,
            fixed_match_index=0, fixed_home_goals=0, fixed_away_goals=4,
            n_simulations=5_000, seed=11,
        )
        self.assertLess(scenario["Poland"], baseline["Poland"])

    def test_fixing_the_last_remaining_match_is_fully_deterministic(self) -> None:
        """With no more randomness left, qualification probabilities collapse to exactly 0.0 or 1.0."""
        # Only one match left to simulate, matching every OTHER match already being played.
        played = [
            PlayedMatch("Poland", "Finland", 3, 0),
            PlayedMatch("Germany", "Netherlands", 3, 0),
            PlayedMatch("Poland", "Netherlands", 3, 0),
            PlayedMatch("Germany", "Finland", 3, 0),
        ]
        remaining = [MatchExpectation("Poland", "Germany", 1.3, 1.3)]
        _, scenario = what_if_fixed_result(
            self.teams, played, remaining, n_qualifying_spots=2,
            fixed_match_index=0, fixed_home_goals=1, fixed_away_goals=0,
            n_simulations=1_000, seed=2,
        )
        self.assertTrue(all(probability in (0.0, 1.0) for probability in scenario.values()))
        self.assertEqual(sum(scenario.values()), 2.0)

    def test_out_of_range_match_index_raises(self) -> None:
        """A caller referencing a match that doesn't exist is a programming error, not a silent no-op."""
        with self.assertRaises(IndexError):
            what_if_fixed_result(
                self.teams, self.played, self.remaining, n_qualifying_spots=2,
                fixed_match_index=99, fixed_home_goals=1, fixed_away_goals=0,
            )


if __name__ == "__main__":
    unittest.main()
