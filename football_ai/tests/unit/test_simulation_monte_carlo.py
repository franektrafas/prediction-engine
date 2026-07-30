"""Tests for football_ai/simulation/monte_carlo.py."""

from __future__ import annotations

import unittest

from football_ai.simulation.monte_carlo import MatchExpectation, run_simulations


class MatchExpectationTestCase(unittest.TestCase):
    """Validate the Poisson-lambda-must-be-positive guard."""

    def test_zero_expected_goals_is_rejected(self) -> None:
        """A model predicting exactly 0 must be clamped by the caller, not silently passed through."""
        with self.assertRaises(ValueError):
            MatchExpectation("Poland", "Germany", expected_home_goals=0.0, expected_away_goals=1.2)

    def test_negative_expected_goals_is_rejected(self) -> None:
        """Never a valid Poisson lambda."""
        with self.assertRaises(ValueError):
            MatchExpectation("Poland", "Germany", expected_home_goals=1.2, expected_away_goals=-0.1)


class RunSimulationsTestCase(unittest.TestCase):
    """Validate shape, determinism, and directional sanity of the simulation engine."""

    def test_output_shape_matches_simulations_and_matches(self) -> None:
        """(n_simulations, n_matches) for both home_goals and away_goals."""
        matches = [
            MatchExpectation("Poland", "Germany", 1.4, 1.6),
            MatchExpectation("Netherlands", "Finland", 2.1, 0.8),
        ]
        result = run_simulations(matches, n_simulations=500, seed=42)
        self.assertEqual(result.home_goals.shape, (500, 2))
        self.assertEqual(result.away_goals.shape, (500, 2))
        self.assertEqual(result.n_simulations, 500)

    def test_same_seed_is_fully_reproducible(self) -> None:
        """Identical seed must give bit-identical results -- required for scenarios.py's baseline/scenario comparison."""
        matches = [MatchExpectation("Poland", "Germany", 1.4, 1.6)]
        first = run_simulations(matches, n_simulations=200, seed=7)
        second = run_simulations(matches, n_simulations=200, seed=7)
        self.assertTrue((first.home_goals == second.home_goals).all())
        self.assertTrue((first.away_goals == second.away_goals).all())

    def test_lopsided_expected_goals_produce_lopsided_results(self) -> None:
        """A statistical sanity check: home team with 6x the expected goals should win the vast majority of sims."""
        matches = [MatchExpectation("Strong", "Weak", expected_home_goals=4.0, expected_away_goals=0.3)]
        result = run_simulations(matches, n_simulations=5_000, seed=1)
        home_wins = (result.home_goals[:, 0] > result.away_goals[:, 0]).sum()
        self.assertGreater(home_wins / 5_000, 0.85)

    def test_empty_matches_list_raises(self) -> None:
        """Nothing to simulate should fail loudly, not silently return an empty result."""
        with self.assertRaises(ValueError):
            run_simulations([], n_simulations=100)

    def test_non_positive_simulation_count_raises(self) -> None:
        """A caller passing n_simulations=0 or negative is almost certainly a bug upstream."""
        matches = [MatchExpectation("Poland", "Germany", 1.4, 1.6)]
        with self.assertRaises(ValueError):
            run_simulations(matches, n_simulations=0)


if __name__ == "__main__":
    unittest.main()
