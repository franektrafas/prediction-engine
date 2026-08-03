"""Tests for the Poisson-derived over/under and BTTS helpers in predictions.service.

Reference values cross-checked against `scipy.stats.poisson.sf` -- the
production code deliberately avoids the scipy dependency (see
`_poisson_over_probability`'s docstring), so these tests exist to pin that
the hand-rolled recursion actually matches the real Poisson distribution.
"""

from __future__ import annotations

import unittest

from football_ai.predictions.service import _poisson_btts_probability, _poisson_over_probability


class PoissonOverProbabilityTestCase(unittest.TestCase):
    """Validate _poisson_over_probability against known scipy.stats.poisson.sf values."""

    def test_matches_scipy_reference_values_at_lambda_2_5(self) -> None:
        """lambda=2.5 (a realistic expected-total-goals value) across all three real thresholds."""
        self.assertAlmostEqual(_poisson_over_probability(2.5, 1.5), 0.7127025048163542, places=9)
        self.assertAlmostEqual(_poisson_over_probability(2.5, 2.5), 0.4561868841166703, places=9)
        self.assertAlmostEqual(_poisson_over_probability(2.5, 3.5), 0.2424238668669339, places=9)

    def test_a_near_zero_expected_total_gives_a_near_zero_over_probability(self) -> None:
        """A very low-scoring expected match should make "over 2.5" very unlikely, not exactly 0."""
        self.assertAlmostEqual(_poisson_over_probability(0.1, 2.5), 0.00015465307026467172, places=9)

    def test_result_is_a_valid_probability(self) -> None:
        """Never outside [0, 1], regardless of how extreme expected_total_goals is."""
        for expected_total in (0.0, -5.0, 0.001, 50.0):
            probability = _poisson_over_probability(expected_total, 2.5)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)

    def test_higher_expected_goals_means_higher_over_probability(self) -> None:
        """Monotonicity sanity check -- more expected goals should never make "over" less likely."""
        low = _poisson_over_probability(1.0, 2.5)
        high = _poisson_over_probability(4.0, 2.5)
        self.assertGreater(high, low)


class PoissonBttsProbabilityTestCase(unittest.TestCase):
    """Validate _poisson_btts_probability's independent-Poisson-sides formula."""

    def test_matches_the_hand_derived_formula(self) -> None:
        """P(both score) = (1 - e^-home_lambda) * (1 - e^-away_lambda) under independence."""
        import math

        home_lambda, away_lambda = 1.4, 1.1
        expected = (1 - math.exp(-home_lambda)) * (1 - math.exp(-away_lambda))
        self.assertAlmostEqual(_poisson_btts_probability(home_lambda, away_lambda), expected, places=9)

    def test_is_not_the_old_degenerate_always_one_behavior(self) -> None:
        """Regression test: the prior default (`float(home_goals > 0 and away_goals > 0)`) was

        effectively always 1.0 for any real positive expected-goals pair, not a real probability.
        A real expected-goals pair like (1.4, 1.1) must land at a real intermediate value.
        """
        probability = _poisson_btts_probability(1.4, 1.1)
        self.assertGreater(probability, 0.3)
        self.assertLess(probability, 0.8)

    def test_result_is_a_valid_probability(self) -> None:
        """Never outside [0, 1], including for non-positive expected-goals inputs."""
        for home_expected, away_expected in [(0.0, 0.0), (-1.0, 3.0), (10.0, 10.0)]:
            probability = _poisson_btts_probability(home_expected, away_expected)
            self.assertGreaterEqual(probability, 0.0)
            self.assertLessEqual(probability, 1.0)


if __name__ == "__main__":
    unittest.main()
