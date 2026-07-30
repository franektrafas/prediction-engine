"""Monte Carlo scoreline simulation.

PLAN_SILNIKA.md's own explicit warning: a simulation must sample
**scorelines**, not 1X2 outcomes, because qualification is decided on goal
difference -- three categories can't tell you a group's final table. Each
side's goals are sampled independently from a Poisson distribution
parameterized by that side's own expected goals (from
`predictions.service.PredictionService`'s `goals` output, or any other
source of expected-goals estimates -- this module doesn't import the
prediction service itself, so it stays testable and reusable without a
trained model or network access).

Fully vectorized (numpy, not a per-simulation Python loop) to meet the
plan's own performance bar: 10,000 simulations of a real group's remaining
matches in well under 5 seconds.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True, frozen=True)
class MatchExpectation:
    """Expected goals for one not-yet-played match -- the engine's unit of input."""

    home_team: str
    away_team: str
    expected_home_goals: float
    expected_away_goals: float

    def __post_init__(self) -> None:
        """Reject non-positive expected goals -- Poisson's lambda must be > 0."""
        if self.expected_home_goals <= 0 or self.expected_away_goals <= 0:
            raise ValueError(
                f"expected goals must be positive (home={self.expected_home_goals}, "
                f"away={self.expected_away_goals}) -- a model predicting exactly 0 "
                "should be clamped to a small positive floor by the caller, not passed through."
            )


@dataclass(slots=True, frozen=True)
class SimulationResult:
    """Vectorized output of `run_simulations`: every simulated scoreline for every match.

    `home_goals`/`away_goals` are `(n_simulations, n_matches)` integer
    arrays -- `home_goals[i, j]` is simulation `i`'s scoreline for
    `matches[j]`. Kept as arrays (not a list of per-simulation dataclasses)
    specifically so `competition.py` can compute standings with vectorized
    numpy operations too, not a second layer of Python loops.
    """

    matches: tuple[MatchExpectation, ...]
    home_goals: np.ndarray
    away_goals: np.ndarray

    @property
    def n_simulations(self) -> int:
        """Number of simulations run."""
        return self.home_goals.shape[0]


def run_simulations(
    matches: list[MatchExpectation],
    n_simulations: int,
    seed: int | None = None,
) -> SimulationResult:
    """Run N independent simulations of every given match, all sides sampled from Poisson(expected_goals).

    `seed=None` (the default) means genuinely random results, appropriate
    for real use -- pass an explicit int for reproducible tests.
    """
    if not matches:
        raise ValueError("matches must be non-empty -- nothing to simulate.")
    if n_simulations <= 0:
        raise ValueError(f"n_simulations must be positive, got {n_simulations}.")
    rng = np.random.default_rng(seed)
    home_lambda = np.array([match.expected_home_goals for match in matches])
    away_lambda = np.array([match.expected_away_goals for match in matches])
    home_goals = rng.poisson(home_lambda, size=(n_simulations, len(matches)))
    away_goals = rng.poisson(away_lambda, size=(n_simulations, len(matches)))
    return SimulationResult(matches=tuple(matches), home_goals=home_goals, away_goals=away_goals)
