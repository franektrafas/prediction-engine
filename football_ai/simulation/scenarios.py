""""What if" scenarios: fix one remaining match's result and re-simulate the rest.

The natural phrasing for this product's audience is a *result*, not an
adjusted expected-goals input ("what if Poland beats Germany 2-0", not
"what if Poland's xG were 2.4 instead of 1.6") -- so `what_if_fixed_result`
takes a concrete scoreline for one match, moves it into the fixed/played
set, and re-runs Monte Carlo on whatever matches are still open.
"""

from __future__ import annotations

from football_ai.simulation.competition import PlayedMatch, compute_standings, simulate_group_qualification
from football_ai.simulation.monte_carlo import MatchExpectation


def what_if_fixed_result(
    teams: list[str],
    played_matches: list[PlayedMatch],
    remaining_matches: list[MatchExpectation],
    n_qualifying_spots: int,
    fixed_match_index: int,
    fixed_home_goals: int,
    fixed_away_goals: int,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> tuple[dict[str, float], dict[str, float]]:
    """Return (baseline, scenario) qualification probabilities -- same everything except one fixed result.

    The same `seed` is used for both runs so the *only* thing that differs
    between baseline and scenario is the fixed match -- comparing two
    independently-seeded runs would mix genuine scenario impact with random
    simulation noise.
    """
    if not 0 <= fixed_match_index < len(remaining_matches):
        raise IndexError(f"fixed_match_index {fixed_match_index} out of range for {len(remaining_matches)} remaining matches.")

    baseline = simulate_group_qualification(
        teams, played_matches, remaining_matches, n_qualifying_spots, n_simulations, seed
    )

    fixed_match = remaining_matches[fixed_match_index]
    still_random = [match for index, match in enumerate(remaining_matches) if index != fixed_match_index]
    scenario_played = [
        *played_matches,
        PlayedMatch(fixed_match.home_team, fixed_match.away_team, fixed_home_goals, fixed_away_goals),
    ]

    if still_random:
        scenario = simulate_group_qualification(
            teams, scenario_played, still_random, n_qualifying_spots, n_simulations, seed
        )
    else:
        # The fixed match was the last open one -- nothing left to simulate,
        # the final table (and therefore qualification) is now deterministic.
        standings = compute_standings(teams, scenario_played, [])
        scenario = {standing.team: (1.0 if rank < n_qualifying_spots else 0.0) for rank, standing in enumerate(standings)}

    return baseline, scenario
