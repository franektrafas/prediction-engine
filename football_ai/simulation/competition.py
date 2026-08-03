"""Group standings and qualification-probability simulation.

Combines already-played results (fixed) with Monte Carlo-simulated
remaining results (different every simulation) into a full table each time,
then reports how often each team finished in a qualifying position across
every simulation -- that frequency *is* the qualification probability.

Known, documented simplification: tiebreaking uses points -> goal
difference -> goals scored only. Real UEFA/FIFA group tiebreakers also
consider head-to-head results before overall goal difference; not
implemented here yet -- flagged, not silently wrong. For most simulated
groups the top-N by points/GD/GF matches the head-to-head-aware result
anyway (they only diverge in specific tied scenarios), but a table this
module produces should not be presented as authoritative for a genuinely
tied edge case.
"""

from __future__ import annotations

from dataclasses import dataclass

from football_ai.simulation.monte_carlo import MatchExpectation, run_simulations


@dataclass(slots=True, frozen=True)
class PlayedMatch:
    """A match with an already-known, real scoreline (not simulated)."""

    home_team: str
    away_team: str
    home_goals: int
    away_goals: int


@dataclass(slots=True, frozen=True)
class GroupStanding:
    """One team's row in a group table, for one (real or simulated) final state."""

    team: str
    points: int
    goal_difference: int
    goals_for: int
    goals_against: int
    wins: int
    draws: int
    losses: int


def _apply_result(standings: dict[str, list[int]], home_team: str, away_team: str, home_goals: int, away_goals: int) -> None:
    """Mutate the running per-team [points, gf, ga, wins, draws, losses] tallies for one match."""
    for team in (home_team, away_team):
        if team not in standings:
            standings[team] = [0, 0, 0, 0, 0, 0]
    standings[home_team][1] += home_goals
    standings[home_team][2] += away_goals
    standings[away_team][1] += away_goals
    standings[away_team][2] += home_goals
    if home_goals > away_goals:
        standings[home_team][0] += 3
        standings[home_team][3] += 1
        standings[away_team][5] += 1
    elif home_goals < away_goals:
        standings[away_team][0] += 3
        standings[away_team][3] += 1
        standings[home_team][5] += 1
    else:
        standings[home_team][0] += 1
        standings[away_team][0] += 1
        standings[home_team][4] += 1
        standings[away_team][4] += 1


def compute_standings(
    teams: list[str],
    played_matches: list[PlayedMatch],
    remaining_results: list[tuple[str, str, int, int]],
) -> list[GroupStanding]:
    """Compute one final group table from played results plus a set of (home, away, home_goals, away_goals) results.

    `remaining_results` is a plain tuple list (not `MatchExpectation` --
    those carry *expected* goals, this needs one concrete simulated or real
    outcome) so this function has no dependency on the Monte Carlo engine
    and can be tested/used with hand-written results directly.
    """
    tallies: dict[str, list[int]] = {team: [0, 0, 0, 0, 0, 0] for team in teams}
    for match in played_matches:
        _apply_result(tallies, match.home_team, match.away_team, match.home_goals, match.away_goals)
    for home_team, away_team, home_goals, away_goals in remaining_results:
        _apply_result(tallies, home_team, away_team, home_goals, away_goals)

    standings = [
        GroupStanding(
            team=team,
            points=values[0],
            goal_difference=values[1] - values[2],
            goals_for=values[1],
            goals_against=values[2],
            wins=values[3],
            draws=values[4],
            losses=values[5],
        )
        for team, values in tallies.items()
    ]
    return sorted(standings, key=lambda s: (s.points, s.goal_difference, s.goals_for), reverse=True)


def simulate_group_qualification(
    teams: list[str],
    played_matches: list[PlayedMatch],
    remaining_matches: list[MatchExpectation],
    n_qualifying_spots: int,
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Run N simulations of the remaining matches and return each team's qualification probability.

    A team's probability is simply how often it finished in the top
    `n_qualifying_spots` across every simulation -- the direct empirical
    frequency, not a closed-form approximation. Probabilities across all
    teams in a group necessarily sum to `n_qualifying_spots` (each
    simulation has exactly that many qualifiers), which is exactly
    PLAN_SILNIKA.md's own completion criterion for this step.
    """
    simulation = run_simulations(remaining_matches, n_simulations, seed)
    qualification_counts = dict.fromkeys(teams, 0)
    for simulation_index in range(simulation.n_simulations):
        remaining_results = [
            (
                match.home_team,
                match.away_team,
                int(simulation.home_goals[simulation_index, match_index]),
                int(simulation.away_goals[simulation_index, match_index]),
            )
            for match_index, match in enumerate(simulation.matches)
        ]
        standings = compute_standings(teams, played_matches, remaining_results)
        for standing in standings[:n_qualifying_spots]:
            qualification_counts[standing.team] += 1
    return {team: count / simulation.n_simulations for team, count in qualification_counts.items()}


def simulate_final_positions(
    teams: list[str],
    played_matches: list[PlayedMatch],
    remaining_matches: list[MatchExpectation],
    n_simulations: int = 10_000,
    seed: int | None = None,
) -> dict[str, list[float]]:
    """How often each team finishes in each final position, 1st through last.

    Same machinery as `simulate_group_qualification` -- real played results
    plus Monte Carlo remaining results, ranked by `compute_standings` -- but
    it keeps the whole finishing distribution instead of collapsing it to
    "top N or not". A single qualification percentage can't distinguish a
    team that is reliably second from one that finishes either first or
    last, which is exactly the difference a group preview needs to show.

    Returns `{team: [P(1st), P(2nd), ...]}` with one entry per team in the
    group; each team's own list sums to 1.0 (every simulation places every
    team exactly once). Deliberately additive: `simulate_group_qualification`
    is untouched, so nothing already depending on it changes behaviour.
    """
    simulation = run_simulations(remaining_matches, n_simulations, seed)
    position_counts = {team: [0] * len(teams) for team in teams}
    for simulation_index in range(simulation.n_simulations):
        remaining_results = [
            (
                match.home_team,
                match.away_team,
                int(simulation.home_goals[simulation_index, match_index]),
                int(simulation.away_goals[simulation_index, match_index]),
            )
            for match_index, match in enumerate(simulation.matches)
        ]
        standings = compute_standings(teams, played_matches, remaining_results)
        for position, standing in enumerate(standings):
            position_counts[standing.team][position] += 1
    return {
        team: [count / simulation.n_simulations for count in counts]
        for team, counts in position_counts.items()
    }
