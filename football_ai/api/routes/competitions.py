"""Krok 5 of PLAN_SILNIKA.md: /v1/competitions/outlook.

Real remaining fixtures (Wikipedia, `scrapers.sources.uefa_nations_league`)
+ real model predictions feed the Monte Carlo simulation via
`simulation.fixture_input.build_simulation_input()` -- the bridge Krok 3
originally flagged as an open external dependency and closed on
2026-07-30.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from football_ai.api.routes.auth import require_api_key
from football_ai.api.schemas import CompetitionOutlookResponse, GroupStandingPayload
from football_ai.predictions.service import PredictionService
from football_ai.scrapers.sources.uefa_nations_league import UefaNationsLeagueSource
from football_ai.simulation.competition import compute_standings, simulate_group_qualification
from football_ai.simulation.fixture_input import build_simulation_input


router = APIRouter(prefix="/v1/competitions", tags=["competitions"], dependencies=[Depends(require_api_key)])

_prediction_service = PredictionService.create()
_uefa_source = UefaNationsLeagueSource()


@router.get("/outlook", response_model=CompetitionOutlookResponse)
def competition_outlook(
    division: str,
    group: int,
    season: str = "2026–27",
    n_qualifying_spots: int = 1,
    n_simulations: int = 10_000,
) -> CompetitionOutlookResponse:
    """Qualification-probability outlook for one UEFA Nations League division/group.

    `n_qualifying_spots` defaults to 1 (group winner) -- deliberately not
    hardcoded to any specific competition's real promotion/relegation
    rules, since `simulation.competition.simulate_group_qualification`
    itself is competition-agnostic; a caller who wants "top 2 advance"
    passes `n_qualifying_spots=2`.
    """
    try:
        fixtures = _uefa_source.fetch_group_matches(division, group, season=season)
    except ValueError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    if fixtures.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No fixtures found for {season} UEFA Nations League {division}, group {group}.",
        )

    simulation_input = build_simulation_input(fixtures, _prediction_service)
    standings_so_far = compute_standings(simulation_input.teams, simulation_input.played_matches, [])
    probabilities = simulate_group_qualification(
        simulation_input.teams,
        simulation_input.played_matches,
        simulation_input.remaining_matches,
        n_qualifying_spots=n_qualifying_spots,
        n_simulations=n_simulations,
    )

    return CompetitionOutlookResponse(
        division=division,
        group_number=group,
        season=season,
        teams=simulation_input.teams,
        standings_so_far=[
            GroupStandingPayload(
                team=standing.team,
                points=standing.points,
                goal_difference=standing.goal_difference,
                goals_for=standing.goals_for,
                goals_against=standing.goals_against,
                wins=standing.wins,
                draws=standing.draws,
                losses=standing.losses,
            )
            for standing in standings_so_far
        ],
        qualification_probabilities=probabilities,
        n_qualifying_spots=n_qualifying_spots,
        n_simulations=n_simulations,
        remaining_matches=len(simulation_input.remaining_matches),
    )
