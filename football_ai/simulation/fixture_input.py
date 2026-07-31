"""Bridges a group's real fixtures + live model predictions into simulation input.

`monte_carlo.py`/`competition.py` deliberately don't import
`predictions.service` (see their own docstrings) so they stay testable and
reusable without a trained model or network access. This module is the one
place allowed to depend on both -- it closes the gap PLAN_SILNIKA.md's Krok
3 flagged as open: turning a real group's real remaining fixtures into
`MatchExpectation` objects via real model output, instead of hand-written
expected goals.

Works with any wide-schema fixtures frame (`home_team_name`/
`away_team_name`/`match_date`/`home_score`/`away_score`, optionally
`stadium`) -- `UefaNationsLeagueSource.fetch_group_matches()` is the first
caller, not the only one this is built for.

Elo gap, closed 2026-07-31: the *static training file*'s `home_elo`/
`away_elo` columns are a real merge only for roughly 2018-2024 (every
2025/2026 row falls back to a flat 1509.0/1483.0 default) -- but that's a
training-data problem, not a live-fixture one. For a genuinely *future*
match (the only kind this module ever builds a `MatchExpectation` for),
there's no leak-safety concern in using today's real, current rating --
unlike a historical row, "today's rating" is the only pre-match Elo that
could ever exist for a not-yet-played fixture. `elo_ratings` (optional,
built via `WorldFootballEloSource.fetch_current_snapshot()` +
`.latest_ratings_by_team()`) plugs that real, live rating straight into
each remaining match's prediction payload. Falls back to `None` per team
when omitted or when a team isn't in the snapshot (e.g. a fetch failure,
or a team eloratings.net doesn't track) -- `PredictionService`'s own
1500.0/1500.0 fallback applies in that case, same as before this existed.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.predictions.schemas import MatchContextPayload
from football_ai.predictions.service import PredictionService
from football_ai.simulation.competition import PlayedMatch
from football_ai.simulation.monte_carlo import MatchExpectation


_EXPECTED_GOALS_FLOOR = 0.05


@dataclass(slots=True, frozen=True)
class SimulationInput:
    """Everything `simulate_group_qualification()` needs, built from one group's real fixtures."""

    teams: list[str]
    played_matches: list[PlayedMatch]
    remaining_matches: list[MatchExpectation]


def build_simulation_input(
    fixtures: pd.DataFrame,
    prediction_service: PredictionService,
    tournament: str = "UEFA Nations League",
    elo_ratings: dict[str, float] | None = None,
) -> SimulationInput:
    """Split a group's real fixtures into played results and model-predicted remaining matches.

    A row counts as played when both `home_score`/`away_score` are present
    -- the same not-yet-played signal `uefa_nations_league.py` already
    produces (`pd.NA` until Wikipedia records a real result). Remaining
    matches get their expected goals from a real `prediction_service.predict()`
    call, not a placeholder -- clamped to a small positive floor since
    `MatchExpectation`'s Poisson lambda must be strictly positive and a
    model is free to predict exactly (or near) 0.

    `elo_ratings` (see module docstring) supplies real, current Elo per
    team when available; a team missing from it gets `None`, letting
    `PredictionService`'s own default fallback apply -- never a guess.
    """
    teams = sorted(set(fixtures["home_team_name"]) | set(fixtures["away_team_name"]))
    ratings = elo_ratings or {}
    played_matches: list[PlayedMatch] = []
    remaining_matches: list[MatchExpectation] = []
    for row in fixtures.itertuples(index=False):
        if pd.notna(row.home_score) and pd.notna(row.away_score):
            played_matches.append(
                PlayedMatch(row.home_team_name, row.away_team_name, int(row.home_score), int(row.away_score))
            )
            continue
        stadium_name = getattr(row, "stadium", None)
        payload = MatchContextPayload(
            match_date=row.match_date,
            home_team_name=row.home_team_name,
            away_team_name=row.away_team_name,
            tournament=tournament,
            stadium_name=stadium_name or None,
            home_elo=ratings.get(row.home_team_name),
            away_elo=ratings.get(row.away_team_name),
        )
        prediction = prediction_service.predict(payload)
        remaining_matches.append(
            MatchExpectation(
                home_team=row.home_team_name,
                away_team=row.away_team_name,
                expected_home_goals=max(_EXPECTED_GOALS_FLOOR, prediction.goals["home"]),
                expected_away_goals=max(_EXPECTED_GOALS_FLOOR, prediction.goals["away"]),
            )
        )
    return SimulationInput(teams=teams, played_matches=played_matches, remaining_matches=remaining_matches)
