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

Known, deliberately unaddressed gap (confirmed 2026-07-30): the default
inference dataset's `home_elo`/`away_elo` columns are a real Elo merge only
for roughly 2018-2024; essentially every 2025/2026 match (i.e. every team's
most recent real form going into these Nations League fixtures) falls back
to a flat 1509.0/1483.0 default -- the SAME number regardless of which team
is playing, worse than useless if surfaced as if it were real. This module
deliberately leaves `home_elo`/`away_elo` unset on the prediction payload
(`PredictionService`'s own 1500.0/1500.0 fallback applies to both sides
equally) rather than wiring up a lookup that would silently present that
stale default as real data. Fixing the Elo merge is engine-tuning work,
explicitly paused by PLAN_SILNIKA.md's 2026-07-30 pivot.
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
) -> SimulationInput:
    """Split a group's real fixtures into played results and model-predicted remaining matches.

    A row counts as played when both `home_score`/`away_score` are present
    -- the same not-yet-played signal `uefa_nations_league.py` already
    produces (`pd.NA` until Wikipedia records a real result). Remaining
    matches get their expected goals from a real `prediction_service.predict()`
    call, not a placeholder -- clamped to a small positive floor since
    `MatchExpectation`'s Poisson lambda must be strictly positive and a
    model is free to predict exactly (or near) 0.
    """
    teams = sorted(set(fixtures["home_team_name"]) | set(fixtures["away_team_name"]))
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
