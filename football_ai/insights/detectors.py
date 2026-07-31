"""Gather every fact type for a match and rank by unusualness, instead of returning all of them.

PLAN_SILNIKA.md's own completion criterion for Krok 2: "detectors.py potrafi
uszeregować fakty według nietypowości, a nie zwracać wszystkiego." Every
submodule already scores its own `Fact.weight` on a comparable ~0-1 scale
(see each module's docstrings) -- this module's only real job is collecting
across all of them and sorting, not re-deriving unusualness itself.
"""

from __future__ import annotations

import pandas as pd

from football_ai.insights import h2h, milestones, player_milestones, records, streaks
from football_ai.insights.schemas import Fact


def facts_for_team(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    goalscorers_frame: pd.DataFrame | None = None,
) -> list[Fact]:
    """Every fact type this package can produce for a single team, unranked.

    `goalscorers_frame` is optional (defaults to `None`, matching this
    function's pre-existing signature -- deliberately backward compatible,
    every prior caller keeps working unchanged) -- when given, also scans
    for player-level goal milestones via `player_milestones`, which needs
    a match_id-linked goalscorers frame (`InternationalResultsSource.
    normalize_goalscorers()`), not `records_frame` alone.
    """
    facts: list[Fact] = []
    facts.extend(streaks.all_current_streaks(records_frame, team_name, as_of_date))
    for fact_fn in (records.biggest_win, records.longest_win_streak_ever, records.longest_unbeaten_streak_ever):
        fact = fact_fn(records_frame, team_name, as_of_date)
        if fact is not None:
            facts.append(fact)
    for fact_fn in (milestones.match_count_milestone, milestones.goal_count_milestone):
        fact = fact_fn(records_frame, team_name, as_of_date)
        if fact is not None:
            facts.append(fact)
    if goalscorers_frame is not None:
        facts.extend(player_milestones.player_goal_milestones_for_team(goalscorers_frame, team_name, as_of_date))
    return facts


def facts_for_match(
    records_frame: pd.DataFrame,
    home_team: str,
    away_team: str,
    as_of_date: pd.Timestamp | None = None,
    top_n: int | None = 10,
    goalscorers_frame: pd.DataFrame | None = None,
) -> list[Fact]:
    """Every fact for both sides of an upcoming/played match, plus head-to-head, ranked by weight.

    `as_of_date` should be the match's own date for a pre-match preview --
    every submodule already respects it as a leak-safety cutoff (no
    submodule here computes anything from data at or after that date), the
    same principle the ML feature pipeline uses. `top_n=None` returns every
    fact, unranked-but-sorted; the default of 10 matches this being a
    "don't return everything" ranking layer, not a full dump.
    """
    facts = facts_for_team(records_frame, home_team, as_of_date, goalscorers_frame)
    facts.extend(facts_for_team(records_frame, away_team, as_of_date, goalscorers_frame))

    home_h2h = h2h.first_win_since(records_frame, home_team, away_team, as_of_date)
    if home_h2h is not None:
        facts.append(home_h2h)
    away_h2h = h2h.first_win_since(records_frame, away_team, home_team, as_of_date)
    if away_h2h is not None:
        facts.append(away_h2h)

    ranked = sorted(facts, key=lambda fact: fact.weight, reverse=True)
    return ranked if top_n is None else ranked[:top_n]
