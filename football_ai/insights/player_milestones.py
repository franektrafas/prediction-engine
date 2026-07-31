"""Player-level goal milestones (Nth international goal), from martj42's goalscorers.csv.

Deliberately scoped to GOAL milestones only, not appearance/cap milestones
("the manager's 50th goal" example in PLAN_SILNIKA.md really means a
player's 50th goal, not a manager's -- managers don't score) -- flagged
here, not silently pretended away: goalscorers.csv records who scored, not
who played, so a player's career match-count ("Nth cap") isn't computable
from this file; that would need real historical lineup/squad data this
project doesn't have.

Deferred at Krok 2 time (2026-07-30), built once the rest of PLAN_SILNIKA.md's
vertical (Krok 1-5) was done -- see `scrapers.sources.international_results.
InternationalResultsSource.normalize_goalscorers()` for how the raw
goalscorers.csv gets linked to `records_frame`'s `match_id` (verified for
real: 100% of 47,914 raw goal events matched, no fuzzy-matching needed).
"""

from __future__ import annotations

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.schemas import Fact


_DEFAULT_ROUND_NUMBERS = tuple(range(10, 151, 10))


def player_goals(
    goalscorers_frame: pd.DataFrame,
    team_name: str,
    player_name: str,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every real international goal `player_name` scored for `team_name`, oldest first.

    Own goals are excluded -- conventionally never credited to the scoring
    player's own career tally. `as_of_date` excludes goals on or after that
    date, the same leak-safety cutoff every other insights/ function uses.
    """
    canonical_team = canonicalize_team_name(team_name)
    mask = (
        (goalscorers_frame["team_name"] == canonical_team)
        & (goalscorers_frame["scorer"] == player_name)
        & (~goalscorers_frame["own_goal"])
    )
    goals = goalscorers_frame.loc[mask].sort_values("match_date")
    if as_of_date is not None:
        goals = goals.loc[goals["match_date"] < as_of_date]
    return goals


def player_goal_milestone(
    goalscorers_frame: pd.DataFrame,
    team_name: str,
    player_name: str,
    as_of_date: pd.Timestamp | None = None,
    round_numbers: tuple[int, ...] = _DEFAULT_ROUND_NUMBERS,
) -> Fact | None:
    """A Fact if `player_name`'s most recent goal for `team_name` (as of the cutoff) was exactly a round number.

    Same "the exact match, not a forecast" principle as
    `milestones.match_count_milestone`/`goal_count_milestone`: only
    surfaces the milestone if it's genuinely the player's most recent
    scored goal, not "approaching" a round number. Provenance is every
    one of the player's goals for this team up to and including the
    milestone one -- a career-goals claim can't be reproduced from one
    row alone, same reasoning as the team-level version.
    """
    goals = player_goals(goalscorers_frame, team_name, player_name, as_of_date)
    if goals.empty:
        return None
    count = len(goals)
    if count not in round_numbers:
        return None
    milestone_goal = goals.iloc[-1]
    canonical_team = canonicalize_team_name(team_name)
    opponent = (
        milestone_goal["away_team_name"]
        if milestone_goal["home_team_name"] == canonical_team
        else milestone_goal["home_team_name"]
    )
    return Fact(
        fact_type="player_goal_milestone",
        subject_team=canonical_team,
        summary=(
            f"{player_name} scored their {count}th international goal for {canonical_team} "
            f"against {opponent} on {milestone_goal['match_date'].date()}"
        ),
        data={"player_name": player_name, "milestone_number": count, "opponent": opponent},
        weight=0.4,
        provenance=tuple(int(match_id) for match_id in goals["match_id"]),
        as_of_match_id=int(milestone_goal["match_id"]),
        related_teams=(opponent,),
    )


def player_goal_milestones_for_team(
    goalscorers_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    round_numbers: tuple[int, ...] = _DEFAULT_ROUND_NUMBERS,
) -> list[Fact]:
    """Every player-goal-milestone Fact for a team, scanning every player who's ever scored for it.

    The natural entry point for "which of this team's players are at a
    personal milestone as of this match" -- a caller doesn't need to
    already know which player to check.
    """
    canonical_team = canonicalize_team_name(team_name)
    team_goals = goalscorers_frame.loc[
        (goalscorers_frame["team_name"] == canonical_team) & (~goalscorers_frame["own_goal"])
    ]
    if as_of_date is not None:
        team_goals = team_goals.loc[team_goals["match_date"] < as_of_date]
    facts: list[Fact] = []
    for player_name in team_goals["scorer"].unique():
        fact = player_goal_milestone(goalscorers_frame, team_name, player_name, as_of_date, round_numbers)
        if fact is not None:
            facts.append(fact)
    return facts
