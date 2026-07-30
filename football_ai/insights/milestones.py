"""Round-number team milestones (Nth match, Nth win, Nth goal scored).

Scoped to team-level milestones computable from `records_frame` alone.
PLAN_SILNIKA.md's own example ("50. gol selekcjonera" -- a manager's 50th
goal) is a *player/manager*-level milestone, which needs martj42's
`goalscorers.csv` (fetched by `InternationalResultsSource.fetch_goalscorers()`
already, but not yet persisted or merged anywhere) -- deliberately deferred,
flagged here rather than silently skipped, not built into this pass.
"""

from __future__ import annotations

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.schemas import Fact
from football_ai.insights.streaks import team_matches

_DEFAULT_ROUND_NUMBERS = tuple(range(50, 1001, 50))


def match_count_milestone(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    round_numbers: tuple[int, ...] = _DEFAULT_ROUND_NUMBERS,
) -> Fact | None:
    """A Fact if the team's most recent match (as of the cutoff) was exactly a round-number match.

    E.g. if a team's 100th-ever international was its last match before
    `as_of_date`, this returns a Fact about it -- not "the team is
    approaching its 100th", only the exact match itself, so the provenance
    is unambiguous (the single match, not a guess about which future match
    will be the 100th).
    """
    matches = team_matches(records_frame, team_name, as_of_date)
    if matches.empty:
        return None
    count = len(matches)
    if count not in round_numbers:
        return None
    milestone_match = matches.iloc[-1]
    canonical = canonicalize_team_name(team_name)
    return Fact(
        fact_type="match_count_milestone",
        subject_team=canonical,
        summary=(
            f"{canonical}'s {count}th international match was against "
            f"{milestone_match['opponent']} on {milestone_match['match_date'].date()}"
        ),
        data={"milestone_number": count, "opponent": milestone_match["opponent"]},
        weight=0.3,
        provenance=(int(milestone_match["match_id"]),),
        as_of_match_id=int(milestone_match["match_id"]),
    )


def goal_count_milestone(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    round_numbers: tuple[int, ...] = tuple(range(100, 2001, 100)),
) -> Fact | None:
    """A Fact if the team's cumulative career goal tally crossed a round number in its most recent match.

    Provenance is every match up to and including the milestone one (the
    whole running total the claim depends on), not just the final match --
    a career-goals claim can't be reproduced from one row alone.
    """
    matches = team_matches(records_frame, team_name, as_of_date)
    if matches.empty:
        return None
    cumulative_goals = matches["goals_for"].cumsum()
    milestone_index = None
    milestone_total = None
    for number in round_numbers:
        hits = cumulative_goals[(cumulative_goals >= number) & (cumulative_goals.shift(1).fillna(0) < number)]
        if not hits.empty:
            milestone_index = hits.index[-1]
            milestone_total = number
    if milestone_index is None or milestone_index != cumulative_goals.index[-1]:
        # Only surface it if the milestone-crossing match IS the most recent
        # match as of the cutoff -- same "the exact match, not a forecast"
        # principle as match_count_milestone.
        return None
    canonical = canonicalize_team_name(team_name)
    up_to_milestone = matches.loc[: milestone_index]
    milestone_match = matches.loc[milestone_index]
    return Fact(
        fact_type="goal_count_milestone",
        subject_team=canonical,
        summary=(
            f"{canonical} passed {milestone_total} career goals in their match against "
            f"{milestone_match['opponent']} on {milestone_match['match_date'].date()}"
        ),
        data={"milestone_number": milestone_total, "opponent": milestone_match["opponent"]},
        weight=0.35,
        provenance=tuple(int(match_id) for match_id in up_to_milestone["match_id"]),
        as_of_match_id=int(milestone_match["match_id"]),
    )
