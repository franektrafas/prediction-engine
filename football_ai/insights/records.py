"""All-time records for a team: biggest win, longest streak ever."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.schemas import Fact
from football_ai.insights.streaks import team_matches


def biggest_win(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
) -> Fact | None:
    """A team's largest margin of victory on record, or None if they've never won."""
    matches = team_matches(records_frame, team_name, as_of_date)
    wins = matches.loc[matches["goals_for"] > matches["goals_against"]]
    if wins.empty:
        return None
    margins = wins["goals_for"] - wins["goals_against"]
    best = wins.loc[margins.idxmax()]
    canonical = canonicalize_team_name(team_name)
    return Fact(
        fact_type="biggest_win",
        subject_team=canonical,
        summary=(
            f"{canonical}'s biggest ever win was {int(best['goals_for'])}-{int(best['goals_against'])} "
            f"against {best['opponent']} ({best['match_date'].date()})"
        ),
        data={
            "opponent": best["opponent"],
            "goals_for": int(best["goals_for"]),
            "goals_against": int(best["goals_against"]),
            "margin": int(best["goals_for"] - best["goals_against"]),
            "date": str(best["match_date"].date()),
        },
        weight=min(1.0, 0.2 + (best["goals_for"] - best["goals_against"]) * 0.08),
        provenance=(int(best["match_id"]),),
        related_teams=(best["opponent"],),
    )


def _longest_run(matches: pd.DataFrame, holds: Callable[[pd.Series], bool]) -> tuple[int, list[int]]:
    """Scan the full match history (not just the trailing run) for the longest streak satisfying `holds`.

    Unlike `streaks._current_streak_length` (which only looks at the tail,
    for "is this team on a streak right now"), records need the longest run
    *anywhere* in history -- a full forward scan, keeping whichever run
    turns out longest, with ties broken by whichever occurred first.
    """
    best_length = 0
    best_match_ids: list[int] = []
    current_length = 0
    current_match_ids: list[int] = []
    for _, row in matches.iterrows():
        if holds(row):
            current_length += 1
            current_match_ids.append(int(row["match_id"]))
        else:
            current_length = 0
            current_match_ids = []
        if current_length > best_length:
            best_length = current_length
            best_match_ids = list(current_match_ids)
    return best_length, best_match_ids


def longest_win_streak_ever(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
) -> Fact | None:
    """The longest consecutive-wins run in a team's entire history."""
    matches = team_matches(records_frame, team_name, as_of_date)
    length, match_ids = _longest_run(matches, lambda row: row["goals_for"] > row["goals_against"])
    if length == 0:
        return None
    canonical = canonicalize_team_name(team_name)
    first_match = records_frame.loc[records_frame["match_id"] == match_ids[0]].iloc[0]
    last_match = records_frame.loc[records_frame["match_id"] == match_ids[-1]].iloc[0]
    return Fact(
        fact_type="longest_win_streak_ever",
        subject_team=canonical,
        summary=(
            f"{canonical}'s longest ever winning run is {length} consecutive matches "
            f"({first_match['match_date'].date()} - {last_match['match_date'].date()})"
        ),
        data={"streak_length": length},
        weight=min(1.0, length * 0.1),
        provenance=tuple(match_ids),
    )


def longest_unbeaten_streak_ever(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
) -> Fact | None:
    """The longest consecutive-without-a-loss run in a team's entire history."""
    matches = team_matches(records_frame, team_name, as_of_date)
    length, match_ids = _longest_run(matches, lambda row: row["goals_for"] >= row["goals_against"])
    if length == 0:
        return None
    canonical = canonicalize_team_name(team_name)
    first_match = records_frame.loc[records_frame["match_id"] == match_ids[0]].iloc[0]
    last_match = records_frame.loc[records_frame["match_id"] == match_ids[-1]].iloc[0]
    return Fact(
        fact_type="longest_unbeaten_streak_ever",
        subject_team=canonical,
        summary=(
            f"{canonical}'s longest ever unbeaten run is {length} consecutive matches "
            f"({first_match['match_date'].date()} - {last_match['match_date'].date()})"
        ),
        data={"streak_length": length},
        weight=min(1.0, length * 0.05),
        provenance=tuple(match_ids),
    )
