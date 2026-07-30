"""Current win/unbeaten/goalless streaks for a team, as of a point in time."""

from __future__ import annotations

from typing import Callable

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.schemas import Fact


def _team_matches(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    venue: str | None = None,
) -> pd.DataFrame:
    """Return a team's matches, oldest first, each row carrying that team's goals_for/goals_against.

    `venue`, if given, restricts to `"home"` or `"away"` only -- the
    home/away split PLAN_SILNIKA.md's Krok 2 outline calls for. `as_of_date`
    excludes matches on or after that date -- the point-in-time cutoff every
    caller generating a pre-match preview must use, so a streak can never
    include the match it's meant to preview.
    """
    canonical = canonicalize_team_name(team_name)
    is_home = records_frame["home_team_name"] == canonical
    is_away = records_frame["away_team_name"] == canonical
    if venue == "home":
        mask = is_home
    elif venue == "away":
        mask = is_away
    elif venue is None:
        mask = is_home | is_away
    else:
        raise ValueError(f"venue must be 'home', 'away', or None, got {venue!r}")

    matches = records_frame.loc[mask].sort_values("match_date")
    if as_of_date is not None:
        matches = matches.loc[matches["match_date"] < as_of_date]
    matches = matches.copy()
    matches["is_home"] = matches["home_team_name"] == canonical
    matches["goals_for"] = matches.apply(
        lambda row: row["home_score"] if row["is_home"] else row["away_score"], axis=1
    )
    matches["goals_against"] = matches.apply(
        lambda row: row["away_score"] if row["is_home"] else row["home_score"], axis=1
    )
    matches["opponent"] = matches.apply(
        lambda row: row["away_team_name"] if row["is_home"] else row["home_team_name"], axis=1
    )
    return matches


def _current_streak_length(matches: pd.DataFrame, holds: Callable[[pd.Series], bool]) -> tuple[int, list[int]]:
    """Walk backward from the most recent match while `holds(row)` stays true.

    Returns `(streak_length, match_ids)` -- `match_ids` in chronological
    (oldest-of-the-streak-first) order, ready to use as a Fact's provenance.
    """
    length = 0
    match_ids: list[int] = []
    for _, row in matches.iloc[::-1].iterrows():
        if not holds(row):
            break
        length += 1
        match_ids.append(int(row["match_id"]))
    match_ids.reverse()
    return length, match_ids


def _streak_fact(
    fact_type: str,
    team_name: str,
    matches: pd.DataFrame,
    holds: Callable[[pd.Series], bool],
    describe: Callable[[int], str],
    weight_per_match: float,
    venue: str | None,
) -> Fact | None:
    """Shared plumbing for every streak type below -- build a Fact if the streak is non-zero."""
    length, match_ids = _current_streak_length(matches, holds)
    if length == 0:
        return None
    canonical = canonicalize_team_name(team_name)
    return Fact(
        fact_type=fact_type,
        subject_team=canonical,
        summary=describe(length),
        data={"streak_length": length, "venue": venue},
        weight=min(1.0, length * weight_per_match),
        provenance=tuple(match_ids),
    )


def win_streak(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    venue: str | None = None,
) -> Fact | None:
    """Current consecutive-wins streak, or None if the team didn't win its most recent match."""
    matches = _team_matches(records_frame, team_name, as_of_date, venue)
    return _streak_fact(
        fact_type="win_streak",
        team_name=team_name,
        matches=matches,
        holds=lambda row: row["goals_for"] > row["goals_against"],
        describe=lambda n: f"{canonicalize_team_name(team_name)} has won {n} consecutive match{'es' if n != 1 else ''}"
        + (f" ({venue})" if venue else ""),
        weight_per_match=0.12,
        venue=venue,
    )


def unbeaten_streak(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    venue: str | None = None,
) -> Fact | None:
    """Current consecutive-without-a-loss streak (wins + draws)."""
    matches = _team_matches(records_frame, team_name, as_of_date, venue)
    return _streak_fact(
        fact_type="unbeaten_streak",
        team_name=team_name,
        matches=matches,
        holds=lambda row: row["goals_for"] >= row["goals_against"],
        describe=lambda n: f"{canonicalize_team_name(team_name)} is unbeaten in {n} consecutive match{'es' if n != 1 else ''}"
        + (f" ({venue})" if venue else ""),
        weight_per_match=0.06,
        venue=venue,
    )


def scoreless_streak(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    venue: str | None = None,
) -> Fact | None:
    """Current consecutive-matches-without-scoring streak."""
    matches = _team_matches(records_frame, team_name, as_of_date, venue)
    return _streak_fact(
        fact_type="scoreless_streak",
        team_name=team_name,
        matches=matches,
        holds=lambda row: row["goals_for"] == 0,
        describe=lambda n: f"{canonicalize_team_name(team_name)} has failed to score in {n} consecutive match{'es' if n != 1 else ''}"
        + (f" ({venue})" if venue else ""),
        weight_per_match=0.15,
        venue=venue,
    )


def clean_sheet_streak(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
    venue: str | None = None,
) -> Fact | None:
    """Current consecutive-matches-without-conceding streak."""
    matches = _team_matches(records_frame, team_name, as_of_date, venue)
    return _streak_fact(
        fact_type="clean_sheet_streak",
        team_name=team_name,
        matches=matches,
        holds=lambda row: row["goals_against"] == 0,
        describe=lambda n: f"{canonicalize_team_name(team_name)} has kept a clean sheet in {n} consecutive match{'es' if n != 1 else ''}"
        + (f" ({venue})" if venue else ""),
        weight_per_match=0.12,
        venue=venue,
    )


def all_current_streaks(
    records_frame: pd.DataFrame,
    team_name: str,
    as_of_date: pd.Timestamp | None = None,
) -> list[Fact]:
    """Every non-zero streak fact for a team (overall, not venue-split)."""
    facts = [
        win_streak(records_frame, team_name, as_of_date),
        unbeaten_streak(records_frame, team_name, as_of_date),
        scoreless_streak(records_frame, team_name, as_of_date),
        clean_sheet_streak(records_frame, team_name, as_of_date),
    ]
    return [fact for fact in facts if fact is not None]
