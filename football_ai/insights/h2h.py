"""Head-to-head history between two teams."""

from __future__ import annotations

import pandas as pd

from football_ai.data.normalization import canonicalize_team_name
from football_ai.insights.schemas import Fact


def h2h_matches(
    records_frame: pd.DataFrame,
    team_a: str,
    team_b: str,
    as_of_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Every match between two teams, oldest first, each row carrying team_a's perspective.

    Adds `team_a_goals`/`team_b_goals`/`team_a_result` (`"win"`/`"draw"`/
    `"loss"`, from team_a's side) regardless of which side was home in any
    given meeting -- head-to-head history spans decades, and which team
    happened to host a given fixture is incidental to "have they ever won".
    """
    canon_a = canonicalize_team_name(team_a)
    canon_b = canonicalize_team_name(team_b)
    a_home = (records_frame["home_team_name"] == canon_a) & (records_frame["away_team_name"] == canon_b)
    b_home = (records_frame["home_team_name"] == canon_b) & (records_frame["away_team_name"] == canon_a)
    matches = records_frame.loc[a_home | b_home].sort_values("match_date").copy()
    if as_of_date is not None:
        matches = matches.loc[matches["match_date"] < as_of_date]
    matches["team_a_is_home"] = matches["home_team_name"] == canon_a
    matches["team_a_goals"] = matches.apply(
        lambda row: row["home_score"] if row["team_a_is_home"] else row["away_score"], axis=1
    )
    matches["team_b_goals"] = matches.apply(
        lambda row: row["away_score"] if row["team_a_is_home"] else row["home_score"], axis=1
    )

    def _result(row: pd.Series) -> str:
        if row["team_a_goals"] > row["team_b_goals"]:
            return "win"
        if row["team_a_goals"] < row["team_b_goals"]:
            return "loss"
        return "draw"

    matches["team_a_result"] = matches.apply(_result, axis=1)
    return matches


def first_win_since(
    records_frame: pd.DataFrame,
    team_a: str,
    team_b: str,
    as_of_date: pd.Timestamp | None = None,
) -> Fact | None:
    """"team_a's first win against team_b since <year>" -- or "team_a has never beaten team_b".

    Finds team_a's most recent win in the head-to-head record and reports
    how long the *previous* gap without a win was -- e.g. Poland's most
    recent win against Germany was 2014-10-11, and the win before that (if
    any) tells you "first win since <that year>". If team_a has exactly one
    win on record, the fact is "first ever win" (no prior win to gap
    against). If team_a has never won, returns a `"never_won"` fact instead
    of None -- that's a genuine, notable, provenance-backed fact in its own
    right (PLAN_SILNIKA.md's own worked example), not an absence of one.
    """
    canon_a = canonicalize_team_name(team_a)
    canon_b = canonicalize_team_name(team_b)
    matches = h2h_matches(records_frame, team_a, team_b, as_of_date)
    if matches.empty:
        return None

    wins = matches.loc[matches["team_a_result"] == "win"]
    if wins.empty:
        first_meeting_year = int(matches.iloc[0]["match_date"].year)
        return Fact(
            fact_type="h2h_never_won",
            subject_team=canon_a,
            summary=f"{canon_a} has never beaten {canon_b} in {len(matches)} meetings since {first_meeting_year}",
            data={"opponent": canon_b, "meeting_count": len(matches)},
            weight=min(1.0, 0.1 + len(matches) * 0.03),
            provenance=tuple(int(match_id) for match_id in matches["match_id"]),
            related_teams=(canon_b,),
        )

    most_recent_win = wins.iloc[-1]
    prior_wins = wins.iloc[:-1]
    if prior_wins.empty:
        gap_summary = f"{canon_a}'s only win against {canon_b} on record came in {int(most_recent_win['match_date'].year)}"
        provenance = tuple(int(match_id) for match_id in matches["match_id"])
    else:
        previous_win_year = int(prior_wins.iloc[-1]["match_date"].year)
        most_recent_win_year = int(most_recent_win["match_date"].year)
        gap_summary = (
            f"{canon_a}'s {most_recent_win_year} win against {canon_b} was their first since {previous_win_year}"
        )
        # Provenance: the two bracketing wins plus every match in between --
        # reproducible end to end, not just "trust the gap number".
        gap_matches = matches.loc[
            (matches["match_date"] >= prior_wins.iloc[-1]["match_date"]) & (matches["match_date"] <= most_recent_win["match_date"])
        ]
        provenance = tuple(int(match_id) for match_id in gap_matches["match_id"])

    gap_years = (
        int(most_recent_win["match_date"].year) - int(prior_wins.iloc[-1]["match_date"].year)
        if not prior_wins.empty
        else None
    )
    return Fact(
        fact_type="h2h_first_win_since",
        subject_team=canon_a,
        summary=gap_summary,
        data={
            "opponent": canon_b,
            "win_year": int(most_recent_win["match_date"].year),
            "gap_years": gap_years,
        },
        weight=min(1.0, 0.2 + (gap_years or 0) * 0.03),
        provenance=provenance,
        related_teams=(canon_b,),
    )
