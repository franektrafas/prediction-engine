"""Reproduce a Fact from its own provenance and confirm it checks out.

PLAN_SILNIKA.md's other non-negotiable, alongside provenance itself: a fact
must be reproducible. This module doesn't re-implement each fact type's
logic in parallel (that would just be a second place for the same bug to
hide) -- it re-runs the exact same detector function that would have
produced the fact, using the fact's own `subject_team`/`related_teams`/
`as_of_match_id`, and checks the result matches. A fact that can't be
regenerated this way is not safe to publish, whatever value it claims.
"""

from __future__ import annotations

import pandas as pd

from football_ai.insights import h2h, milestones, records, streaks
from football_ai.insights.schemas import Fact


class UnverifiableFactTypeError(ValueError):
    """Raised when `verify` is given a fact_type no verifier is registered for."""


def _as_of_date(fact: Fact, records_frame: pd.DataFrame) -> pd.Timestamp | None:
    """Recover the as_of cutoff a fact was computed with, from its recorded match_id.

    `as_of_match_id` records the *last match included* in the fact (e.g. the
    milestone match itself) -- but every detector's own `as_of_date` cutoff
    is exclusive (`match_date < as_of_date`, see streaks.team_matches /
    h2h.h2h_matches), so recomputing with the match's own date would exclude
    that match and silently under-count. A one-second nudge forward makes
    the cutoff include it again, matching how the fact was originally
    produced without needing to know the *next* match's date.
    """
    if fact.as_of_match_id is None:
        return None
    match_row = records_frame.loc[records_frame["match_id"] == fact.as_of_match_id]
    if match_row.empty:
        return None
    return match_row.iloc[0]["match_date"] + pd.Timedelta(seconds=1)


def _venue(fact: Fact) -> str | None:
    return fact.data.get("venue")


_STREAK_RECOMPUTE = {
    "win_streak": streaks.win_streak,
    "unbeaten_streak": streaks.unbeaten_streak,
    "scoreless_streak": streaks.scoreless_streak,
    "clean_sheet_streak": streaks.clean_sheet_streak,
}

_RECORD_RECOMPUTE = {
    "biggest_win": records.biggest_win,
    "longest_win_streak_ever": records.longest_win_streak_ever,
    "longest_unbeaten_streak_ever": records.longest_unbeaten_streak_ever,
}

_MILESTONE_RECOMPUTE = {
    "match_count_milestone": milestones.match_count_milestone,
    "goal_count_milestone": milestones.goal_count_milestone,
}


def verify(fact: Fact, records_frame: pd.DataFrame) -> bool:
    """Recompute `fact` from scratch and confirm the result matches exactly.

    Matches on `provenance` (the actual match_ids the claim rests on) --
    stricter than comparing `summary` strings, which could coincidentally
    match while the underlying evidence differs.
    """
    if not fact.provenance:
        return False
    provenance_matches = records_frame.loc[records_frame["match_id"].isin(fact.provenance)]
    if len(provenance_matches) != len(set(fact.provenance)):
        return False  # a claimed match_id doesn't exist in this records_frame

    as_of_date = _as_of_date(fact, records_frame)

    if fact.fact_type in _STREAK_RECOMPUTE:
        recomputed = _STREAK_RECOMPUTE[fact.fact_type](
            records_frame, fact.subject_team, as_of_date, venue=_venue(fact)
        )
    elif fact.fact_type in _RECORD_RECOMPUTE:
        recomputed = _RECORD_RECOMPUTE[fact.fact_type](records_frame, fact.subject_team, as_of_date)
    elif fact.fact_type in _MILESTONE_RECOMPUTE:
        recomputed = _MILESTONE_RECOMPUTE[fact.fact_type](records_frame, fact.subject_team, as_of_date)
    elif fact.fact_type in ("h2h_first_win_since", "h2h_never_won"):
        if not fact.related_teams:
            return False
        recomputed = h2h.first_win_since(records_frame, fact.subject_team, fact.related_teams[0], as_of_date)
    else:
        raise UnverifiableFactTypeError(f"No verifier registered for fact_type={fact.fact_type!r}")

    if recomputed is None:
        return False
    return recomputed.fact_type == fact.fact_type and recomputed.provenance == fact.provenance


def verify_all(facts: list[Fact], records_frame: pd.DataFrame) -> list[Fact]:
    """Return only the facts that pass `verify` -- a safety net before anything gets published."""
    return [fact for fact in facts if verify(fact, records_frame)]
