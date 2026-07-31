"""Fact -> Polish sentence dispatch."""

from __future__ import annotations

from football_ai.insights.schemas import Fact
from football_ai.narrative import templates_pl


class UnknownFactTypeError(ValueError):
    """Raised when no template is registered for a Fact's fact_type."""


_TEMPLATES = {
    "win_streak": templates_pl.win_streak,
    "unbeaten_streak": templates_pl.unbeaten_streak,
    "scoreless_streak": templates_pl.scoreless_streak,
    "clean_sheet_streak": templates_pl.clean_sheet_streak,
    "biggest_win": templates_pl.biggest_win,
    "longest_win_streak_ever": templates_pl.longest_win_streak_ever,
    "longest_unbeaten_streak_ever": templates_pl.longest_unbeaten_streak_ever,
    "match_count_milestone": templates_pl.match_count_milestone,
    "goal_count_milestone": templates_pl.goal_count_milestone,
    "player_goal_milestone": templates_pl.player_goal_milestone,
    "h2h_never_won": templates_pl.h2h_never_won,
    "h2h_first_win_since": templates_pl.h2h_first_win_since,
}


def render_fact(fact: Fact) -> str:
    """Render one Fact into a single Polish sentence.

    Raises rather than silently skipping an unregistered fact_type --
    insights/ is only ever expected to produce the 11 types
    `templates_pl.py` covers; a mismatch here means the two packages have
    drifted (a new insights/ fact type shipped without a matching
    template), which should fail loudly in development, not surface as a
    silently-missing sentence in a published preview.
    """
    template = _TEMPLATES.get(fact.fact_type)
    if template is None:
        raise UnknownFactTypeError(f"No narrative template registered for fact_type={fact.fact_type!r}.")
    return template(fact)


def render_facts(facts: list[Fact]) -> list[str]:
    """Render a ranked list of Facts (e.g. from `insights.detectors.facts_for_match`) into sentences."""
    return [render_fact(fact) for fact in facts]
