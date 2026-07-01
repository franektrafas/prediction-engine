"""Normalization utilities shared across data sources."""

from __future__ import annotations

import re

from football_ai.config.team_aliases import TEAM_ALIASES


def normalize_text(value: object | None) -> str:
    """Normalize a text field for canonical matching."""
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    collapsed = re.sub(r"\s+", " ", value.strip().lower())
    return collapsed


def canonicalize_team_name(team_name: object | None) -> str:
    """Map team aliases to a canonical national team name."""
    normalized = normalize_text(team_name)
    if not normalized:
        return ""
    if not isinstance(team_name, str):
        team_name = str(team_name)
    return TEAM_ALIASES.get(normalized, team_name.strip())


def compute_match_outcome(home_score: int | float, away_score: int | float) -> str:
    """Convert a scoreline into a three-class outcome label."""
    if home_score > away_score:
        return "home_win"
    if home_score < away_score:
        return "away_win"
    return "draw"
