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


# Single source of truth for the `match_outcome` label <-> training-code mapping.
# `features.builder._build_targets` encodes the target with this before fitting
# (xgboost's sklearn API rejects arbitrary string class labels outright, and
# every other model in the suite is trained on the same shared target Series),
# and `predictions.service._canonicalize_class_label` decodes a model's numeric
# `classes_` back with the same table, so the two ends can't drift apart.
MATCH_OUTCOME_LABELS: tuple[str, ...] = ("away_win", "draw", "home_win")
MATCH_OUTCOME_CODES: dict[str, int] = {
    label: code for code, label in enumerate(MATCH_OUTCOME_LABELS)
}
