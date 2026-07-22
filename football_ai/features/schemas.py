"""Feature engineering schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class FeatureBuilderConfig:
    """Configuration for automated feature generation."""

    rolling_windows: tuple[int, ...] = (3, 5, 10, 15)
    rolling_metrics: tuple[str, ...] = (
        "goals_for",
        "goals_against",
        "xg",
        "xga",
        "shots",
        "shots_on_target",
        "corners",
        "cards",
        "possession",
        "pressing_intensity",
        # Outcome-rate indicators (see features.builder._add_outcome_indicator_columns) --
        # rolling these gives e.g. "BTTS % over the last 5 matches". The raw,
        # unrolled per-match values are excluded from the feature set via
        # RAW_MATCH_OUTCOME_FIELDS since they're derived from this match's own score.
        "btts_flag",
        "clean_sheet_flag",
        "over_2_5_flag",
        # Match-over-match Elo delta (see features.builder._add_elo_momentum_column) --
        # rolling this gives "is this team's Elo trending up or down lately".
        "elo_change",
    )
    rolling_aggregations: tuple[str, ...] = ("mean", "std", "min", "max")
    h2h_windows: tuple[int, ...] = (3, 5)
    recent_match_window_days: int = 30


@dataclass(slots=True)
class FeatureBuildResult:
    """Output of the feature engineering pipeline."""

    frame: pd.DataFrame
    feature_columns: list[str]
    categorical_columns: list[str]
    numeric_columns: list[str]
    targets: dict[str, pd.Series] = field(default_factory=dict)
