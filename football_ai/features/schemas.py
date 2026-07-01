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
