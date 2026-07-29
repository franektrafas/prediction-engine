"""Feature engineering schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(slots=True)
class FeatureBuilderConfig:
    """Configuration for automated feature generation."""

    rolling_windows: tuple[int, ...] = (3, 5, 10, 15)
    # xg/xga/shots/shots_on_target/corners/cards/possession/pressing_intensity were
    # dropped from this list (2026-07-24): real ablation on the enriched dataset
    # showed StatsBomb-derived rolling columns contributing only 0.1-0.2% of total
    # feature importance across match_outcome/btts/over_2_5, with 587 of the
    # resulting columns at exactly zero importance in all three -- StatsBomb only
    # covers 1.1% of rows (285/25,434), so LightGBM correctly learns to ignore
    # them. Re-add once StatsBomb (or another source) covers enough of the
    # dataset for these to carry real signal; the raw per-match stats
    # (home_xg, home_corners, ...) are unaffected -- they're still valid
    # `_build_targets` targets, this only removes their *rolling* aggregates
    # as input features. Side effect: `_add_composite_indices`'s attack/defense
    # index needs xg_mean/xga_mean alongside goals_for_mean/goals_against_mean
    # and now silently no-ops (it already guards on all four columns existing)
    # -- acceptable, since with 1.1% coverage that composite was already ~=
    # goals_for_mean for 98.9% of rows anyway.
    rolling_metrics: tuple[str, ...] = (
        "goals_for",
        "goals_against",
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
