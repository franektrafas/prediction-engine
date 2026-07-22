"""Automated feature generation for football matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_ai.data.normalization import MATCH_OUTCOME_CODES, compute_match_outcome
from football_ai.features.schemas import FeatureBuildResult, FeatureBuilderConfig


BASE_MATCH_COLUMNS = [
    "match_id",
    "match_date",
    "home_team_name",
    "away_team_name",
    "tournament",
    "stadium_name",
    "country",
    "neutral_venue",
]


def _safe_series(frame: pd.DataFrame, column: str, default_value: float = 0.0) -> pd.Series:
    """Return a column when present or a default-valued Series otherwise."""
    if column in frame.columns:
        return frame[column].fillna(default_value)
    return pd.Series(default_value, index=frame.index, dtype=float)


def build_team_match_frame(matches: pd.DataFrame) -> pd.DataFrame:
    """Convert wide match rows into a long team-match frame."""
    match_frame = matches.copy().sort_values("match_date").reset_index(drop=True)
    if "match_id" not in match_frame.columns:
        match_frame["match_id"] = np.arange(1, len(match_frame) + 1)

    shared_columns = [column for column in BASE_MATCH_COLUMNS if column in match_frame.columns]

    home_frame = pd.DataFrame(
        {
            **{column: match_frame[column] for column in shared_columns},
            "team_name": match_frame["home_team_name"],
            "opponent_name": match_frame["away_team_name"],
            "is_home": 1,
            "goals_for": _safe_series(match_frame, "home_score"),
            "goals_against": _safe_series(match_frame, "away_score"),
            "xg": _safe_series(match_frame, "home_xg"),
            "xga": _safe_series(match_frame, "away_xg"),
            "shots": _safe_series(match_frame, "home_shots"),
            "shots_on_target": _safe_series(match_frame, "home_shots_on_target"),
            "corners": _safe_series(match_frame, "home_corners"),
            "cards": _safe_series(match_frame, "home_yellow_cards")
            + _safe_series(match_frame, "home_red_cards"),
            "possession": _safe_series(match_frame, "home_possession"),
            "pressing_intensity": _safe_series(match_frame, "home_pressing_intensity"),
            "elo_pre_match": _safe_series(match_frame, "home_elo"),
            "lineup_average_age": _safe_series(match_frame, "home_lineup_average_age"),
            "lineup_market_value": _safe_series(match_frame, "home_lineup_market_value"),
            "goalkeeper_form": _safe_series(match_frame, "home_goalkeeper_form"),
            "forward_form": _safe_series(match_frame, "home_forward_form"),
            "travel_distance_km": _safe_series(match_frame, "home_travel_distance_km"),
        }
    )

    away_frame = pd.DataFrame(
        {
            **{column: match_frame[column] for column in shared_columns},
            "team_name": match_frame["away_team_name"],
            "opponent_name": match_frame["home_team_name"],
            "is_home": 0,
            "goals_for": _safe_series(match_frame, "away_score"),
            "goals_against": _safe_series(match_frame, "home_score"),
            "xg": _safe_series(match_frame, "away_xg"),
            "xga": _safe_series(match_frame, "home_xg"),
            "shots": _safe_series(match_frame, "away_shots"),
            "shots_on_target": _safe_series(match_frame, "away_shots_on_target"),
            "corners": _safe_series(match_frame, "away_corners"),
            "cards": _safe_series(match_frame, "away_yellow_cards")
            + _safe_series(match_frame, "away_red_cards"),
            "possession": _safe_series(match_frame, "away_possession"),
            "pressing_intensity": _safe_series(match_frame, "away_pressing_intensity"),
            "elo_pre_match": _safe_series(match_frame, "away_elo"),
            "lineup_average_age": _safe_series(match_frame, "away_lineup_average_age"),
            "lineup_market_value": _safe_series(match_frame, "away_lineup_market_value"),
            "goalkeeper_form": _safe_series(match_frame, "away_goalkeeper_form"),
            "forward_form": _safe_series(match_frame, "away_forward_form"),
            "travel_distance_km": _safe_series(match_frame, "away_travel_distance_km"),
        }
    )

    long_frame = pd.concat([home_frame, away_frame], ignore_index=True)
    long_frame["match_date"] = pd.to_datetime(long_frame["match_date"], utc=True)
    long_frame = long_frame.sort_values(["team_name", "match_date", "match_id"]).reset_index(
        drop=True
    )
    return long_frame


def _add_rolling_features(
    frame: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Create rolling team and venue-specific features."""
    engineered = frame.copy()
    derived_columns: dict[str, pd.Series] = {}
    for metric in config.rolling_metrics:
        for window in config.rolling_windows:
            shifted = engineered.groupby("team_name")[metric].shift(1)
            venue_shifted = engineered.groupby(["team_name", "is_home"])[metric].shift(1)
            rolling_all = shifted.groupby(engineered["team_name"]).rolling(
                window=window, min_periods=1
            )
            rolling_venue = venue_shifted.groupby(
                [engineered["team_name"], engineered["is_home"]]
            ).rolling(window=window, min_periods=1)
            for aggregation in config.rolling_aggregations:
                derived_columns[f"{metric}_{aggregation}_{window}"] = (
                    rolling_all.aggregate(aggregation).reset_index(level=0, drop=True)
                )
                derived_columns[f"{metric}_{aggregation}_{window}_venue"] = (
                    rolling_venue.aggregate(aggregation).reset_index(level=[0, 1], drop=True)
                )
    if derived_columns:
        engineered = pd.concat([engineered, pd.DataFrame(derived_columns, index=engineered.index)], axis=1)
    return engineered


def _add_recency_features(
    frame: pd.DataFrame,
    config: FeatureBuilderConfig,
) -> pd.DataFrame:
    """Add rest-day and recent-volume features."""
    engineered = frame.copy()
    previous_match_date = engineered.groupby("team_name")["match_date"].shift(1)
    engineered["rest_days"] = (
        engineered["match_date"] - previous_match_date
    ).dt.total_seconds() / 86400.0
    engineered["rest_days"] = engineered["rest_days"].fillna(14.0)

    recent_counts: list[int] = []
    for _, team_frame in engineered.groupby("team_name", sort=False):
        dates = team_frame["match_date"].tolist()
        counts_for_team: list[int] = []
        for current_date in dates:
            start_date = current_date - pd.Timedelta(days=config.recent_match_window_days)
            counts_for_team.append(sum(start_date <= value < current_date for value in dates))
        recent_counts.extend(counts_for_team)
    engineered["matches_last_30_days"] = recent_counts
    engineered["fatigue_index"] = engineered["matches_last_30_days"] / engineered["rest_days"].clip(
        lower=1.0
    )
    return engineered


def _add_h2h_features(frame: pd.DataFrame, config: FeatureBuilderConfig) -> pd.DataFrame:
    """Add head-to-head summary features."""
    engineered = frame.copy()
    engineered["h2h_pair"] = engineered.apply(
        lambda row: " vs ".join(sorted([row["team_name"], row["opponent_name"]])),
        axis=1,
    )
    engineered["h2h_signed_result"] = engineered["goals_for"] - engineered["goals_against"]
    for window in config.h2h_windows:
        shifted = engineered.groupby(["team_name", "opponent_name"])["h2h_signed_result"].shift(1)
        rolling = shifted.groupby([engineered["team_name"], engineered["opponent_name"]]).rolling(
            window=window, min_periods=1
        )
        engineered[f"h2h_goal_diff_mean_{window}"] = rolling.mean().reset_index(
            level=[0, 1], drop=True
        )
        engineered[f"h2h_goal_diff_std_{window}"] = rolling.std().reset_index(
            level=[0, 1], drop=True
        )
    return engineered.drop(columns=["h2h_pair", "h2h_signed_result"])


def _drop_columns_shadowing_source(
    engineered_frame: pd.DataFrame,
    matches: pd.DataFrame,
    join_key: str,
) -> pd.DataFrame:
    """Drop engineered columns that would collide with a column already in `matches`.

    `matches` is always the source of truth for any column it already defines
    (raw per-match outcome stats such as ``home_xg``/``home_shots``, or
    pre-match context such as ``home_lineup_average_age``). Without this guard,
    ``DataFrame.merge`` silently renames both copies to ``*_x``/``*_y`` on a
    name collision. That has two dangerous consequences: (1) every later
    exact-name lookup (targets in ``_build_targets``, categorical columns,
    relative-feature diffs) silently stops finding the column and drops it,
    and (2) the *engineered* copy of a raw outcome stat (the current match's
    own xG, shots, possession, ...) survives under the suffixed name and gets
    picked up as a plain numeric feature -- i.e. the model is trained on the
    answer to the question it is meant to predict. Dropping the shadowing
    engineered column keeps `matches`' original value intact and removes the
    collision (and the leak) entirely.
    """
    source_columns = set(matches.columns)
    columns_to_drop = [
        column
        for column in engineered_frame.columns
        if column in source_columns and column != join_key
    ]
    return engineered_frame.drop(columns=columns_to_drop)


def _wide_merge(engineered_long_frame: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Merge home and away engineered features back into match rows."""
    home_features = engineered_long_frame.loc[engineered_long_frame["is_home"] == 1].copy()
    away_features = engineered_long_frame.loc[engineered_long_frame["is_home"] == 0].copy()
    home_features = home_features.add_prefix("home_")
    away_features = away_features.add_prefix("away_")
    home_features = _drop_columns_shadowing_source(home_features, matches, join_key="home_match_id")
    away_features = _drop_columns_shadowing_source(away_features, matches, join_key="away_match_id")
    merged = matches.copy()
    merged = merged.merge(
        home_features,
        left_on="match_id",
        right_on="home_match_id",
        how="left",
    )
    merged = merged.merge(
        away_features,
        left_on="match_id",
        right_on="away_match_id",
        how="left",
    )
    return merged


def _add_relative_features(frame: pd.DataFrame, config: FeatureBuilderConfig) -> pd.DataFrame:
    """Create home-away deltas and ratios for core rolling features."""
    engineered = frame.copy()
    derived_columns: dict[str, pd.Series] = {}
    for metric in config.rolling_metrics + (
        "elo_pre_match",
        "rest_days",
        "matches_last_30_days",
        "fatigue_index",
        "lineup_average_age",
        "lineup_market_value",
        "goalkeeper_form",
        "forward_form",
        "travel_distance_km",
    ):
        for window in config.rolling_windows:
            for aggregation in ("mean", "std"):
                home_column = f"home_{metric}_{aggregation}_{window}"
                away_column = f"away_{metric}_{aggregation}_{window}"
                if home_column in engineered.columns and away_column in engineered.columns:
                    derived_columns[f"{metric}_{aggregation}_{window}_diff"] = (
                        engineered[home_column] - engineered[away_column]
                    )
                    denominator = engineered[away_column].replace(0, np.nan).abs()
                    derived_columns[f"{metric}_{aggregation}_{window}_ratio"] = (
                        engineered[home_column] / denominator
                    ).replace([np.inf, -np.inf], np.nan)

    direct_pairs = [
        ("home_elo_pre_match", "away_elo_pre_match", "elo_diff"),
        ("home_rest_days", "away_rest_days", "rest_days_diff"),
        ("home_matches_last_30_days", "away_matches_last_30_days", "recent_load_diff"),
        ("home_travel_distance_km", "away_travel_distance_km", "travel_distance_diff"),
    ]
    for home_column, away_column, output_column in direct_pairs:
        if home_column in engineered.columns and away_column in engineered.columns:
            derived_columns[output_column] = engineered[home_column] - engineered[away_column]
    if derived_columns:
        engineered = pd.concat([engineered, pd.DataFrame(derived_columns, index=engineered.index)], axis=1)
    return engineered


def _build_targets(frame: pd.DataFrame, source_columns: set[str]) -> dict[str, pd.Series]:
    """Create model targets from historical results.

    `source_columns` must be the *original* `matches` input's columns, not
    `frame`'s. `frame` is the post-merge feature frame, where e.g. `home_xg`/
    `away_shots`/... can exist as an all-zero pass-through column even when
    the raw stat was never in the source data: those names are also
    `FeatureBuilderConfig.rolling_metrics` entries, so `build_team_match_frame`
    always builds a long-frame column for them via `_safe_series`'s
    zero-filled fallback, and `_wide_merge` only drops a column when it
    collides with something already in `matches` (`_drop_columns_shadowing_source`)
    -- a name that was never in `matches` at all doesn't collide with
    anything and survives. Checking `frame.columns` here would silently turn
    "this stat was never observed" into a constant, meaningless regression
    target instead of correctly leaving it out.
    """
    targets: dict[str, pd.Series] = {}
    if {"home_score", "away_score"}.issubset(frame.columns):
        total_goals = frame["home_score"] + frame["away_score"]
        # Encoded as int (not the raw "home_win"/"draw"/"away_win" string) because
        # xgboost's sklearn API rejects arbitrary string class labels on `.fit()`,
        # and this Series is shared across every model in the training suite --
        # see `data.normalization.MATCH_OUTCOME_CODES` for the shared mapping that
        # `predictions.service` decodes back with.
        targets["match_outcome"] = frame.apply(
            lambda row: MATCH_OUTCOME_CODES[
                compute_match_outcome(row["home_score"], row["away_score"])
            ],
            axis=1,
        )
        targets["home_goals"] = frame["home_score"]
        targets["away_goals"] = frame["away_score"]
        targets["btts"] = ((frame["home_score"] > 0) & (frame["away_score"] > 0)).astype(int)
        targets["over_1_5"] = (total_goals > 1.5).astype(int)
        targets["over_2_5"] = (total_goals > 2.5).astype(int)
        targets["over_3_5"] = (total_goals > 3.5).astype(int)
    for column in [
        "home_xg",
        "away_xg",
        "home_corners",
        "away_corners",
        "home_yellow_cards",
        "away_yellow_cards",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_possession",
        "away_possession",
    ]:
        if column in source_columns:
            targets[column] = frame[column]
    return targets


def build_features(
    matches: pd.DataFrame,
    config: FeatureBuilderConfig | None = None,
) -> FeatureBuildResult:
    """Build automated feature tables from historical match rows."""
    feature_config = config or FeatureBuilderConfig()
    match_frame = matches.copy().sort_values("match_date").reset_index(drop=True)
    if "match_id" not in match_frame.columns:
        match_frame["match_id"] = np.arange(1, len(match_frame) + 1)

    long_frame = build_team_match_frame(match_frame)
    long_frame = _add_rolling_features(long_frame, feature_config)
    long_frame = _add_recency_features(long_frame, feature_config)
    long_frame = _add_h2h_features(long_frame, feature_config)
    feature_frame = _wide_merge(long_frame, match_frame)
    feature_frame = _add_relative_features(feature_frame, feature_config)

    categorical_columns = [
        column
        for column in ["tournament", "stadium_name", "country", "home_team_name", "away_team_name"]
        if column in feature_frame.columns
    ]
    excluded_columns = set(
        categorical_columns
        + [
            "match_id",
            "match_date",
            "home_score",
            "away_score",
            "home_match_id",
            "away_match_id",
            "home_match_date",
            "away_match_date",
        ]
    )
    # Raw per-match outcome statistics (as opposed to their rolling/historical
    # aggregates) are only known *after* kickoff. Some of these already become
    # explicit prediction targets in `_build_targets`, but the raw `matches`
    # frame coming from `team_statistics`-style sources can carry additional
    # post-match fields (fouls, offsides, duels won, saves, pass counts, ...)
    # that have no corresponding target here. Left unguarded, any such column
    # would flow straight into `numeric_columns` as a feature and leak the
    # match outcome into its own prediction. Exclude the full raw-stat family
    # explicitly rather than relying on the target list to happen to cover it.
    RAW_MATCH_OUTCOME_FIELDS = (
        "goals",
        "xg",
        "xga",
        "possession_pct",
        "possession",
        "shots",
        "shots_on_target",
        "corners",
        "yellow_cards",
        "red_cards",
        "fouls",
        "offsides",
        "passes_completed",
        "passes_attempted",
        "pressing_intensity",
        "duels_won",
        "saves",
    )
    excluded_columns.update(
        f"{side}_{field}" for side in ("home", "away") for field in RAW_MATCH_OUTCOME_FIELDS
    )
    source_columns = set(matches.columns)
    target_columns = set(_build_targets(feature_frame, source_columns).keys())
    numeric_columns = [
        column
        for column in feature_frame.columns
        if column not in excluded_columns
        and column not in target_columns
        and pd.api.types.is_numeric_dtype(feature_frame[column])
    ]
    feature_columns = numeric_columns + categorical_columns
    feature_frame = feature_frame.replace([np.inf, -np.inf], np.nan)
    return FeatureBuildResult(
        frame=feature_frame,
        feature_columns=feature_columns,
        categorical_columns=categorical_columns,
        numeric_columns=numeric_columns,
        targets=_build_targets(feature_frame, source_columns),
    )
