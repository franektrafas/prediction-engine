"""Automated feature generation for football matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from football_ai.data.normalization import (
    MATCH_OUTCOME_CODES,
    canonicalize_team_name,
    compute_match_outcome,
)
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
            "injured_market_value": _safe_series(match_frame, "home_injured_market_value"),
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
            "injured_market_value": _safe_series(match_frame, "away_injured_market_value"),
        }
    )

    long_frame = pd.concat([home_frame, away_frame], ignore_index=True)
    long_frame["match_date"] = pd.to_datetime(long_frame["match_date"], utc=True)
    long_frame = long_frame.sort_values(["team_name", "match_date", "match_id"]).reset_index(
        drop=True
    )
    return long_frame


def _add_outcome_indicator_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Add 0/1 per-match outcome indicators so rolling features can capture historical rates.

    These raw indicators are excluded from the final feature set via
    `RAW_MATCH_OUTCOME_FIELDS` in `build_features()` -- only their rolling
    aggregates (e.g. `btts_flag_mean_5`, "BTTS % over the last 5 matches")
    are safe to use as features, since the raw per-match value is derived
    from this match's own goals and would otherwise leak the outcome being
    predicted straight back in as an input.
    """
    engineered = frame.copy()
    total_goals = engineered["goals_for"] + engineered["goals_against"]
    engineered["btts_flag"] = (
        (engineered["goals_for"] > 0) & (engineered["goals_against"] > 0)
    ).astype(float)
    engineered["clean_sheet_flag"] = (engineered["goals_against"] == 0).astype(float)
    engineered["over_2_5_flag"] = (total_goals > 2.5).astype(float)
    return engineered


def _add_elo_momentum_column(frame: pd.DataFrame) -> pd.DataFrame:
    """Add each team's match-over-match Elo delta, feeding rolling "Elo momentum" features.

    `elo_change` for a row reflects the *previous* match's outcome (this
    match's own `elo_pre_match` minus the prior match's), not this match's --
    legitimate pre-match information, the same as `elo_pre_match` itself, so
    no `RAW_MATCH_OUTCOME_FIELDS` exclusion is needed here.
    """
    engineered = frame.copy()
    engineered["elo_change"] = engineered.groupby("team_name")["elo_pre_match"].diff()
    return engineered


def _win_loss_streaks_for_team(group: pd.DataFrame) -> pd.DataFrame:
    """Compute win/loss streak lengths (as of just before each match) for one team's ordered matches."""
    result = np.sign(group["goals_for"] - group["goals_against"])
    shifted = result.shift(1)
    win_streaks: list[float] = []
    loss_streaks: list[float] = []
    current_win = 0
    current_loss = 0
    for value in shifted:
        if pd.isna(value) or value == 0:
            current_win = 0
            current_loss = 0
        elif value > 0:
            current_win += 1
            current_loss = 0
        else:
            current_loss += 1
            current_win = 0
        win_streaks.append(float(current_win))
        loss_streaks.append(float(current_loss))
    return pd.DataFrame({"win_streak": win_streaks, "loss_streak": loss_streaks}, index=group.index)


def _form_trend_for_team(group: pd.DataFrame) -> pd.Series:
    """Compute a recent-vs-prior 3-match league-points trend for one team's ordered matches."""
    result = np.sign(group["goals_for"] - group["goals_against"])
    points = pd.Series(
        np.select([result > 0, result == 0], [3.0, 1.0], default=0.0), index=group.index
    )
    shifted_points = points.shift(1)
    recent_points = shifted_points.rolling(window=3, min_periods=1).mean()
    return (recent_points - recent_points.shift(3)).fillna(0.0)


def _add_streak_and_momentum_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add win/loss streaks and a recent-form trend, computed from strictly prior matches.

    Assumes `frame` is already sorted chronologically per team (true for the
    long frame at this point in the pipeline, per `build_team_match_frame`'s
    own sort) -- both helpers walk each team's group in row order. Neither
    needs a `RAW_MATCH_OUTCOME_FIELDS` exclusion: every value here is derived
    from `.shift(1)`-ed history, never this match's own result.
    """
    engineered = frame.copy()
    streaks = engineered.groupby("team_name", sort=False, group_keys=False).apply(
        _win_loss_streaks_for_team
    )
    engineered["win_streak"] = streaks["win_streak"]
    engineered["loss_streak"] = streaks["loss_streak"]
    engineered["form_trend"] = engineered.groupby("team_name", sort=False, group_keys=False).apply(
        _form_trend_for_team
    )
    return engineered


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
        ("home_injured_market_value", "away_injured_market_value", "injured_market_value_diff"),
        ("home_win_streak", "away_win_streak", "win_streak_diff"),
        ("home_loss_streak", "away_loss_streak", "loss_streak_diff"),
        ("home_form_trend", "away_form_trend", "form_trend_diff"),
    ]
    for home_column, away_column, output_column in direct_pairs:
        if home_column in engineered.columns and away_column in engineered.columns:
            derived_columns[output_column] = engineered[home_column] - engineered[away_column]
    if derived_columns:
        engineered = pd.concat([engineered, pd.DataFrame(derived_columns, index=engineered.index)], axis=1)
    return engineered


def _add_composite_indices(frame: pd.DataFrame, config: FeatureBuilderConfig) -> pd.DataFrame:
    """Add simple, interpretable attack/defense composite indices per rolling window.

    Deliberately just goals + xG (attack) / goals-against + xGA (defense) --
    both already on comparable ~0-3 goal scales, rather than an arbitrarily
    weighted formula. shots/pressing_intensity are left out of the composite
    (a different scale, would need an arbitrary normalization factor) but
    stay available to models as their own separate rolling features either
    way.
    """
    engineered = frame.copy()
    derived_columns: dict[str, pd.Series] = {}
    for window in config.rolling_windows:
        home_goals_for = f"home_goals_for_mean_{window}"
        home_xg = f"home_xg_mean_{window}"
        away_goals_for = f"away_goals_for_mean_{window}"
        away_xg = f"away_xg_mean_{window}"
        if all(column in engineered.columns for column in (home_goals_for, home_xg, away_goals_for, away_xg)):
            home_attack = engineered[home_goals_for] + engineered[home_xg]
            away_attack = engineered[away_goals_for] + engineered[away_xg]
            derived_columns[f"home_attack_index_{window}"] = home_attack
            derived_columns[f"away_attack_index_{window}"] = away_attack
            derived_columns[f"attack_index_{window}_diff"] = home_attack - away_attack

        home_goals_against = f"home_goals_against_mean_{window}"
        home_xga = f"home_xga_mean_{window}"
        away_goals_against = f"away_goals_against_mean_{window}"
        away_xga = f"away_xga_mean_{window}"
        if all(
            column in engineered.columns
            for column in (home_goals_against, home_xga, away_goals_against, away_xga)
        ):
            home_defense = engineered[home_goals_against] + engineered[home_xga]
            away_defense = engineered[away_goals_against] + engineered[away_xga]
            derived_columns[f"home_defense_index_{window}"] = home_defense
            derived_columns[f"away_defense_index_{window}"] = away_defense
            derived_columns[f"defense_index_{window}_diff"] = home_defense - away_defense
    if derived_columns:
        engineered = pd.concat([engineered, pd.DataFrame(derived_columns, index=engineered.index)], axis=1)
    return engineered


_FRIENDLY_KEYWORDS = ("friendly",)
_QUALIFIER_KEYWORDS = ("qualif",)
_MAJOR_TOURNAMENT_KEYWORDS = (
    "world cup",
    "euro",
    "copa am",
    "africa cup",
    "afcon",
    "nations league",
    "confederations cup",
    "gold cup",
    "asian cup",
)


def _match_importance_tier(tournament: object) -> int:
    """Best-effort match-importance classification from the free-text tournament name.

    Tournament strings vary by source (StatsBomb gives e.g. "FIFA World Cup",
    api-football gives round names, a Kaggle base file has its own
    vocabulary) -- this is a keyword heuristic, not an exhaustive mapping.
    An unrecognized string falls back to tier 1 (treated as a regular
    competitive match) rather than erroring.
    """
    text = str(tournament or "").lower()
    if any(keyword in text for keyword in _FRIENDLY_KEYWORDS):
        return 0
    if any(keyword in text for keyword in _QUALIFIER_KEYWORDS):
        return 1
    if any(keyword in text for keyword in _MAJOR_TOURNAMENT_KEYWORDS):
        return 2
    return 1


def _add_match_importance_tier(frame: pd.DataFrame) -> pd.DataFrame:
    """Add a match-importance tier derived from the tournament name."""
    engineered = frame.copy()
    if "tournament" in engineered.columns:
        engineered["match_importance_tier"] = engineered["tournament"].map(_match_importance_tier)
    return engineered


def _add_tournament_host_flag(frame: pd.DataFrame) -> pd.DataFrame:
    """Add whether each side is playing in its own country (a tournament-host signal).

    A cheap, already-available refinement beyond the existing `neutral_venue`
    flag -- does the match's `country` field match that side's own canonical
    team name. Continent and distance-from-host-country are deliberately not
    implemented here (need a country-to-continent reference table this
    project doesn't have yet); flagged as deferred in the roadmap plan.
    Silently a no-op when `country` isn't present in this source's data
    (e.g. StatsBomb's normalize_matches doesn't set it).
    """
    engineered = frame.copy()
    if "country" not in engineered.columns:
        return engineered
    canonical_country = engineered["country"].map(canonicalize_team_name)
    if "home_team_name" in engineered.columns:
        engineered["home_is_tournament_host"] = (
            canonical_country == engineered["home_team_name"].map(canonicalize_team_name)
        ).astype(float)
    if "away_team_name" in engineered.columns:
        engineered["away_is_tournament_host"] = (
            canonical_country == engineered["away_team_name"].map(canonicalize_team_name)
        ).astype(float)
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
    long_frame = _add_outcome_indicator_columns(long_frame)
    long_frame = _add_elo_momentum_column(long_frame)
    long_frame = _add_streak_and_momentum_features(long_frame)
    long_frame = _add_rolling_features(long_frame, feature_config)
    long_frame = _add_recency_features(long_frame, feature_config)
    long_frame = _add_h2h_features(long_frame, feature_config)
    feature_frame = _wide_merge(long_frame, match_frame)
    feature_frame = _add_relative_features(feature_frame, feature_config)
    feature_frame = _add_composite_indices(feature_frame, feature_config)
    feature_frame = _add_match_importance_tier(feature_frame)
    feature_frame = _add_tournament_host_flag(feature_frame)

    categorical_columns = [
        column
        for column in ["tournament", "country", "home_team_name", "away_team_name"]
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
            # Identity/administrative metadata a supplemental source (e.g.
            # StatsBomb via data.enrichment.merge_advanced_stats) attaches
            # alongside its real stat columns -- not intended as predictive
            # signal. Confirmed as a real leak vector: with these merged in,
            # match_outcome hit ~0.9997 validation F1 (near-perfect on a
            # 3-class task is a leak, not a good model) -- CatBoost's native
            # categorical handling on "stadium_name" (285/25434 rows non-null,
            # ~200 distinct values) is the leading suspect, and the plain
            # numeric "competition_id" doubles as a near-unique tournament
            # identifier for the same sparse row set. Excluded wholesale
            # rather than guessing which one leaks -- none of these carry
            # genuine match-outcome signal a model should be learning from.
            "external_id",
            "source",
            "competition_id",
            "competition_name",
            "season_name",
            "stadium_name",
            "referee",
            # The base `matches` frame's own pre-computed total, present in
            # both the Kaggle-style historical file and the real-data smoke
            # test that caught this -- not a `TARGET_TASK_MAPPING` name
            # itself (unlike "btts"/"over_2_5"/...), so `target_columns`
            # below never catches it. It doesn't fully determine
            # match_outcome alone, but is a severe partial leak (total goals
            # scored, known only after kickoff).
            "total_goals",
            # Live-only enrichment fields (lineup market value, goalkeeper/
            # forward form, injury impact) -- always zero-filled in historical
            # training data (no point-in-time historical record exists, only
            # current snapshots), so no model can learn a relationship from
            # them; confirmed at exactly zero importance in real ablation
            # (2026-07-24). Excluded from the trainable feature set for
            # honesty about what the model actually uses -- still fully
            # computed and surfaced to callers via
            # `predictions.schemas.MatchContext` / `PredictionResponse.match_context`
            # as informational context, not as a claimed model input.
            "home_lineup_market_value",
            "away_lineup_market_value",
            "home_goalkeeper_form",
            "away_goalkeeper_form",
            "home_forward_form",
            "away_forward_form",
            "home_injured_market_value",
            "away_injured_market_value",
            "injured_market_value_diff",
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
    #
    # CONFIRMED REAL, SEVERE LEAK (found training on real data for the first
    # time ever, this session): `build_team_match_frame` names the raw
    # per-team goal columns "goals_for"/"goals_against" (see lines ~51-52,
    # 78-79), not "goals" -- so the "goals" entry that was here never matched
    # `home_goals_for`/`home_goals_against` at all. Those are literally this
    # match's own final score, wide-merged in unguarded; a real training run
    # picked them up as the single most important features by two orders of
    # magnitude (LightGBM/CatBoost/XGBoost all hit 100% test accuracy on
    # match_outcome). This bug predates this session entirely -- it's been
    # live since `goals_for`/`goals_against` were introduced -- but nothing
    # ever ran `run_training()` against real full-scale data to surface it
    # until now; the small synthetic fixtures in the test suite never
    # asserted on accuracy values, only that training completed.
    RAW_MATCH_OUTCOME_FIELDS = (
        "goals_for",
        "goals_against",
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
        # Raw per-match outcome indicators (see _add_outcome_indicator_columns) --
        # derived from this match's own score, so only their rolling
        # aggregates (btts_flag_mean_5, ...) are safe as features.
        "btts_flag",
        "clean_sheet_flag",
        "over_2_5_flag",
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
