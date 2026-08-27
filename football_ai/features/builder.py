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
from football_ai.scrapers.sources.footballsquads import _COMPETITION_TOURNAMENT_NAMES


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
    """Return a column when present or a default-valued Series otherwise.

    `default_value` should be `np.nan`, not `0.0`, for any column that ever
    feeds a rolling aggregate (`_add_rolling_features`'s `.rolling().mean()`
    treats `0.0` as a real observation, dragging the average toward zero for
    matches where the stat was simply never recorded -- indistinguishable
    from "this team genuinely scored/possessed nothing"). `0.0` remains the
    right default for columns that are never rolled and where "no data"
    should read as a neutral zero to a caller (e.g. `injured_market_value` --
    see `test_injured_market_value_diff_defaults_to_zero_without_source_data`).
    """
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
            # NaN default (not 0.0) -- these feed rolling means whenever a
            # caller's config re-adds them to rolling_metrics (see
            # `_safe_series`'s docstring); a StatsBomb-sparse row's "no data"
            # must not silently read as "team recorded exactly 0" once rolled.
            "xg": _safe_series(match_frame, "home_xg", default_value=np.nan),
            "xga": _safe_series(match_frame, "away_xg", default_value=np.nan),
            "shots": _safe_series(match_frame, "home_shots", default_value=np.nan),
            "shots_on_target": _safe_series(match_frame, "home_shots_on_target", default_value=np.nan),
            "corners": _safe_series(match_frame, "home_corners", default_value=np.nan),
            "cards": _safe_series(match_frame, "home_yellow_cards", default_value=np.nan)
            + _safe_series(match_frame, "home_red_cards", default_value=np.nan),
            "possession": _safe_series(match_frame, "home_possession", default_value=np.nan),
            "pressing_intensity": _safe_series(match_frame, "home_pressing_intensity", default_value=np.nan),
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
            "xg": _safe_series(match_frame, "away_xg", default_value=np.nan),
            "xga": _safe_series(match_frame, "home_xg", default_value=np.nan),
            "shots": _safe_series(match_frame, "away_shots", default_value=np.nan),
            "shots_on_target": _safe_series(match_frame, "away_shots_on_target", default_value=np.nan),
            "corners": _safe_series(match_frame, "away_corners", default_value=np.nan),
            "cards": _safe_series(match_frame, "away_yellow_cards", default_value=np.nan)
            + _safe_series(match_frame, "away_red_cards", default_value=np.nan),
            "possession": _safe_series(match_frame, "away_possession", default_value=np.nan),
            "pressing_intensity": _safe_series(match_frame, "away_pressing_intensity", default_value=np.nan),
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
    """Create rolling team and venue-specific features.

    The venue split groups on `effective_is_home`, not raw `is_home` --
    CONFIRMED REAL BUG (found via `PROMPT_HOME_ADVANTAGE_CHECK.md`'s
    diagnostic, fixed 2026-08-25 per `PROMPT_ZAMKNIECIE_WATKOW.md`): raw
    `is_home` is purely a positional label from the source data (whichever
    side is listed as `home_team_name`), so a team playing a neutral-venue
    match (a World Cup final, a Nations League tie on neutral ground) still
    got that match folded into its "home" rolling bucket exactly as if it
    had real home-crowd advantage. `effective_is_home` is 1 only when a
    team is both listed as home AND the match isn't on neutral ground, so
    neutral matches feed the away/`False` bucket for both sides instead --
    the model can then only learn genuine home advantage from truly-home
    matches, not from a data leak of the label alone. `neutral_venue` is
    treated as all-False (i.e. this is a no-op, matching prior behavior)
    when the column isn't present in `frame` at all, consistent with this
    project's "not every source has every field" rule elsewhere in this
    module (see `_add_tournament_host_flag`).
    """
    engineered = frame.copy()
    if "neutral_venue" in engineered.columns:
        effective_is_home = engineered["is_home"] & (~engineered["neutral_venue"])
    else:
        effective_is_home = engineered["is_home"]
    derived_columns: dict[str, pd.Series] = {}
    for metric in config.rolling_metrics:
        for window in config.rolling_windows:
            shifted = engineered.groupby("team_name")[metric].shift(1)
            venue_shifted = engineered.groupby(
                ["team_name", effective_is_home]
            )[metric].shift(1)
            rolling_all = shifted.groupby(engineered["team_name"]).rolling(
                window=window, min_periods=1
            )
            rolling_venue = venue_shifted.groupby(
                [engineered["team_name"], effective_is_home]
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


_SQUAD_SUMMARY_NUMERIC_FIELDS = (
    "average_age_years",
    "average_height_m",
    "squad_continuity_ratio",
    "largest_club_cluster_size",
    "largest_club_cluster_pairs",
    "total_intra_club_pairs",
)
_SQUAD_MERGE_TOLERANCE = pd.Timedelta(days=45)


def merge_squad_indices(
    matches: pd.DataFrame, squad_summary: pd.DataFrame
) -> pd.DataFrame:
    """Attach each side's tournament-squad indices (age/height/continuity/cohesion).

    `squad_summary` is `footballsquads.py`'s per-team-per-tournament output
    (`FootballSquadsSource.build_squads()`'s second return value, i.e.
    `tournament_squads_summary.parquet`) -- CONFIRMED REAL GAP found
    2026-08-25 auditing `PROMPT_ZAMKNIECIE_WATKOW.md`'s item 2: these
    indices were computed but never merged anywhere, so no model ever saw
    them despite the source existing since 2026-08-20. One row per team per
    *tournament instance*, not per match, so this only ever covers a match
    whose `tournament` is one of the five footballsquads-covered
    competitions and whose date falls near that instance's run -- matched
    via `merge_asof` per (team, competition), taking the most recent squad
    snapshot at-or-before the match. `tolerance` bounds "at-or-before" to a
    single tournament's own duration (45 days comfortably covers all five:
    World Cup ~32 days is the longest) so a team's NEXT tournament years
    later can never be mistaken for a match near this one. Every other
    match (friendlies, qualifiers, non-covered competitions -- the vast
    majority of rows) gets `NaN` here, not a fabricated average: genuinely
    unobserved, not zero.
    """
    if squad_summary.empty or "tournament" not in matches.columns:
        return matches

    competition_by_tournament = {
        name: slug for slug, name in _COMPETITION_TOURNAMENT_NAMES.items()
    }
    numeric_fields = [
        field
        for field in _SQUAD_SUMMARY_NUMERIC_FIELDS
        if field in squad_summary.columns
    ]
    if not numeric_fields:
        return matches

    summary = squad_summary.dropna(subset=["tournament_start_date"]).copy()
    summary["team_name"] = summary["team_name"].map(canonicalize_team_name)
    summary["_squad_date"] = pd.to_datetime(summary["tournament_start_date"], utc=True)
    summary = summary[["competition", "team_name", "_squad_date", *numeric_fields]]

    engineered = matches.copy()
    engineered["_squad_competition"] = engineered["tournament"].map(
        competition_by_tournament
    )
    engineered["_squad_date"] = pd.to_datetime(engineered["match_date"], utc=True)

    for side, team_column in (("home", "home_team_name"), ("away", "away_team_name")):
        side_frame = pd.DataFrame(
            {
                "_original_index": engineered.index,
                "competition": engineered["_squad_competition"],
                "team_name": engineered[team_column].map(canonicalize_team_name),
                "_squad_date": engineered["_squad_date"],
            }
        ).sort_values("_squad_date")
        joined = pd.merge_asof(
            side_frame,
            summary.sort_values("_squad_date"),
            on="_squad_date",
            by=["competition", "team_name"],
            direction="backward",
            tolerance=_SQUAD_MERGE_TOLERANCE,
        ).set_index("_original_index")
        joined = joined.reindex(engineered.index)
        for field in numeric_fields:
            # `field` is already "squad_continuity_ratio" for one entry --
            # strip a leading "squad_" so the output name doesn't double up
            # to "home_squad_squad_continuity_ratio".
            output_column = f"{side}_squad_{field.removeprefix('squad_')}"
            if output_column in engineered.columns:
                continue  # matches already provides this column -- never overwrite it
            engineered[output_column] = joined[field].to_numpy()

    return engineered.drop(columns=["_squad_competition", "_squad_date"])


def build_features(
    matches: pd.DataFrame,
    config: FeatureBuilderConfig | None = None,
    squad_summary: pd.DataFrame | None = None,
) -> FeatureBuildResult:
    """Build automated feature tables from historical match rows."""
    feature_config = config or FeatureBuilderConfig()

    # CONFIRMED REAL naming mismatch (found 2026-08-19 diagnosing why every
    # `home_possession`/`away_possession` prediction came back as the same
    # hardcoded default regardless of the match): `scrapers/sources/espn.py`'s
    # `_STAT_NAME_MAP` -- the only enrichment source that supplies possession
    # at all -- emits `home_possession_pct`/`away_possession_pct` (matching
    # `database/models.py`'s `TeamStatistic.possession_pct` and the
    # `records_frame`/Streamlit heatmap's own "possession_pct" convention),
    # while `_build_targets` below and `build_team_match_frame` both only ever
    # look for `home_possession`/`away_possession` (no `_pct`). Since
    # `data.enrichment.merge_advanced_stats` merges supplemental columns under
    # whatever name they already have -- no semantic renaming -- the `_pct`
    # column always lands in the wide `matches` frame under its ESPN name and
    # `home_possession` never exists, so `run_training()` silently skips the
    # target every single time regardless of how much real possession data
    # was actually fetched. Aliased here (added, not renamed) rather than
    # fixed at the ESPN source, because `_STAT_NAME_MAP`'s "possession_pct"
    # name is a real, separate contract other consumers (the DB model, the
    # records-frame heatmap) already depend on -- renaming it there would fix
    # this and break those. Reassigned onto `matches` itself, before
    # `match_frame` is derived from it and before `source_columns` below reads
    # `matches.columns` -- `_build_targets`'s docstring is explicit that
    # `source_columns` must reflect the true original input, not a
    # `_safe_series`-filled pass-through, so the alias has to land on `matches`
    # itself, not just on the `match_frame` copy, or `_build_targets` would
    # still never see it. This is the one shared frame both
    # `training/orchestrator.py` and `predictions/inference_feature_builder.py`
    # build features from, so training and inference stay on the same code
    # path per this module's own invariant.
    matches = matches.copy()
    for _side in ("home", "away"):
        _target_column = f"{_side}_possession"
        _source_column = f"{_side}_possession_pct"
        if _target_column not in matches.columns and _source_column in matches.columns:
            matches[_target_column] = matches[_source_column]

    if squad_summary is not None:
        matches = merge_squad_indices(matches, squad_summary)

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

    # `home_team_name`/`away_team_name` used to be here too. CONFIRMED REAL
    # LEAK via direct SHAP inspection (2026-08-27,
    # PROMPT_USUNIECIE_TOZSAMOSCI_DRUZYNY.md): one-hot-encoded team
    # identity gave the model a *separate* learned
    # prior for "TeamX-as-home" vs "TeamX-as-away" -- for a historically
    # dominant team (Germany), `away_team_name_Germany` was the single
    # highest-|SHAP| feature in a real match_outcome prediction, ahead of
    # every Elo/rolling feature, and it doesn't respect `neutral_venue` at
    # all (unlike `home_is_home`, already excluded, this isn't even a
    # positional flag -- it's the team's raw identity, so no gate on
    # `is_home` could fix it). A more "generic" pair (Portugal/Netherlands)
    # showed far less of this effect, confirming it tracks a specific
    # team's historical lopsidedness, not a universal artifact.
    # `tournament`/`country` don't carry the same risk: neither is a
    # near-unique per-team identifier (many different away teams share the
    # same host `country`; `tournament` is a competition name, not a team).
    # Same-data, same-`--n-trials 10` ablation (match_outcome, LightGBM
    # best model both times) confirmed the metric cost is small, not a
    # hidden major signal source: accuracy 0.5842 -> 0.5766 (-0.0077),
    # f1 0.5085 -> 0.5023 (-0.0062), both under 1 percentage point.
    categorical_columns = [
        column
        for column in ["tournament", "country"]
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
            # Same "always zero-filled historically" story as the block above
            # (confirmed 2026-07-30: neither `home_lineup_average_age` nor
            # `home_travel_distance_km` exist as columns anywhere in the real
            # training source), just missed when that list was written --
            # documentation-only fix, not a behavior change for tree models
            # (a constant column carries zero split information either way),
            # but keeps the exclusion list honestly matching its own claim.
            "home_lineup_average_age",
            "away_lineup_average_age",
            "home_travel_distance_km",
            "away_travel_distance_km",
            "travel_distance_diff",
            # CONFIRMED REAL LEAK, structurally different from the
            # neutral-venue rolling-feature bug fixed above (found
            # 2026-08-26 via PROMPT_NAPRAWA_PRZECIEKU_HOME.md's diagnostic,
            # after that fix still left the trained model reacting to
            # which side is "home" on a neutral-venue match). `is_home` is
            # a purely positional flag from the source sheet (1 for
            # whichever side `home_team_name` names, 0 otherwise) --
            # `_add_rolling_features`'s `effective_is_home` gate only
            # governs how *rolling aggregates* are bucketed; it never
            # touches this raw per-row value, which survives `_wide_merge`
            # unexcluded and reaches the model as a direct, ungated
            # "am I the sheet-labeled home side" signal -- on a neutral
            # match this is 1/0 exactly as if it were a real home match.
            # Genuine home advantage is already captured properly by the
            # venue-split rolling features and `home_is_tournament_host`/
            # `away_is_tournament_host` (both `neutral_venue`-aware); this
            # raw flag adds nothing legitimate on top of those, only the
            # leak. Excluded, not gated -- `_wide_merge`'s own use of raw
            # `is_home` for deciding which prefix a row's stats land under
            # is unrelated and must not change (see module's own hard
            # rule).
            "home_is_home",
            "away_is_home",
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
        # CONFIRMED REAL LEAK, same class as goals_for/goals_against above
        # (found 2026-07-30 auditing for more of these): `build_team_match_frame`
        # (lines ~58-59, 85-86) builds a *separate*, differently-named "cards"
        # column = this match's own yellow_cards + red_cards. "yellow_cards"/
        # "red_cards" above only cover the raw source columns they're spelled
        # after -- "cards" is a distinct name that neither this list nor
        # `_build_targets`'s target-name matching ever caught, so
        # home_cards/away_cards flowed in unguarded as a near-exact restatement
        # of the home_yellow_cards/away_yellow_cards regression targets
        # (red cards are rare, so cards ~= yellow_cards for most rows).
        "cards",
        "fouls",
        # CONFIRMED REAL LEAK, same class as "cards" above (found 2026-08-19
        # auditing why home_possession/away_possession scored a suspicious
        # test R^2 of 0.99+): the real source columns are spelled
        # "fouls_committed", "total_passes" and "pass_pct" -- "fouls" and
        # "passes_completed"/"passes_attempted" below never matched them, so
        # home_pass_pct/away_pass_pct/home_total_passes/away_total_passes/
        # home_fouls_committed/away_fouls_committed flowed into the possession
        # models unguarded. Post-match pass accuracy is a near-restatement of
        # possession, which is exactly what inflated that R^2 -- and at real
        # /predict time none of those columns exist yet for a future match,
        # so the model was learning to lean on a signal it will never
        # actually have at inference.
        "fouls_committed",
        "total_passes",
        "pass_pct",
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
