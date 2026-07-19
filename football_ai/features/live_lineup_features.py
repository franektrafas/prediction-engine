"""Live lineup-derived features for a single fixture."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


POSITION_GROUPS = ("G", "D", "M", "F")


@dataclass(slots=True)
class LiveLineupFeatureResult:
    """Output of live lineup feature extraction."""

    fixture_id: int
    frame: pd.DataFrame
    feature_columns: list[str]


def _numeric_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric series or a default-valued fallback."""
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _string_series(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a normalized string series or an empty fallback."""
    if column not in frame.columns:
        return pd.Series("", index=frame.index, dtype="object")
    return frame[column].fillna("").astype(str)


def _empty_team_feature_row(
    fixture_id: int,
    team_name: str,
    prefix: str,
) -> dict[str, float | int | str]:
    """Return a default-valued feature row for one team."""
    row: dict[str, float | int | str] = {
        f"{prefix}_fixture_id": fixture_id,
        f"{prefix}_team_name": team_name,
        f"{prefix}_starting_xi_count": 0,
        f"{prefix}_bench_count": 0,
        f"{prefix}_goalkeeper_started": 0,
        f"{prefix}_defender_count": 0,
        f"{prefix}_midfielder_count": 0,
        f"{prefix}_forward_count": 0,
        f"{prefix}_bench_goalkeeper_count": 0,
        f"{prefix}_bench_defender_count": 0,
        f"{prefix}_bench_midfielder_count": 0,
        f"{prefix}_bench_forward_count": 0,
        f"{prefix}_formation_line_count": 0,
        f"{prefix}_formation_density": 0.0,
        f"{prefix}_offensive_share_starting_xi": 0.0,
        f"{prefix}_defensive_share_starting_xi": 0.0,
        f"{prefix}_bench_offensive_depth": 0,
        f"{prefix}_bench_defensive_depth": 0,
        f"{prefix}_bench_depth_total": 0,
        f"{prefix}_lineup_structure_score": 0.0,
        f"{prefix}_avg_age_starting_xi": 0.0,
        f"{prefix}_avg_market_value_starting_xi": 0.0,
        f"{prefix}_total_market_value_starting_xi": 0.0,
        f"{prefix}_national_caps_sum_starting_xi": 0.0,
        f"{prefix}_national_minutes_sum_starting_xi": 0.0,
        f"{prefix}_club_form_mean_starting_xi": 0.0,
        f"{prefix}_injury_risk_mean_starting_xi": 0.0,
        f"{prefix}_rest_days_mean_starting_xi": 0.0,
        f"{prefix}_fatigue_mean_starting_xi": 0.0,
        f"{prefix}_same_club_pair_count": 0,
        f"{prefix}_same_league_pair_count": 0,
        f"{prefix}_captain_started": 0,
        f"{prefix}_experienced_core_count": 0,
        f"{prefix}_debutant_count": 0,
        f"{prefix}_returning_starter_count": 0,
        f"{prefix}_starter_continuity_ratio": 0.0,
        f"{prefix}_goalkeeper_continuity": 0,
    }
    return row


def _pair_count_with_same_attribute(values: pd.Series) -> int:
    """Count player pairs sharing the same non-empty categorical attribute."""
    filtered = values.loc[values.astype(str) != ""]
    if filtered.empty:
        return 0
    counts = filtered.value_counts()
    return int(sum(count * (count - 1) // 2 for count in counts))


def _compute_team_lineup_features(
    fixture_id: int,
    team_frame: pd.DataFrame,
    prefix: str,
) -> dict[str, float | int | str]:
    """Compute live lineup features for one team."""
    team_name = str(team_frame["team_name"].iloc[0]) if not team_frame.empty else ""
    row = _empty_team_feature_row(fixture_id=fixture_id, team_name=team_name, prefix=prefix)
    if team_frame.empty:
        return row

    starters = team_frame.loc[team_frame["started"] == True].copy()
    bench = team_frame.loc[team_frame["started"] == False].copy()
    starters["position"] = _string_series(starters, "position")
    bench["position"] = _string_series(bench, "position")

    row[f"{prefix}_starting_xi_count"] = int(len(starters))
    row[f"{prefix}_bench_count"] = int(len(bench))
    row[f"{prefix}_goalkeeper_started"] = int((starters["position"] == "G").any())

    for position_name, feature_suffix in (
        ("D", "defender_count"),
        ("M", "midfielder_count"),
        ("F", "forward_count"),
    ):
        row[f"{prefix}_{feature_suffix}"] = int((starters["position"] == position_name).sum())

    for position_name, feature_suffix in (
        ("G", "bench_goalkeeper_count"),
        ("D", "bench_defender_count"),
        ("M", "bench_midfielder_count"),
        ("F", "bench_forward_count"),
    ):
        row[f"{prefix}_{feature_suffix}"] = int((bench["position"] == position_name).sum())

    non_null_grid = starters["grid"].dropna().astype(str)
    formation_lines = sorted({value.split(":")[0] for value in non_null_grid if ":" in value})
    row[f"{prefix}_formation_line_count"] = int(len(formation_lines))
    row[f"{prefix}_formation_density"] = (
        float(len(non_null_grid) / len(starters))
        if len(starters) > 0
        else 0.0
    )

    attacking_players = int(starters["position"].isin(["M", "F"]).sum())
    defensive_players = int(starters["position"].isin(["D", "G"]).sum())
    row[f"{prefix}_offensive_share_starting_xi"] = (
        float(attacking_players / len(starters))
        if len(starters) > 0
        else 0.0
    )
    row[f"{prefix}_defensive_share_starting_xi"] = (
        float(defensive_players / len(starters))
        if len(starters) > 0
        else 0.0
    )

    row[f"{prefix}_bench_offensive_depth"] = int(bench["position"].isin(["M", "F"]).sum())
    row[f"{prefix}_bench_defensive_depth"] = int(bench["position"].isin(["D", "G"]).sum())
    row[f"{prefix}_bench_depth_total"] = int(len(bench))

    row[f"{prefix}_avg_age_starting_xi"] = float(_numeric_series(starters, "age").mean())
    row[f"{prefix}_avg_market_value_starting_xi"] = float(
        _numeric_series(starters, "market_value_eur").mean()
    )
    row[f"{prefix}_total_market_value_starting_xi"] = float(
        _numeric_series(starters, "market_value_eur").sum()
    )
    row[f"{prefix}_national_caps_sum_starting_xi"] = float(
        _numeric_series(starters, "national_team_caps").sum()
    )
    row[f"{prefix}_national_minutes_sum_starting_xi"] = float(
        _numeric_series(starters, "national_team_minutes").sum()
    )
    row[f"{prefix}_club_form_mean_starting_xi"] = float(
        _numeric_series(starters, "club_form_score").mean()
    )
    row[f"{prefix}_injury_risk_mean_starting_xi"] = float(
        _numeric_series(starters, "injury_risk_score").mean()
    )
    row[f"{prefix}_rest_days_mean_starting_xi"] = float(
        _numeric_series(starters, "rest_days").mean()
    )
    row[f"{prefix}_fatigue_mean_starting_xi"] = float(
        _numeric_series(starters, "fatigue_index").mean()
    )

    row[f"{prefix}_same_club_pair_count"] = _pair_count_with_same_attribute(
        _string_series(starters, "club_name")
    )
    row[f"{prefix}_same_league_pair_count"] = _pair_count_with_same_attribute(
        _string_series(starters, "league_name")
    )
    row[f"{prefix}_captain_started"] = int(_numeric_series(starters, "is_captain").max() > 0)
    row[f"{prefix}_experienced_core_count"] = int(
        (_numeric_series(starters, "national_team_caps") >= 25).sum()
    )
    row[f"{prefix}_debutant_count"] = int(
        (_numeric_series(starters, "national_team_caps") <= 1).sum()
    )
    row[f"{prefix}_returning_starter_count"] = int(
        (_numeric_series(starters, "started_previous_match") > 0).sum()
    )
    row[f"{prefix}_starter_continuity_ratio"] = (
        float(row[f"{prefix}_returning_starter_count"] / len(starters))
        if len(starters) > 0
        else 0.0
    )
    starter_goalkeepers = starters.loc[starters["position"] == "G"]
    row[f"{prefix}_goalkeeper_continuity"] = int(
        (_numeric_series(starter_goalkeepers, "started_previous_match") > 0).any()
    )
    row[f"{prefix}_lineup_structure_score"] = (
        float(row[f"{prefix}_formation_density"]) * 0.4
        + float(row[f"{prefix}_goalkeeper_started"]) * 0.2
        + float(min(len(starters), 11) / 11.0) * 0.2
        + float(row[f"{prefix}_starter_continuity_ratio"]) * 0.2
    )
    return row


def build_live_lineup_features(lineups: pd.DataFrame) -> LiveLineupFeatureResult:
    """Build single-row live lineup features for a fixture."""
    if lineups.empty:
        empty_frame = pd.DataFrame([{}])
        return LiveLineupFeatureResult(
            fixture_id=0,
            frame=empty_frame,
            feature_columns=[],
        )

    required_columns = {
        "fixture_id",
        "team_name",
        "position",
        "grid",
        "started",
    }
    missing_columns = required_columns.difference(lineups.columns)
    if missing_columns:
        missing_list = ", ".join(sorted(missing_columns))
        raise ValueError(f"Lineup frame is missing required columns: {missing_list}")

    fixture_id = int(lineups["fixture_id"].iloc[0])
    team_names = list(pd.Series(lineups["team_name"]).dropna().astype(str).unique())
    if len(team_names) != 2:
        raise ValueError("Lineup feature extraction expects exactly two teams in one fixture.")

    home_team_name = team_names[0]
    away_team_name = team_names[1]
    home_row = _compute_team_lineup_features(
        fixture_id=fixture_id,
        team_frame=lineups.loc[lineups["team_name"] == home_team_name].copy(),
        prefix="home",
    )
    away_row = _compute_team_lineup_features(
        fixture_id=fixture_id,
        team_frame=lineups.loc[lineups["team_name"] == away_team_name].copy(),
        prefix="away",
    )
    combined_row = {
        "fixture_id": fixture_id,
        "home_team_name": home_team_name,
        "away_team_name": away_team_name,
        **home_row,
        **away_row,
    }
    frame = pd.DataFrame([combined_row])
    frame["starting_xi_count_diff"] = (
        frame["home_starting_xi_count"] - frame["away_starting_xi_count"]
    )
    frame["bench_depth_diff"] = frame["home_bench_depth_total"] - frame["away_bench_depth_total"]
    frame["offensive_share_diff"] = (
        frame["home_offensive_share_starting_xi"] - frame["away_offensive_share_starting_xi"]
    )
    frame["defensive_share_diff"] = (
        frame["home_defensive_share_starting_xi"] - frame["away_defensive_share_starting_xi"]
    )
    frame["lineup_structure_score_diff"] = (
        frame["home_lineup_structure_score"] - frame["away_lineup_structure_score"]
    )
    frame["market_value_diff"] = (
        frame["home_total_market_value_starting_xi"] - frame["away_total_market_value_starting_xi"]
    )
    frame["avg_age_diff"] = frame["home_avg_age_starting_xi"] - frame["away_avg_age_starting_xi"]
    frame["national_caps_diff"] = (
        frame["home_national_caps_sum_starting_xi"] - frame["away_national_caps_sum_starting_xi"]
    )
    frame["club_form_diff"] = (
        frame["home_club_form_mean_starting_xi"] - frame["away_club_form_mean_starting_xi"]
    )
    frame["injury_risk_diff"] = (
        frame["home_injury_risk_mean_starting_xi"] - frame["away_injury_risk_mean_starting_xi"]
    )
    frame["starter_continuity_diff"] = (
        frame["home_starter_continuity_ratio"] - frame["away_starter_continuity_ratio"]
    )
    frame["same_club_pair_diff"] = (
        frame["home_same_club_pair_count"] - frame["away_same_club_pair_count"]
    )
    feature_columns = [
        column
        for column in frame.columns
        if column not in {"fixture_id", "home_team_name", "away_team_name", "home_fixture_id", "away_fixture_id"}
    ]
    return LiveLineupFeatureResult(
        fixture_id=fixture_id,
        frame=frame,
        feature_columns=feature_columns,
    )
