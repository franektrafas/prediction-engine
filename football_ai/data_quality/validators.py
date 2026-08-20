"""Integrity checks for records_frame/goalscorers_frame-shaped data.

Each function takes a DataFrame and returns `list[QualityIssue]` -- pure,
side-effect-free, and independently testable without a database. Thresholds
below are calibrated against the real data already in this project (see
each check's docstring for the real numbers), not guessed: e.g. real
possession pairs sum to 100.00-101.00 (841 real rows, 2026-08-20), real
goal-event minutes never exceed 122 (47,914 real rows).
"""

from __future__ import annotations

import pandas as pd

from football_ai.data_quality.schemas import QualityIssue

_POSSESSION_SUM_TOLERANCE = 1.5
_MAX_PLAUSIBLE_EVENT_MINUTE = 130

_TEAM_STAT_COLUMNS = (
    "shots",
    "shots_on_target",
    "corners",
    "yellow_cards",
    "red_cards",
    "fouls_committed",
    "offsides",
    "total_passes",
    "saves",
)


def find_duplicate_matches(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag (match_date, home_team_name, away_team_name) groups seen more than once.

    Same fixture recorded twice under the same key is either a duplicate
    ingest or two legitimately different matches sharing a date (a rare
    same-day double-header) -- either way it's worth a human look before
    training on it, hence "warning" not "error".
    """
    key_columns = ["match_date", "home_team_name", "away_team_name"]
    if not set(key_columns).issubset(records_frame.columns):
        return []
    counts = records_frame.groupby(key_columns, dropna=False).size()
    duplicated = counts[counts > 1]
    return [
        QualityIssue(
            check_name="duplicate_matches",
            severity="warning",
            message=(
                f"{home} vs {away} on {date} appears {count} times in records_frame."
            ),
            subject=(date, home, away),
            details={"count": int(count)},
        )
        for (date, home, away), count in duplicated.items()
    ]


def find_duplicate_match_ids(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag any `match_id` value used by more than one row -- must be a stable key."""
    if "match_id" not in records_frame.columns:
        return []
    counts = records_frame["match_id"].value_counts()
    duplicated = counts[counts > 1]
    return [
        QualityIssue(
            check_name="duplicate_match_ids",
            severity="error",
            message=f"match_id={match_id} is used by {count} rows.",
            subject=match_id,
            details={"count": int(count)},
        )
        for match_id, count in duplicated.items()
    ]


def find_duplicate_events(goalscorers_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag exact full-row duplicates in a goalscorers-shaped frame.

    Two different players genuinely scoring in the same match/minute is
    real and must NOT be flagged -- only an identical row (same match,
    team, scorer, minute, own_goal, penalty) appearing twice is a duplicate.
    """
    duplicated_mask = goalscorers_frame.duplicated(keep=False)
    duplicated_rows = goalscorers_frame[duplicated_mask]
    issues = []
    for index, row in duplicated_rows.iterrows():
        issues.append(
            QualityIssue(
                check_name="duplicate_events",
                severity="warning",
                message=(
                    f"Row at index {index} duplicates another event row exactly: "
                    f"{row.to_dict()}"
                ),
                subject=index,
            )
        )
    return issues


def find_self_matches(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag rows where home_team_name == away_team_name -- structurally impossible."""
    if not {"home_team_name", "away_team_name"}.issubset(records_frame.columns):
        return []
    mask = records_frame["home_team_name"] == records_frame["away_team_name"]
    return [
        QualityIssue(
            check_name="self_match",
            severity="error",
            message=f"match_id={row.get('match_id', index)}: {row['home_team_name']} "
            "played itself.",
            subject=row.get("match_id", index),
        )
        for index, row in records_frame[mask].iterrows()
    ]


def find_impossible_scores(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag negative or non-integer home_score/away_score values.

    Deliberately no upper bound -- the real data already contains a
    legitimate 31-0 (Australia v American Samoa, 2001), so "very high"
    isn't itself evidence of a problem.
    """
    issues: list[QualityIssue] = []
    for side in ("home_score", "away_score"):
        if side not in records_frame.columns:
            continue
        column = records_frame[side].dropna()
        invalid = column[(column < 0) | (column % 1 != 0)]
        for index, value in invalid.items():
            issues.append(
                QualityIssue(
                    check_name="impossible_score",
                    severity="error",
                    message=f"{side}={value!r} at index {index} is not a valid score.",
                    subject=records_frame.loc[index].get("match_id", index),
                    details={"column": side, "value": value},
                )
            )
    return issues


def find_negative_statistics(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag negative values in any home_*/away_* team-statistic column present."""
    issues: list[QualityIssue] = []
    for side in ("home", "away"):
        for stat in _TEAM_STAT_COLUMNS:
            column_name = f"{side}_{stat}"
            if column_name not in records_frame.columns:
                continue
            column = records_frame[column_name].dropna()
            negative = column[column < 0]
            for index, value in negative.items():
                issues.append(
                    QualityIssue(
                        check_name="negative_statistic",
                        severity="error",
                        message=f"{column_name}={value!r} at row {index} is negative.",
                        subject=records_frame.loc[index].get("match_id", index),
                        details={"column": column_name, "value": value},
                    )
                )
    return issues


def find_invalid_possession(records_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag possession values outside [0, 100], or pairs that don't sum to ~100.

    Tolerance is +/-1.5 around the real observed range (100.00-101.00 across
    841 real rows, 2026-08-20) -- wide enough for legitimate rounding, tight
    enough to catch a genuinely broken pair (e.g. 60/70).
    """
    issues: list[QualityIssue] = []
    if {"home_possession_pct", "away_possession_pct"}.issubset(records_frame.columns):
        for side in ("home_possession_pct", "away_possession_pct"):
            column = records_frame[side].dropna()
            out_of_range = column[(column < 0) | (column > 100)]
            for index, value in out_of_range.items():
                issues.append(
                    QualityIssue(
                        check_name="possession_out_of_range",
                        severity="error",
                        message=f"{side}={value!r} at row {index} is outside [0, 100].",
                        subject=records_frame.loc[index].get("match_id", index),
                        details={"column": side, "value": value},
                    )
                )

        both_known = records_frame.dropna(
            subset=["home_possession_pct", "away_possession_pct"]
        )
        total = both_known["home_possession_pct"] + both_known["away_possession_pct"]
        off_target = both_known[(total - 100).abs() > _POSSESSION_SUM_TOLERANCE]
        for index, row in off_target.iterrows():
            total_pct = row["home_possession_pct"] + row["away_possession_pct"]
            issues.append(
                QualityIssue(
                    check_name="possession_does_not_sum_to_100",
                    severity="error",
                    message=(
                        f"match_id={row.get('match_id', index)}: possession sums "
                        f"to {total_pct:.2f}, not ~100."
                    ),
                    subject=row.get("match_id", index),
                )
            )
    return issues


def find_shots_less_than_shots_on_target(
    records_frame: pd.DataFrame,
) -> list[QualityIssue]:
    """Flag shots_on_target > shots for either side -- physically impossible."""
    issues: list[QualityIssue] = []
    for side in ("home", "away"):
        shots_col, sot_col = f"{side}_shots", f"{side}_shots_on_target"
        if shots_col not in records_frame.columns:
            continue
        if sot_col not in records_frame.columns:
            continue
        both_known = records_frame.dropna(subset=[shots_col, sot_col])
        invalid = both_known[both_known[sot_col] > both_known[shots_col]]
        for index, row in invalid.iterrows():
            issues.append(
                QualityIssue(
                    check_name="shots_on_target_exceeds_shots",
                    severity="error",
                    message=(
                        f"match_id={row.get('match_id', index)}: {side} "
                        f"{sot_col}={row[sot_col]} > {shots_col}={row[shots_col]}."
                    ),
                    subject=row.get("match_id", index),
                )
            )
    return issues


def find_date_inconsistencies(
    records_frame: pd.DataFrame, as_of: pd.Timestamp
) -> list[QualityIssue]:
    """Flag rows whose `match_date` is after `as_of` -- a match can't be in the future.

    `as_of` is a required parameter, not `pd.Timestamp.now()` internally --
    keeps this function deterministic and testable without mocking the clock.
    """
    if "match_date" not in records_frame.columns:
        return []
    future = records_frame[records_frame["match_date"] > as_of]
    return [
        QualityIssue(
            check_name="future_match_date",
            severity="error",
            message=(
                f"match_id={row.get('match_id', index)}: match_date "
                f"{row['match_date']} is after as_of={as_of}."
            ),
            subject=row.get("match_id", index),
        )
        for index, row in future.iterrows()
    ]


def find_impossible_event_timestamps(
    goalscorers_frame: pd.DataFrame, max_minute: int = _MAX_PLAUSIBLE_EVENT_MINUTE
) -> list[QualityIssue]:
    """Flag negative or implausibly large event minutes.

    `max_minute` defaults to 130 -- real data (47,914 goal events,
    2026-08-20) tops out at 122 (extra time plus generous stoppage); a
    missing minute (NaN, 254 of those real rows) is a coverage gap, not a
    violation, and is skipped here rather than flagged.
    """
    if "minute" not in goalscorers_frame.columns:
        return []
    known = goalscorers_frame.dropna(subset=["minute"])
    invalid = known[(known["minute"] < 0) | (known["minute"] > max_minute)]
    return [
        QualityIssue(
            check_name="impossible_event_timestamp",
            severity="error",
            message=(
                f"match_id={row.get('match_id', index)}: event minute "
                f"{row['minute']} is out of the plausible [0, {max_minute}] range."
            ),
            subject=row.get("match_id", index),
        )
        for index, row in invalid.iterrows()
    ]


def run_all_checks(
    records_frame: pd.DataFrame,
    goalscorers_frame: pd.DataFrame | None = None,
    as_of: pd.Timestamp | None = None,
) -> list[QualityIssue]:
    """Run every records_frame/goalscorers_frame check and return the combined findings.

    `goalscorers_frame`/`as_of` are optional so this can be run against just
    a results frame with no event data yet -- event and future-date checks
    are simply skipped rather than raising.
    """
    issues = [
        *find_duplicate_matches(records_frame),
        *find_duplicate_match_ids(records_frame),
        *find_self_matches(records_frame),
        *find_impossible_scores(records_frame),
        *find_negative_statistics(records_frame),
        *find_invalid_possession(records_frame),
        *find_shots_less_than_shots_on_target(records_frame),
    ]
    if as_of is not None:
        issues.extend(find_date_inconsistencies(records_frame, as_of))
    if goalscorers_frame is not None:
        issues.extend(find_duplicate_events(goalscorers_frame))
        issues.extend(find_impossible_event_timestamps(goalscorers_frame))
    return issues
