"""Integrity checks for `match_sources`-shaped provenance data (Faza 7's ledger).

Operates on a plain DataFrame with the same columns as
`database.models.MatchStatisticSource` (match_id, statistic_name,
home_value, away_value, source, source_license, license_status,
confidence) -- not a live database session, so these checks run just as
well on rows about to be inserted as on rows already persisted, and are
testable without a database.
"""

from __future__ import annotations

import pandas as pd

from football_ai.data_quality.schemas import QualityIssue

_VALUE_AGREEMENT_TOLERANCE = 1e-6


def find_missing_source(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag rows with a null/blank `source`."""
    if "source" not in provenance_frame.columns:
        return []
    missing = provenance_frame[
        provenance_frame["source"].isna() | (provenance_frame["source"] == "")
    ]
    return [
        QualityIssue(
            check_name="missing_source",
            severity="error",
            message=f"Row at index {index} has no source for "
            f"statistic {row.get('statistic_name')!r}.",
            subject=row.get("match_id", index),
        )
        for index, row in missing.iterrows()
    ]


def find_missing_license_status(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag rows with a null/blank `license_status`.

    A row missing this entirely (as opposed to carrying the real value
    "UNKNOWN") means the ingestion step itself forgot to classify it --
    worth catching before the row reaches a production dataset, since an
    unclassified row can't be excluded from ML training even if it should be.
    """
    if "license_status" not in provenance_frame.columns:
        return []
    missing = provenance_frame[
        provenance_frame["license_status"].isna()
        | (provenance_frame["license_status"] == "")
    ]
    return [
        QualityIssue(
            check_name="missing_license_status",
            severity="error",
            message=f"Row at index {index} (source={row.get('source')!r}) has no "
            "license_status.",
            subject=row.get("match_id", index),
        )
        for index, row in missing.iterrows()
    ]


def find_unverified_licenses(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag UNKNOWN/RESEARCH_ONLY rows -- informational, not an error.

    Per the project's own rule, these aren't necessarily wrong, but they
    must never silently feed production ML -- this surfaces them so a
    caller can filter before training, rather than treating "present in
    the table" as "safe to use". FORBIDDEN rows are excluded here --
    they're not a softer "not yet verified" case, they're a confirmed
    violation, and get their own louder check (`find_forbidden_licenses`)
    instead of being lumped into this warning-level one.
    """
    if "license_status" not in provenance_frame.columns:
        return []
    unverified = provenance_frame[
        ~provenance_frame["license_status"].isin(["VERIFIED", "FORBIDDEN"])
    ]
    return [
        QualityIssue(
            check_name="unverified_license",
            severity="warning",
            message=(
                f"match_id={row.get('match_id', index)}, "
                f"statistic={row.get('statistic_name')!r}: source "
                f"{row.get('source')!r} has license_status="
                f"{row.get('license_status')!r}, not VERIFIED."
            ),
            subject=row.get("match_id", index),
        )
        for index, row in unverified.iterrows()
    ]


def find_forbidden_licenses(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag rows whose license_status is FORBIDDEN -- a confirmed violation.

    Distinct from `find_unverified_licenses`: this isn't "we haven't
    checked yet", it's "we checked, and this source (e.g. ESPN under
    Disney's Terms of Use, confirmed 2026-08-20) explicitly bars this use".
    Any of these rows reaching a production training dataset is a real bug,
    not a caution -- hence "error", matching this project's other
    factually-impossible-value checks.
    """
    if "license_status" not in provenance_frame.columns:
        return []
    forbidden = provenance_frame[provenance_frame["license_status"] == "FORBIDDEN"]
    return [
        QualityIssue(
            check_name="forbidden_license_used",
            severity="error",
            message=(
                f"match_id={row.get('match_id', index)}, "
                f"statistic={row.get('statistic_name')!r}: source "
                f"{row.get('source')!r} is FORBIDDEN -- must never feed "
                "production ML."
            ),
            subject=row.get("match_id", index),
        )
        for index, row in forbidden.iterrows()
    ]


def find_conflicting_sources(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Flag (match_id, statistic_name) groups where sources disagree on the value.

    Per the project's Source Priority rule, a conflict is never a reason to
    drop either source -- this only *detects and records* the disagreement
    (as a "warning"-severity QualityIssue) so a caller can resolve it by
    priority at query time. Two sources reporting the *same* value are not
    a conflict, even though they're still two separate provenance rows.
    """
    required = {"match_id", "statistic_name", "source", "home_value", "away_value"}
    if not required.issubset(provenance_frame.columns):
        return []

    issues: list[QualityIssue] = []
    grouped = provenance_frame.groupby(["match_id", "statistic_name"], dropna=False)
    for (match_id, statistic_name), group in grouped:
        if group["source"].nunique() < 2:
            continue
        if _values_agree(group["home_value"]) and _values_agree(group["away_value"]):
            continue
        issues.append(
            QualityIssue(
                check_name="conflicting_sources",
                severity="warning",
                message=(
                    f"match_id={match_id}, statistic={statistic_name!r}: "
                    f"{group['source'].nunique()} sources disagree "
                    f"({sorted(group['source'].unique())})."
                ),
                subject=(match_id, statistic_name),
                details={
                    "sources": sorted(group["source"].unique().tolist()),
                    "home_values": group["home_value"].tolist(),
                    "away_values": group["away_value"].tolist(),
                },
            )
        )
    return issues


def _values_agree(values: pd.Series) -> bool:
    """True if every non-null value in `values` is within tolerance of the first."""
    known = values.dropna()
    if len(known) <= 1:
        return True
    reference = known.iloc[0]
    return bool((known - reference).abs().le(_VALUE_AGREEMENT_TOLERANCE).all())


def run_all_checks(provenance_frame: pd.DataFrame) -> list[QualityIssue]:
    """Run every provenance check and return the combined findings."""
    return [
        *find_missing_source(provenance_frame),
        *find_missing_license_status(provenance_frame),
        *find_unverified_licenses(provenance_frame),
        *find_forbidden_licenses(provenance_frame),
        *find_conflicting_sources(provenance_frame),
    ]
