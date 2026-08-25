"""Faza 9 of the Data Engine plan: import real data into the live PostgreSQL database.

Loads martj42's results/goalscorers/shootouts (via `database.loaders`) plus
provenance rows for every statistic that has a real source -- final_score
from martj42 (VERIFIED, CC0), and any ESPN-sourced team stats found in
`records_frame` (shots/corners/cards/possession/...) tagged FORBIDDEN.

Confirmed 2026-08-20 against Disney's Terms of Use (ESPN is a Disney
property; `scrapers/sources/espn.py`'s own docstring already called it
"not an official/published API", but the operative text is stronger than
that: "no right to reproduce, distribute... or transform any Disney
Product, including in connection with any... training, testing,
benchmarking or validation of any artificial intelligence or machine
learning tool" plus a blanket "we do not allow uses... that are commercial
or business-related" -- not silence, an explicit bar with no
research-only carve-out. This is exactly why the provenance ledger exists
instead of one flat table: the identity/score data in the same wide row
stays VERIFIED, only the ESPN-derived stat columns are FORBIDDEN.

Faza 10 (coverage report) runs automatically at the end via
`data_quality.report`.
"""

from __future__ import annotations

import argparse
import logging
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data_quality import report as quality_report
from football_ai.database.loaders import (
    MARTJ42_LICENSE,
    MARTJ42_SOURCE,
    MARTJ42_URL,
    load_match_events,
    load_matches,
    load_penalty_shootouts,
    record_provenance,
)
from football_ai.database.session import ENGINE, create_database
from football_ai.scrapers.sources.international_results import (
    InternationalResultsSource,
)

LOGGER = logging.getLogger(__name__)

# ESPN's site API is real data, but confirmed FORBIDDEN for this use (see
# module docstring: Disney's Terms of Use bars both AI/ML training and
# commercial use outright) -- provenance rows for its stats must say so,
# not inherit martj42's VERIFIED status just because they live in the
# same wide row.
_ESPN_SOURCE = "espn_undocumented_api"
_ESPN_LICENSE = "unknown (Disney ToU bars this use -- see module docstring)"
_ESPN_STAT_COLUMNS = {
    "shots": ("home_shots", "away_shots"),
    "shots_on_target": ("home_shots_on_target", "away_shots_on_target"),
    "corners": ("home_corners", "away_corners"),
    "yellow_cards": ("home_yellow_cards", "away_yellow_cards"),
    "possession_pct": ("home_possession_pct", "away_possession_pct"),
}


def _record_final_score_provenance(
    session: Session,
    records_frame: pd.DataFrame,
    matches_by_id: dict,
    retrieved_at: datetime,
) -> int:
    """Record martj42 as the VERIFIED source for every match's final score."""
    count = 0
    for row in records_frame.itertuples(index=False):
        match = matches_by_id.get(row.match_id)
        if match is None or pd.isna(row.home_score) or pd.isna(row.away_score):
            continue
        record_provenance(
            session,
            match=match,
            statistic_name="final_score",
            home_value=float(row.home_score),
            away_value=float(row.away_score),
            source=MARTJ42_SOURCE,
            source_license=MARTJ42_LICENSE,
            source_url=MARTJ42_URL,
            license_status="VERIFIED",
            confidence="HIGH",
            retrieved_at=retrieved_at,
        )
        count += 1
    return count


def _record_espn_stat_provenance(
    session: Session,
    records_frame: pd.DataFrame,
    matches_by_id: dict,
    retrieved_at: datetime,
) -> int:
    """Record ESPN-sourced team statistics, marked FORBIDDEN, not VERIFIED."""
    available = {
        name: (home_col, away_col)
        for name, (home_col, away_col) in _ESPN_STAT_COLUMNS.items()
        if home_col in records_frame.columns and away_col in records_frame.columns
    }
    if not available:
        return 0

    count = 0
    for row in records_frame.itertuples(index=False):
        match = matches_by_id.get(row.match_id)
        if match is None:
            continue
        row_dict = row._asdict()
        for statistic_name, (home_col, away_col) in available.items():
            home_value, away_value = row_dict.get(home_col), row_dict.get(away_col)
            if pd.isna(home_value) or pd.isna(away_value):
                continue
            record_provenance(
                session,
                match=match,
                statistic_name=statistic_name,
                home_value=float(home_value),
                away_value=float(away_value),
                source=_ESPN_SOURCE,
                source_license=_ESPN_LICENSE,
                license_status="FORBIDDEN",
                confidence="LOW",
                retrieved_at=retrieved_at,
            )
            count += 1
    return count


def run_import(
    records_frame: pd.DataFrame,
    goalscorers_frame: pd.DataFrame | None,
    shootouts_frame: pd.DataFrame | None,
    include_provenance: bool = True,
) -> None:
    """Run the full Faza 9 import against the live database configured in settings."""
    retrieved_at = datetime.now(UTC)
    with Session(ENGINE) as session:
        matches_by_id = load_matches(session, records_frame)
        session.commit()
        LOGGER.info("Loaded/found %d matches.", len(matches_by_id))

        if goalscorers_frame is not None:
            n_events = load_match_events(session, goalscorers_frame, matches_by_id)
            session.commit()
            LOGGER.info("Loaded %d match events.", n_events)

        if shootouts_frame is not None:
            n_shootouts = load_penalty_shootouts(
                session, shootouts_frame, matches_by_id
            )
            session.commit()
            LOGGER.info("Flagged %d penalty shootouts.", n_shootouts)

        if include_provenance:
            n_scores = _record_final_score_provenance(
                session, records_frame, matches_by_id, retrieved_at
            )
            n_espn = _record_espn_stat_provenance(
                session, records_frame, matches_by_id, retrieved_at
            )
            session.commit()
            LOGGER.info(
                "Recorded %d final_score + %d ESPN-stat provenance rows.",
                n_scores,
                n_espn,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Faza 9: import martj42 data + provenance into PostgreSQL."
    )
    parser.add_argument(
        "--records-frame",
        default="football_ai/data/processed/international_results_full_history.parquet",
    )
    parser.add_argument(
        "--skip-events",
        action="store_true",
        help="Skip fetching/loading goalscorers.csv (network access).",
    )
    parser.add_argument(
        "--skip-shootouts",
        action="store_true",
        help="Skip fetching/loading shootouts.csv (network access).",
    )
    parser.add_argument(
        "--skip-provenance", action="store_true", help="Skip recording provenance rows."
    )
    arguments = parser.parse_args()

    log_format = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)

    create_database()
    records = pd.read_parquet(arguments.records_frame)

    source = InternationalResultsSource()
    goalscorers = None
    shootouts = None
    if not arguments.skip_events:
        goalscorers = source.normalize_goalscorers(source.fetch_goalscorers(), records)
    if not arguments.skip_shootouts:
        shootouts = source.normalize_shootouts(source.fetch_shootouts())

    run_import(
        records,
        goalscorers_frame=goalscorers,
        shootouts_frame=shootouts,
        include_provenance=not arguments.skip_provenance,
    )

    print("\n=== Faza 10: Data Quality Report (post-import) ===\n")
    generated = quality_report.generate_report(
        records, goalscorers_frame=goalscorers, as_of=pd.Timestamp.now(tz="UTC")
    )
    quality_report.print_report(generated)
