"""Tests for the Faza 9 import orchestration (database.import_records)."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from football_ai.database import models  # noqa: F401 -- registers tables on Base
from football_ai.database.base import Base
from football_ai.database.import_records import (
    _record_espn_stat_provenance,
    _record_final_score_provenance,
    run_import,
)
from football_ai.database.loaders import load_matches
from football_ai.database.models import Match, MatchEvent, MatchStatisticSource

_RECORDS_FRAME = pd.DataFrame(
    [
        {
            "match_id": 30115,
            "match_date": pd.Timestamp("2006-07-09", tz="UTC"),
            "home_team_name": "Italy",
            "away_team_name": "France",
            "home_score": 1,
            "away_score": 1,
            "tournament": "FIFA World Cup",
            "city": "Berlin",
            "country": "Germany",
            "neutral_venue": True,
            "home_shots": 15.0,
            "away_shots": 12.0,
        }
    ]
)

_GOALSCORERS_FRAME = pd.DataFrame(
    [
        {
            "match_id": 30115,
            "team_name": "France",
            "scorer": "Zinedine Zidane",
            "minute": 7.0,
            "own_goal": False,
            "penalty": True,
        }
    ]
)

_SHOOTOUTS_FRAME = pd.DataFrame(
    [
        {
            "match_date": pd.Timestamp("2006-07-09", tz="UTC"),
            "home_team_name": "Italy",
            "away_team_name": "France",
            "winner_team_name": "Italy",
        }
    ]
)


class RunImportTestCase(unittest.TestCase):
    """End-to-end orchestration, fully offline (SQLite in-memory)."""

    def setUp(self) -> None:
        """Build a fresh in-memory schema for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)

    def test_run_import_loads_matches_events_shootouts_and_provenance(self) -> None:
        """A run against the real 2006 World Cup Final fixture populates every table."""
        import football_ai.database.import_records as import_records_module

        original_engine = import_records_module.ENGINE
        import_records_module.ENGINE = self.engine
        try:
            run_import(_RECORDS_FRAME, _GOALSCORERS_FRAME, _SHOOTOUTS_FRAME)
        finally:
            import_records_module.ENGINE = original_engine

        with Session(self.engine) as session:
            match = session.execute(select(Match)).scalars().one()
            self.assertEqual(match.home_score, 1)
            self.assertTrue(match.penalty_shootout)

            events = session.execute(select(MatchEvent)).scalars().all()
            self.assertEqual(len(events), 1)
            self.assertEqual(events[0].event_type, "penalty")

            provenance = session.execute(select(MatchStatisticSource)).scalars().all()
            statistic_names = {row.statistic_name for row in provenance}
            self.assertIn("final_score", statistic_names)
            self.assertIn("shots", statistic_names)

    def test_run_import_without_provenance_skips_the_ledger(self) -> None:
        """include_provenance=False must leave match_sources empty."""
        import football_ai.database.import_records as import_records_module

        original_engine = import_records_module.ENGINE
        import_records_module.ENGINE = self.engine
        try:
            run_import(_RECORDS_FRAME, None, None, include_provenance=False)
        finally:
            import_records_module.ENGINE = original_engine

        with Session(self.engine) as session:
            provenance = session.execute(select(MatchStatisticSource)).scalars().all()
            self.assertEqual(provenance, [])


class RecordFinalScoreProvenanceTestCase(unittest.TestCase):
    """The final_score provenance is always VERIFIED/HIGH -- it's martj42's own data."""

    def setUp(self) -> None:
        """Build a fresh in-memory schema plus one loaded match for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session = Session(self.engine)
        self.matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()

    def tearDown(self) -> None:
        """Close the session after each test."""
        self.session.close()

    def test_records_verified_high_confidence_score(self) -> None:
        """The final score is always VERIFIED/HIGH -- it's martj42's own data."""
        retrieved_at = datetime(2026, 8, 20, tzinfo=UTC)
        count = _record_final_score_provenance(
            self.session, _RECORDS_FRAME, self.matches_by_id, retrieved_at
        )
        self.session.commit()

        self.assertEqual(count, 1)
        row = self.session.execute(select(MatchStatisticSource)).scalars().one()
        self.assertEqual(row.license_status, "VERIFIED")
        self.assertEqual(row.confidence, "HIGH")
        self.assertEqual(row.home_value, 1.0)
        self.assertEqual(row.away_value, 1.0)


class RecordEspnStatProvenanceTestCase(unittest.TestCase):
    """ESPN stats must never be marked VERIFIED -- their license was never confirmed."""

    def setUp(self) -> None:
        """Build a fresh in-memory schema plus one loaded match for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session = Session(self.engine)
        self.matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()

    def tearDown(self) -> None:
        """Close the session after each test."""
        self.session.close()

    def test_espn_stats_are_marked_unknown_not_verified(self) -> None:
        """ESPN's undocumented API is real data, but never VERIFIED."""
        retrieved_at = datetime(2026, 8, 20, tzinfo=UTC)
        count = _record_espn_stat_provenance(
            self.session, _RECORDS_FRAME, self.matches_by_id, retrieved_at
        )
        self.session.commit()

        self.assertEqual(count, 1)
        row = self.session.execute(select(MatchStatisticSource)).scalars().one()
        self.assertEqual(row.statistic_name, "shots")
        self.assertEqual(row.license_status, "UNKNOWN")
        self.assertEqual(row.confidence, "LOW")
        self.assertEqual(row.home_value, 15.0)
        self.assertEqual(row.away_value, 12.0)

    def test_missing_stat_columns_produce_no_rows(self) -> None:
        """No ESPN-shaped columns at all must yield zero provenance rows."""
        bare_frame = _RECORDS_FRAME[
            ["match_id", "match_date", "home_team_name", "away_team_name"]
        ]
        retrieved_at = datetime(2026, 8, 20, tzinfo=UTC)
        count = _record_espn_stat_provenance(
            self.session, bare_frame, self.matches_by_id, retrieved_at
        )
        self.assertEqual(count, 0)


if __name__ == "__main__":
    unittest.main()
