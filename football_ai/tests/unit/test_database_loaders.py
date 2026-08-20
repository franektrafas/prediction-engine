"""Tests for the Canonical Match Model + Data Provenance Layer loaders."""

from __future__ import annotations

import unittest
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from football_ai.database import models  # noqa: F401 -- registers tables on Base
from football_ai.database.base import Base
from football_ai.database.loaders import (
    load_match_events,
    load_matches,
    load_penalty_shootouts,
    record_provenance,
)
from football_ai.database.models import Match, MatchEvent, MatchStatisticSource

# Real 2006 FIFA World Cup Final facts (Italy 1-1 France, Italy won on
# penalties) -- used as a golden fixture, same discipline as the rest of
# this project's tests (known, checkable real-world facts, not fabricated).
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
        },
        {
            "match_id": 30115,
            "team_name": "Italy",
            "scorer": "Marco Materazzi",
            "minute": 19.0,
            "own_goal": False,
            "penalty": False,
        },
        # A match_id not present in matches_by_id -- must be skipped, not raise.
        {
            "match_id": 999999,
            "team_name": "Nowhereland",
            "scorer": "Nobody",
            "minute": 1.0,
            "own_goal": False,
            "penalty": False,
        },
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


class DatabaseLoadersTestCase(unittest.TestCase):
    """Validate the loaders against real historical facts, fully offline (SQLite)."""

    def setUp(self) -> None:
        """Build a fresh in-memory schema for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session = Session(self.engine)

    def tearDown(self) -> None:
        """Close the session after each test."""
        self.session.close()

    def test_load_matches_creates_team_and_match_rows(self) -> None:
        """A single real match should create two teams and one match row."""
        matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()

        self.assertEqual(len(matches_by_id), 1)
        match = matches_by_id[30115]
        self.assertEqual(match.tournament, "FIFA World Cup")
        self.assertEqual(match.home_score, 1)
        self.assertEqual(match.away_score, 1)
        self.assertTrue(match.neutral_venue)
        self.assertEqual(match.source, "martj42")
        self.assertEqual(match.external_id, "30115")

    def test_load_matches_is_idempotent(self) -> None:
        """Loading the same records_frame twice must not create duplicate matches."""
        load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()
        load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()

        all_matches = self.session.execute(select(Match)).scalars().all()
        self.assertEqual(len(all_matches), 1)

    def test_load_match_events_classifies_goal_types_correctly(self) -> None:
        """Zidane's goal is a penalty, Materazzi's is a plain goal."""
        matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        inserted = load_match_events(self.session, _GOALSCORERS_FRAME, matches_by_id)
        self.session.commit()

        # 3 rows in the fixture, but one has an unloaded match_id -- skipped.
        self.assertEqual(inserted, 2)
        events = self.session.execute(select(MatchEvent)).scalars().all()
        events_by_scorer = {event.player_name: event for event in events}
        self.assertEqual(events_by_scorer["Zinedine Zidane"].event_type, "penalty")
        self.assertEqual(events_by_scorer["Zinedine Zidane"].minute, 7)
        self.assertEqual(events_by_scorer["Marco Materazzi"].event_type, "goal")

    def test_load_match_events_skips_events_for_unloaded_matches(self) -> None:
        """An event referencing a match_id outside matches_by_id must not raise."""
        matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        load_match_events(self.session, _GOALSCORERS_FRAME, matches_by_id)
        self.session.commit()
        # "Nowhereland" scorer's match_id=999999 was never loaded -- must be absent.
        names = {
            row.player_name
            for row in self.session.execute(select(MatchEvent)).scalars().all()
        }
        self.assertNotIn("Nobody", names)

    def test_load_penalty_shootouts_flags_the_real_winner(self) -> None:
        """Italy's 2006 shootout win over France must set the winner team id."""
        matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        updated = load_penalty_shootouts(self.session, _SHOOTOUTS_FRAME, matches_by_id)
        self.session.commit()

        self.assertEqual(updated, 1)
        match = matches_by_id[30115]
        self.assertTrue(match.penalty_shootout)
        winner_team = self.session.get(
            models.Team, match.penalty_shootout_winner_team_id
        )
        self.assertEqual(winner_team.name, "Italy")

    def test_load_penalty_shootouts_defaults_to_false_when_unmatched(self) -> None:
        """A match absent from shootouts.csv must keep penalty_shootout=False."""
        matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        empty_shootouts = _SHOOTOUTS_FRAME.iloc[0:0]
        updated = load_penalty_shootouts(self.session, empty_shootouts, matches_by_id)
        self.session.commit()

        self.assertEqual(updated, 0)
        self.assertFalse(matches_by_id[30115].penalty_shootout)


class RecordProvenanceTestCase(unittest.TestCase):
    """Validate the Data Provenance Layer's conflict-preserving design."""

    def setUp(self) -> None:
        """Build a fresh in-memory schema plus one loaded match for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)
        self.session = Session(self.engine)
        self.matches_by_id = load_matches(self.session, _RECORDS_FRAME)
        self.session.commit()
        self.match = self.matches_by_id[30115]

    def tearDown(self) -> None:
        """Close the session after each test."""
        self.session.close()

    def test_records_a_provenance_row(self) -> None:
        """A basic provenance row round-trips all its fields."""
        row = record_provenance(
            self.session,
            match=self.match,
            statistic_name="final_score",
            home_value=1.0,
            away_value=1.0,
            source="martj42",
            source_license="CC0",
            license_status="VERIFIED",
            confidence="HIGH",
            retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
        )
        self.session.commit()
        self.assertIsNotNone(row.id)
        self.assertEqual(row.match_id, self.match.id)
        self.assertEqual(row.statistic_name, "final_score")

    def test_two_sources_disagreeing_are_both_preserved(self) -> None:
        """Per the project's Source Priority rule: never drop a conflicting source."""
        record_provenance(
            self.session,
            match=self.match,
            statistic_name="attendance",
            home_value=69000.0,
            away_value=None,
            source="martj42",
            source_license="CC0",
            license_status="VERIFIED",
            confidence="HIGH",
            retrieved_at=datetime(2026, 8, 20, 0, 0, 0, tzinfo=UTC),
        )
        record_provenance(
            self.session,
            match=self.match,
            statistic_name="attendance",
            home_value=70000.0,
            away_value=None,
            source="unverified_scraper",
            source_license="unknown",
            license_status="UNKNOWN",
            confidence="LOW",
            retrieved_at=datetime(2026, 8, 20, 0, 0, 1, tzinfo=UTC),
        )
        self.session.commit()

        rows = (
            self.session.execute(
                select(MatchStatisticSource).where(
                    MatchStatisticSource.match_id == self.match.id,
                    MatchStatisticSource.statistic_name == "attendance",
                )
            )
            .scalars()
            .all()
        )
        self.assertEqual(len(rows), 2)
        values = {row.source: row.home_value for row in rows}
        self.assertEqual(values["martj42"], 69000.0)
        self.assertEqual(values["unverified_scraper"], 70000.0)

    def test_invalid_confidence_is_rejected_by_the_database(self) -> None:
        """The CHECK constraint, not just a type hint, must reject a bad tier."""
        with self.assertRaises(IntegrityError):
            record_provenance(
                self.session,
                match=self.match,
                statistic_name="final_score",
                home_value=1.0,
                away_value=1.0,
                source="martj42",
                source_license="CC0",
                license_status="VERIFIED",
                confidence="SUPER_DUPER_SURE",  # not a real tier
                retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
            )

    def test_invalid_license_status_is_rejected_by_the_database(self) -> None:
        """Same CHECK-constraint enforcement for license_status."""
        with self.assertRaises(IntegrityError):
            record_provenance(
                self.session,
                match=self.match,
                statistic_name="final_score",
                home_value=1.0,
                away_value=1.0,
                source="martj42",
                source_license="CC0",
                license_status="TOTALLY_FINE",  # not a real status
                confidence="HIGH",
                retrieved_at=datetime(2026, 8, 20, tzinfo=UTC),
            )


if __name__ == "__main__":
    unittest.main()
