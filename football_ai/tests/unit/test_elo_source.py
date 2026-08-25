"""Tests for football_ai/scrapers/sources/elo.py."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from football_ai.database import models  # noqa: F401 -- registers tables on Base
from football_ai.database.base import Base
from football_ai.database.models import EloRating, Team
from football_ai.scrapers.sources.elo import WorldFootballEloSource


class LatestRatingsByTeamTestCase(unittest.TestCase):
    """Validate the snapshot -> {team: rating} reshape used to feed live fixture predictions."""

    def test_builds_a_simple_team_to_rating_lookup(self) -> None:
        snapshot = pd.DataFrame(
            [
                {"rank": 1, "team_name": "Spain", "rating": 2259.0},
                {"rank": 43, "team_name": "Poland", "rating": 1710.0},
            ]
        )
        ratings = WorldFootballEloSource.latest_ratings_by_team(snapshot)
        self.assertEqual(ratings, {"Spain": 2259.0, "Poland": 1710.0})

    def test_empty_snapshot_returns_an_empty_dict(self) -> None:
        empty = pd.DataFrame(columns=["rank", "team_name", "rating"])
        self.assertEqual(WorldFootballEloSource.latest_ratings_by_team(empty), {})


class IngestSnapshotToDatabaseTestCase(unittest.TestCase):
    """Regression tests for the real UniqueViolation hit 2026-08-25.

    The first real `launchd`-triggered run failed: a plain INSERT on
    `(team_id, rating_date, source)` raised `UniqueViolation` the moment
    this ran twice on the same calendar day (an earlier manual test run,
    then the scheduled run), aborting the whole batch. Fixed to upsert.
    """

    def setUp(self) -> None:
        """Build a fresh in-memory schema for each test."""
        self.engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=self.engine)

    def _snapshot(self, rating: float, rank: int = 1) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "team_name": "Spain",
                    "rating": rating,
                    "rank": rank,
                    "rating_date": date(2026, 8, 25),
                }
            ]
        )

    def test_first_ingest_creates_the_team_and_the_rating_row(self) -> None:
        """A brand-new team/date pair inserts cleanly."""
        with Session(self.engine) as session:
            WorldFootballEloSource().ingest_snapshot_to_database(
                session, self._snapshot(2259.0)
            )
            team = session.execute(select(Team)).scalars().one()
            self.assertEqual(team.current_elo, 2259.0)
            rating_row = session.execute(select(EloRating)).scalars().one()
            self.assertEqual(rating_row.rating, 2259.0)

    def test_same_day_reingest_updates_in_place_instead_of_raising(self) -> None:
        """Real bug: re-running the same day used to raise UniqueViolation.

        Real-world trigger: a manual `launchctl start` test run, then the
        scheduled run firing later the same day -- must update the
        existing row, not crash or create a duplicate.
        """
        with Session(self.engine) as session:
            source = WorldFootballEloSource()
            source.ingest_snapshot_to_database(session, self._snapshot(2259.0, rank=1))
            source.ingest_snapshot_to_database(session, self._snapshot(2270.0, rank=2))

            rating_rows = session.execute(select(EloRating)).scalars().all()
            self.assertEqual(len(rating_rows), 1)
            self.assertEqual(rating_rows[0].rating, 2270.0)
            self.assertEqual(rating_rows[0].rank, 2)

            team = session.execute(select(Team)).scalars().one()
            self.assertEqual(team.current_elo, 2270.0)

    def test_a_new_day_still_inserts_a_second_row(self) -> None:
        """A genuinely new rating_date must add a row, not overwrite the old one."""
        with Session(self.engine) as session:
            source = WorldFootballEloSource()
            source.ingest_snapshot_to_database(session, self._snapshot(2259.0))
            next_day = self._snapshot(2265.0)
            next_day["rating_date"] = date(2026, 8, 26)
            source.ingest_snapshot_to_database(session, next_day)

            rating_rows = session.execute(select(EloRating)).scalars().all()
            self.assertEqual(len(rating_rows), 2)


if __name__ == "__main__":
    unittest.main()
