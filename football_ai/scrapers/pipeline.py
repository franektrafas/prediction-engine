"""Automated data ingestion pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from sqlalchemy import select

from football_ai.database.models import Match
from football_ai.database.session import SessionLocal
from football_ai.scrapers.sources.elo import WorldFootballEloSource
from football_ai.scrapers.sources.fbref import FBrefSource
from football_ai.scrapers.sources.football_data import FootballDataSource
from football_ai.scrapers.sources.kaggle_importer import KaggleFileImporter
from football_ai.scrapers.sources.open_meteo import OpenMeteoSource
from football_ai.scrapers.sources.statsbomb import StatsBombOpenDataSource


@dataclass(slots=True)
class DataIngestionPipeline:
    """High-level pipeline orchestrating source imports."""

    statsbomb_source: StatsBombOpenDataSource
    football_data_source: FootballDataSource
    elo_source: WorldFootballEloSource
    open_meteo_source: OpenMeteoSource
    fbref_source: FBrefSource
    kaggle_source: KaggleFileImporter

    @classmethod
    def create(cls) -> "DataIngestionPipeline":
        """Build the default ingestion pipeline."""
        return cls(
            statsbomb_source=StatsBombOpenDataSource(),
            football_data_source=FootballDataSource(),
            elo_source=WorldFootballEloSource(),
            open_meteo_source=OpenMeteoSource(),
            fbref_source=FBrefSource(),
            kaggle_source=KaggleFileImporter(),
        )

    def ingest_statsbomb_competition(self, competition_id: int, season_id: int) -> None:
        """Ingest a StatsBomb competition and season into the database."""
        matches = self.statsbomb_source.fetch_matches(competition_id, season_id)
        normalized = self.statsbomb_source.normalize_matches(matches)
        events_by_match = {
            row["external_id"]: self.statsbomb_source.fetch_events(int(row["external_id"]))
            for row in normalized.to_dict(orient="records")
        }
        with SessionLocal() as session:
            self.statsbomb_source.ingest_matches_to_database(session, normalized, events_by_match)

    def ingest_football_data_csv(self, csv_path: str | Path) -> None:
        """Ingest a local Football-Data CSV into the database."""
        normalized = self.football_data_source.load_csv(csv_path)
        with SessionLocal() as session:
            self.football_data_source.ingest_matches_to_database(session, normalized)

    def ingest_elo_snapshot(self, year: int) -> None:
        """Ingest a historical Elo snapshot."""
        snapshot = self.elo_source.fetch_year_snapshot(year)
        with SessionLocal() as session:
            self.elo_source.ingest_snapshot_to_database(session, snapshot)

    def enrich_weather(self) -> None:
        """Enrich matches with historical weather snapshots."""
        with SessionLocal() as session:
            matches = list(
                session.execute(
                    select(Match).where(
                        Match.venue_latitude.is_not(None),
                        Match.venue_longitude.is_not(None),
                    )
                ).scalars()
            )
            self.open_meteo_source.enrich_matches(session, matches)

    def import_fbref_file(self, path: str | Path) -> None:
        """Normalize and persist an FBref export."""
        self.fbref_source.import_from_file(path)

    def import_kaggle_file(self, path: str | Path) -> None:
        """Normalize and persist a user-provided Kaggle file."""
        self.kaggle_source.import_file(path)

    def run_scheduled_refresh(self) -> None:
        """Execute configured refresh steps from environment variables."""
        statsbomb_competition = os.getenv("STATSBOMB_COMPETITION_ID")
        statsbomb_season = os.getenv("STATSBOMB_SEASON_ID")
        football_data_csv = os.getenv("FOOTBALL_DATA_CSV_PATH")
        elo_year = os.getenv("ELO_SNAPSHOT_YEAR")
        fbref_file = os.getenv("FBREF_FILE_PATH")
        kaggle_file = os.getenv("KAGGLE_FILE_PATH")
        refresh_weather = os.getenv("REFRESH_WEATHER", "false").lower() == "true"

        if statsbomb_competition and statsbomb_season:
            self.ingest_statsbomb_competition(
                competition_id=int(statsbomb_competition),
                season_id=int(statsbomb_season),
            )
        if football_data_csv:
            self.ingest_football_data_csv(football_data_csv)
        if elo_year:
            self.ingest_elo_snapshot(int(elo_year))
        if fbref_file:
            self.import_fbref_file(fbref_file)
        if kaggle_file:
            self.import_kaggle_file(kaggle_file)
        if refresh_weather:
            self.enrich_weather()


if __name__ == "__main__":
    pipeline = DataIngestionPipeline.create()
    pipeline.run_scheduled_refresh()
    print("Scheduled refresh completed.")
