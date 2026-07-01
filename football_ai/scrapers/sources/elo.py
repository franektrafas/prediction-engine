"""World Football Elo ratings importer."""

from __future__ import annotations

from datetime import date

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data.normalization import canonicalize_team_name
from football_ai.database.models import EloRating, Team
from football_ai.scrapers.base import BaseDataSource


class WorldFootballEloSource(BaseDataSource):
    """Importer for World Football Elo ratings snapshots."""

    BASE_URL = "https://eloratings.net"

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="world_football_elo")

    def fetch_year_snapshot(self, year: int) -> pd.DataFrame:
        """Fetch an Elo table snapshot for a given year."""
        frame = self.fetch_table(f"{self.BASE_URL}/{year}")
        normalized = self.normalize_snapshot(frame, snapshot_date=date(year, 12, 31))
        self.persist_frame(normalized, f"elo_{year}.parquet")
        return normalized

    def normalize_snapshot(
        self,
        frame: pd.DataFrame,
        snapshot_date: date,
    ) -> pd.DataFrame:
        """Normalize a historical Elo table."""
        renamed = frame.rename(
            columns={
                "Rank": "rank",
                "Team": "team_name",
                "Rating": "rating",
            }
        )
        normalized = renamed.copy()
        normalized["team_name"] = normalized["team_name"].astype(str).map(
            canonicalize_team_name
        )
        normalized["rating_date"] = snapshot_date
        normalized["source"] = "world_football_elo"
        return normalized[[column for column in ["rank", "team_name", "rating", "rating_date", "source"] if column in normalized.columns]]

    def ingest_snapshot_to_database(self, session: Session, snapshot: pd.DataFrame) -> None:
        """Persist a normalized Elo snapshot to the database."""
        for row in snapshot.to_dict(orient="records"):
            team = session.query(Team).filter(Team.name == row["team_name"]).one_or_none()
            if team is None:
                team = Team(name=row["team_name"], current_elo=row["rating"])
                session.add(team)
                session.flush()
            team.current_elo = row["rating"]
            session.add(
                EloRating(
                    team_id=team.id,
                    rating_date=row["rating_date"],
                    rating=float(row["rating"]),
                    rank=int(row["rank"]) if pd.notna(row.get("rank")) else None,
                    source="world_football_elo",
                )
            )
        session.commit()


if __name__ == "__main__":
    source = WorldFootballEloSource()
    print(source.fetch_year_snapshot(2024).head())
