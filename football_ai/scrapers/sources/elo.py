"""World Football Elo ratings importer."""

from __future__ import annotations

from datetime import date
from io import StringIO

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
        frame = self._fetch_tsv_snapshot(f"{self.BASE_URL}/{year}.tsv")
        normalized = self.normalize_snapshot(frame, snapshot_date=date(year, 12, 31))
        self.persist_frame(normalized, f"elo_{year}.parquet")
        return normalized

    def fetch_current_snapshot(self) -> pd.DataFrame:
        """Fetch the latest global Elo snapshot."""
        frame = self._fetch_tsv_snapshot(f"{self.BASE_URL}/World.tsv")
        normalized = self.normalize_snapshot(frame, snapshot_date=date.today())
        self.persist_frame(normalized, "elo_current.parquet")
        return normalized

    def _fetch_tsv_snapshot(self, url: str) -> pd.DataFrame:
        """Fetch and parse a TSV snapshot."""
        team_mapping = self._fetch_team_mapping()
        text = self.fetch_text(url)
        frame = pd.read_csv(
            StringIO(text),
            sep="\t",
            header=None,
            engine="python",
        )
        if frame.shape[1] < 4:
            raise ValueError(f"Unexpected Elo snapshot format at {url}")
        frame = frame.rename(
            columns={
                1: "rank",
                2: "team_code",
                3: "rating",
            }
        )
        frame["team_name"] = frame["team_code"].astype(str).map(team_mapping)
        frame["team_name"] = frame["team_name"].fillna(frame["team_code"].astype(str))
        frame = frame[["rank", "team_code", "team_name", "rating"]].copy()
        return frame

    def _fetch_team_mapping(self) -> dict[str, str]:
        """Fetch the Elo team-code mapping."""
        text = self.fetch_text(f"{self.BASE_URL}/teams.tsv")
        mapping: dict[str, str] = {}
        for line in text.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            code = parts[0].strip()
            name = parts[1].strip()
            if code and name:
                mapping[code] = canonicalize_team_name(name)
        if not mapping:
            raise ValueError("Could not parse team mapping from teams.tsv")
        return mapping

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
        normalized["rating"] = pd.to_numeric(normalized["rating"], errors="coerce")
        normalized["rank"] = pd.to_numeric(normalized["rank"], errors="coerce")
        normalized["rating_date"] = snapshot_date
        normalized["source"] = "world_football_elo"
        normalized = normalized.dropna(subset=["team_name", "rating"])
        return normalized[
            [
                column
                for column in ["rank", "team_code", "team_name", "rating", "rating_date", "source"]
                if column in normalized.columns
            ]
        ]

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
