"""Football-Data.co.uk importer."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data.normalization import canonicalize_team_name
from football_ai.database.models import Match, Team, TeamStatistic
from football_ai.scrapers.base import BaseDataSource


class FootballDataSource(BaseDataSource):
    """Importer for Football-Data CSV datasets."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="football_data")

    def load_csv(self, csv_path: str | Path) -> pd.DataFrame:
        """Load a local Football-Data CSV export."""
        frame = pd.read_csv(csv_path)
        return self.normalize_matches(frame)

    def fetch_csv(self, url: str) -> pd.DataFrame:
        """Fetch a remote Football-Data CSV file."""
        frame = pd.read_csv(url)
        normalized = self.normalize_matches(frame)
        self.persist_frame(normalized, "normalized_matches.parquet")
        return normalized

    def normalize_matches(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize Football-Data columns into the project schema."""
        renamed = frame.rename(
            columns={
                "Date": "match_date",
                "HomeTeam": "home_team_name",
                "AwayTeam": "away_team_name",
                "FTHG": "home_score",
                "FTAG": "away_score",
                "HS": "home_shots",
                "AS": "away_shots",
                "HST": "home_shots_on_target",
                "AST": "away_shots_on_target",
                "HC": "home_corners",
                "AC": "away_corners",
                "HY": "home_yellow_cards",
                "AY": "away_yellow_cards",
                "HR": "home_red_cards",
                "AR": "away_red_cards",
            }
        )
        normalized = renamed.copy()
        normalized["source"] = "football_data"
        normalized["match_date"] = pd.to_datetime(normalized["match_date"], dayfirst=True)
        normalized["home_team_name"] = normalized["home_team_name"].map(canonicalize_team_name)
        normalized["away_team_name"] = normalized["away_team_name"].map(canonicalize_team_name)
        if "Div" in normalized.columns:
            normalized["competition_id"] = normalized["Div"].astype(str)
        return normalized

    def ingest_matches_to_database(self, session: Session, normalized: pd.DataFrame) -> None:
        """Persist normalized Football-Data rows to the database."""
        for row in normalized.to_dict(orient="records"):
            home_team = self._get_or_create_team(session, row["home_team_name"])
            away_team = self._get_or_create_team(session, row["away_team_name"])
            match = Match(
                external_id=str(
                    row.get(
                        "external_id",
                        f"{row['competition_id']}_{row['match_date']}_{row['home_team_name']}_{row['away_team_name']}",
                    )
                ),
                source="football_data",
                competition_id=str(row.get("competition_id", "")),
                competition_name=row.get("competition_name"),
                season_name=row.get("Season"),
                tournament=row.get("competition_name"),
                match_date=pd.to_datetime(row["match_date"], utc=True),
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                home_score=int(row["home_score"]) if pd.notna(row.get("home_score")) else None,
                away_score=int(row["away_score"]) if pd.notna(row.get("away_score")) else None,
            )
            session.add(match)
            session.flush()
            stats_mapping = [
                ("home", home_team, True),
                ("away", away_team, False),
            ]
            for prefix, team, is_home in stats_mapping:
                session.add(
                    TeamStatistic(
                        match_id=match.id,
                        team_id=team.id,
                        is_home=is_home,
                        goals=row.get(f"{prefix}_score"),
                        shots=row.get(f"{prefix}_shots"),
                        shots_on_target=row.get(f"{prefix}_shots_on_target"),
                        corners=row.get(f"{prefix}_corners"),
                        yellow_cards=row.get(f"{prefix}_yellow_cards"),
                        red_cards=row.get(f"{prefix}_red_cards"),
                    )
                )
        session.commit()

    def _get_or_create_team(self, session: Session, team_name: str) -> Team:
        """Fetch or create a team."""
        team = session.query(Team).filter(Team.name == team_name).one_or_none()
        if team is not None:
            return team
        team = Team(name=team_name)
        session.add(team)
        session.flush()
        return team


if __name__ == "__main__":
    print("Use FootballDataSource.load_csv(path) or fetch_csv(url).")
