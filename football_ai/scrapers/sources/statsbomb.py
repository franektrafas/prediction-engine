"""StatsBomb Open Data importer."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data.normalization import canonicalize_team_name
from football_ai.database.models import Match, Team, TeamStatistic
from football_ai.scrapers.base import BaseDataSource


class StatsBombOpenDataSource(BaseDataSource):
    """Importer for StatsBomb Open Data."""

    BASE_URL = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"

    def __init__(self) -> None:
        """Initialize the data source."""
        super().__init__(source_name="statsbomb")

    def fetch_competitions(self) -> list[dict[str, Any]]:
        """Fetch the list of available competitions."""
        competitions = self.fetch_json(f"{self.BASE_URL}/competitions.json")
        self.persist_raw_json(competitions, "competitions.json")
        return competitions

    def fetch_matches(self, competition_id: int, season_id: int) -> list[dict[str, Any]]:
        """Fetch match metadata for a competition and season."""
        matches = self.fetch_json(
            f"{self.BASE_URL}/matches/{competition_id}/{season_id}.json"
        )
        self.persist_raw_json(matches, f"matches_{competition_id}_{season_id}.json")
        return matches

    def fetch_lineups(self, match_id: int) -> list[dict[str, Any]]:
        """Fetch lineups for a match."""
        lineups = self.fetch_json(f"{self.BASE_URL}/lineups/{match_id}.json")
        self.persist_raw_json(lineups, f"lineups_{match_id}.json")
        return lineups

    def fetch_events(self, match_id: int) -> list[dict[str, Any]]:
        """Fetch event stream for a match."""
        events = self.fetch_json(f"{self.BASE_URL}/events/{match_id}.json")
        self.persist_raw_json(events, f"events_{match_id}.json")
        return events

    def normalize_matches(
        self,
        matches: list[dict[str, Any]],
    ) -> pd.DataFrame:
        """Normalize StatsBomb match payloads."""
        rows: list[dict[str, Any]] = []
        for item in matches:
            rows.append(
                {
                    "external_id": str(item["match_id"]),
                    "source": "statsbomb",
                    "competition_id": str(item["competition"]["competition_id"]),
                    "competition_name": item["competition"]["competition_name"],
                    "season_name": item["season"]["season_name"],
                    "tournament": item["competition"]["competition_name"],
                    "match_date": datetime.fromisoformat(item["match_date"]).replace(
                        tzinfo=UTC
                    ),
                    "stadium_name": (item.get("stadium") or {}).get("name"),
                    "referee": (item.get("referee") or {}).get("name"),
                    "home_team_name": canonicalize_team_name(item["home_team"]["home_team_name"]),
                    "away_team_name": canonicalize_team_name(item["away_team"]["away_team_name"]),
                    "home_score": item.get("home_score"),
                    "away_score": item.get("away_score"),
                    "neutral_venue": bool(item.get("metadata", {}).get("neutral_ground", False)),
                }
            )
        frame = pd.DataFrame(rows).sort_values("match_date")
        self.persist_frame(frame, "normalized_matches.parquet")
        return frame

    def aggregate_team_statistics(self, events: list[dict[str, Any]]) -> pd.DataFrame:
        """Aggregate basic team statistics from the event stream."""
        records: list[dict[str, Any]] = []
        for event in events:
            team = (event.get("team") or {}).get("name")
            if not team:
                continue
            records.append(
                {
                    "team_name": canonicalize_team_name(team),
                    "event_type": (event.get("type") or {}).get("name"),
                    "shot_xg": ((event.get("shot") or {}).get("statsbomb_xg")),
                    "shot_outcome": ((event.get("shot") or {}).get("outcome") or {}).get("name"),
                    "pass_type": ((event.get("pass") or {}).get("type") or {}).get("name"),
                    "card_name": ((event.get("foul_committed") or {}).get("card") or {}).get(
                        "name"
                    ),
                }
            )
        frame = pd.DataFrame(records)
        if frame.empty:
            return frame
        grouped = frame.groupby("team_name")
        statistics = pd.DataFrame(
            {
                "team_name": list(grouped.groups.keys()),
                "xg": grouped["shot_xg"].sum(min_count=1).values,
                "shots": grouped.apply(lambda grp: (grp["event_type"] == "Shot").sum()).values,
                "shots_on_target": grouped.apply(
                    lambda grp: grp["shot_outcome"].isin(
                        ["Goal", "Saved", "Saved To Post"]
                    ).sum()
                ).values,
                "corners": grouped.apply(lambda grp: (grp["pass_type"] == "Corner").sum()).values,
                "yellow_cards": grouped.apply(
                    lambda grp: (grp["card_name"] == "Yellow Card").sum()
                ).values,
                "red_cards": grouped.apply(
                    lambda grp: grp["card_name"].isin(["Red Card", "Second Yellow"]).sum()
                ).values,
                "pressing_intensity": grouped.apply(
                    lambda grp: (grp["event_type"] == "Pressure").sum()
                ).values,
            }
        )
        return statistics

    def ingest_matches_to_database(
        self,
        session: Session,
        normalized_matches: pd.DataFrame,
        events_by_match_id: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        """Persist matches and derived team statistics."""
        for row in normalized_matches.to_dict(orient="records"):
            home_team = self._get_or_create_team(session, row["home_team_name"])
            away_team = self._get_or_create_team(session, row["away_team_name"])
            match = Match(
                external_id=row["external_id"],
                source=row["source"],
                competition_id=row["competition_id"],
                competition_name=row["competition_name"],
                season_name=row["season_name"],
                tournament=row["tournament"],
                match_date=row["match_date"],
                stadium_name=row["stadium_name"],
                referee=row["referee"],
                home_team_id=home_team.id,
                away_team_id=away_team.id,
                home_score=row["home_score"],
                away_score=row["away_score"],
                neutral_venue=row["neutral_venue"],
            )
            session.add(match)
            session.flush()
            if events_by_match_id and row["external_id"] in events_by_match_id:
                stats_frame = self.aggregate_team_statistics(events_by_match_id[row["external_id"]])
                for _, stats_row in stats_frame.iterrows():
                    team_name = stats_row["team_name"]
                    is_home = team_name == row["home_team_name"]
                    team = home_team if is_home else away_team
                    session.add(
                        TeamStatistic(
                            match_id=match.id,
                            team_id=team.id,
                            is_home=is_home,
                            goals=row["home_score"] if is_home else row["away_score"],
                            xg=float(stats_row.get("xg") or 0.0),
                            shots=int(stats_row.get("shots") or 0),
                            shots_on_target=int(stats_row.get("shots_on_target") or 0),
                            corners=int(stats_row.get("corners") or 0),
                            yellow_cards=int(stats_row.get("yellow_cards") or 0),
                            red_cards=int(stats_row.get("red_cards") or 0),
                            pressing_intensity=float(
                                stats_row.get("pressing_intensity") or 0.0
                            ),
                        )
                    )
        session.commit()

    def _get_or_create_team(self, session: Session, team_name: str) -> Team:
        """Create a team when not present in the database."""
        team = session.query(Team).filter(Team.name == team_name).one_or_none()
        if team is not None:
            return team
        team = Team(name=team_name)
        session.add(team)
        session.flush()
        return team


if __name__ == "__main__":
    source = StatsBombOpenDataSource()
    print(source.fetch_competitions()[:3])
