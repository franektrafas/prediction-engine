"""Live Football API connector for fixtures, lineups, and injuries."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data.normalization import canonicalize_team_name
from football_ai.database.models import Injury, Match, Player, Team
from football_ai.predictions.player_reference_store import (
    PlayerReferenceUpdateResult,
    update_player_reference_store,
)
from football_ai.scrapers.base import BaseDataSource


LOGGER = logging.getLogger(__name__)


class FootballApiSource(BaseDataSource):
    """Connector for a key-based live football data provider."""

    def __init__(self) -> None:
        """Initialize the API-backed source."""
        super().__init__(source_name="football_api")

    def _build_headers(self) -> dict[str, str]:
        """Build authenticated request headers."""
        headers: dict[str, str] = {}
        if self.settings.football_api_key:
            headers["x-apisports-key"] = self.settings.football_api_key
        if self.settings.football_api_host:
            headers["x-rapidapi-host"] = self.settings.football_api_host
        return headers

    def _request(self, endpoint: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """Perform an authenticated API request."""
        if not self.settings.football_api_key:
            raise ValueError(
                "FOOTBALL_API_KEY is not configured. Add it to your .env before calling FootballApiSource."
            )
        url = f"{self.settings.football_api_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        response = self.fetch_json_with_headers(
            url=url,
            headers=self._build_headers(),
            params=params,
        )
        if not isinstance(response, dict):
            raise ValueError("Unexpected Football API response format.")
        return response

    def fetch_fixtures(
        self,
        fixture_date: str,
        league_id: int | None = None,
        season: int | None = None,
        team_id: int | None = None,
        last: int | None = None,
        next_count: int | None = None,
    ) -> pd.DataFrame:
        """Fetch fixtures and normalize them to the shared project schema."""
        params: dict[str, Any] = {"date": fixture_date, "timezone": self.settings.football_api_timezone}
        if league_id is not None:
            params["league"] = league_id
        if season is not None:
            params["season"] = season
        if team_id is not None:
            params["team"] = team_id
        if last is not None:
            params["last"] = last
        if next_count is not None:
            params["next"] = next_count
        payload = self._request("fixtures", params=params)
        self.persist_raw_json(payload, f"fixtures_{fixture_date}.json")
        self._raise_for_api_errors(payload)
        return self.normalize_fixtures(payload)

    def fetch_fixture_lineups(self, fixture_id: int) -> pd.DataFrame:
        """Fetch confirmed or projected lineups for a fixture."""
        payload = self._request("fixtures/lineups", params={"fixture": fixture_id})
        self.persist_raw_json(payload, f"lineups_{fixture_id}.json")
        self._raise_for_api_errors(payload)
        return self.normalize_lineups(payload, fixture_id=fixture_id)

    def fetch_injuries(self, league_id: int, season: int) -> pd.DataFrame:
        """Fetch player injuries for a league and season."""
        payload = self._request("injuries", params={"league": league_id, "season": season})
        self.persist_raw_json(payload, f"injuries_{league_id}_{season}.json")
        self._raise_for_api_errors(payload)
        return self.normalize_injuries(payload)

    def fetch_team_squad(self, team_id: int) -> pd.DataFrame:
        """Fetch the current squad metadata for a team."""
        payload = self._request("players/squads", params={"team": team_id})
        self.persist_raw_json(payload, f"squad_{team_id}.json")
        self._raise_for_api_errors(payload)
        return self.normalize_team_squad(payload)

    def refresh_player_reference(
        self,
        team_ids: list[int],
        player_reference_path: str | Path | None = None,
    ) -> PlayerReferenceUpdateResult:
        """Fetch squads for the given teams and merge them into the player reference store.

        Resilient per team: a single team's fetch failing (bad id, rate
        limit, network) is logged and skipped rather than aborting the whole
        refresh, so a scheduled batch still updates whatever teams it could
        reach. If every team fails, the store is left untouched and the
        result reports zero additions/updates rather than raising -- this is
        a maintenance job, not a user-facing request, so a partial or total
        failure should show up in logs, not crash the caller.
        """
        resolved_path = Path(player_reference_path) if player_reference_path else self.settings.player_reference_path
        squads: list[pd.DataFrame] = []
        for team_id in team_ids:
            try:
                squad = self.fetch_team_squad(team_id)
            except Exception:
                LOGGER.warning("Failed to fetch squad for team %s", team_id, exc_info=True)
                continue
            if not squad.empty:
                squads.append(squad)
        combined = pd.concat(squads, ignore_index=True) if squads else pd.DataFrame()
        return update_player_reference_store(
            resolved_path,
            observed_players=combined,
            source_name="football_api",
        )

    def _raise_for_api_errors(self, payload: dict[str, Any]) -> None:
        """Raise a readable exception when the provider returns API-level errors."""
        errors = payload.get("errors")
        if not errors:
            return
        if isinstance(errors, dict):
            active_errors = {
                str(key): str(value)
                for key, value in errors.items()
                if value not in ("", None, [], {})
            }
            if active_errors:
                details = ", ".join(f"{key}={value}" for key, value in active_errors.items())
                raise ValueError(f"Football API returned an error: {details}")
            return
        raise ValueError(f"Football API returned an error: {errors}")

    def normalize_fixtures(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize fixture payload into match-level records."""
        rows: list[dict[str, Any]] = []
        for item in payload.get("response", []):
            fixture = item.get("fixture", {})
            league = item.get("league", {})
            teams = item.get("teams", {})
            goals = item.get("goals", {})
            venue = fixture.get("venue", {}) or {}
            home_team = teams.get("home", {}) or {}
            away_team = teams.get("away", {}) or {}
            rows.append(
                {
                    "external_id": str(fixture.get("id", "")),
                    "source": "football_api",
                    "competition_id": str(league.get("id", "")) or None,
                    "competition_name": league.get("name"),
                    "season_name": str(league.get("season")) if league.get("season") is not None else None,
                    "tournament": league.get("round") or league.get("name"),
                    "status": (fixture.get("status", {}) or {}).get("short"),
                    "match_date": pd.to_datetime(fixture.get("date"), utc=True),
                    "stadium_name": venue.get("name"),
                    "city": venue.get("city"),
                    "country": league.get("country"),
                    "neutral_venue": bool(fixture.get("neutral") or False),
                    "home_team_id_external": home_team.get("id"),
                    "home_team_name": canonicalize_team_name(home_team.get("name")),
                    "away_team_id_external": away_team.get("id"),
                    "away_team_name": canonicalize_team_name(away_team.get("name")),
                    "home_score": goals.get("home"),
                    "away_score": goals.get("away"),
                    "league_logo": league.get("logo"),
                    "home_team_logo": home_team.get("logo"),
                    "away_team_logo": away_team.get("logo"),
                }
            )
        return pd.DataFrame(rows)

    def normalize_lineups(self, payload: dict[str, Any], fixture_id: int) -> pd.DataFrame:
        """Normalize lineups into player-level rows."""
        rows: list[dict[str, Any]] = []
        for team_payload in payload.get("response", []):
            team = team_payload.get("team", {}) or {}
            start_xi = team_payload.get("startXI", []) or []
            substitutes = team_payload.get("substitutes", []) or []
            for started, players in ((True, start_xi), (False, substitutes)):
                for player_wrapper in players:
                    player = player_wrapper.get("player", {}) or {}
                    rows.append(
                        {
                            "fixture_id": fixture_id,
                            "team_external_id": team.get("id"),
                            "team_name": canonicalize_team_name(team.get("name")),
                            "player_external_id": player.get("id"),
                            "player_name": player.get("name"),
                            "shirt_number": player.get("number"),
                            "position": player.get("pos"),
                            "grid": player.get("grid"),
                            "started": started,
                        }
                    )
        return pd.DataFrame(rows)

    def normalize_injuries(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize injury payload into player availability records."""
        rows: list[dict[str, Any]] = []
        for item in payload.get("response", []):
            player = item.get("player", {}) or {}
            team = item.get("team", {}) or {}
            fixture = item.get("fixture", {}) or {}
            rows.append(
                {
                    "player_external_id": player.get("id"),
                    "player_name": player.get("name"),
                    "team_external_id": team.get("id"),
                    "team_name": canonicalize_team_name(team.get("name")),
                    "fixture_id": fixture.get("id"),
                    "injury_date": pd.to_datetime(fixture.get("date"), utc=True).date()
                    if fixture.get("date")
                    else None,
                    "description": item.get("reason"),
                    "severity": item.get("type"),
                    "source": "football_api",
                }
            )
        return pd.DataFrame(rows)

    def normalize_team_squad(self, payload: dict[str, Any]) -> pd.DataFrame:
        """Normalize squad payload into player reference rows."""
        rows: list[dict[str, Any]] = []
        for squad_payload in payload.get("response", []):
            team = squad_payload.get("team", {}) or {}
            players = squad_payload.get("players", []) or []
            for player in players:
                age_value = player.get("age")
                rows.append(
                    {
                        "team_external_id": team.get("id"),
                        "team_name": canonicalize_team_name(team.get("name")),
                        "player_external_id": player.get("id"),
                        "player_name": player.get("name"),
                        "age": float(age_value) if age_value not in (None, "") else None,
                        "position": player.get("position"),
                        "number": player.get("number"),
                    }
                )
        return pd.DataFrame(rows)

    def ingest_fixtures_to_database(self, session: Session, fixtures: pd.DataFrame) -> None:
        """Persist normalized fixtures to the project database."""
        for row in fixtures.to_dict(orient="records"):
            home_team = self._get_or_create_team(
                session=session,
                team_name=row["home_team_name"],
                external_id=row.get("home_team_id_external"),
            )
            away_team = self._get_or_create_team(
                session=session,
                team_name=row["away_team_name"],
                external_id=row.get("away_team_id_external"),
            )
            existing_match = (
                session.query(Match)
                .filter(Match.source == "football_api", Match.external_id == row["external_id"])
                .one_or_none()
            )
            if existing_match is not None:
                continue
            session.add(
                Match(
                    external_id=row["external_id"],
                    source="football_api",
                    competition_id=row.get("competition_id"),
                    competition_name=row.get("competition_name"),
                    season_name=row.get("season_name"),
                    tournament=row.get("tournament"),
                    status=row.get("status"),
                    match_date=row["match_date"],
                    stadium_name=row.get("stadium_name"),
                    city=row.get("city"),
                    country=row.get("country"),
                    neutral_venue=bool(row.get("neutral_venue", False)),
                    home_team_id=home_team.id,
                    away_team_id=away_team.id,
                    home_score=row.get("home_score"),
                    away_score=row.get("away_score"),
                )
            )
        session.commit()

    def ingest_injuries_to_database(self, session: Session, injuries: pd.DataFrame) -> None:
        """Persist normalized injuries to the project database."""
        for row in injuries.to_dict(orient="records"):
            team = self._get_or_create_team(
                session=session,
                team_name=row["team_name"],
                external_id=row.get("team_external_id"),
            )
            player = self._get_or_create_player(
                session=session,
                team=team,
                player_name=row.get("player_name"),
                external_id=row.get("player_external_id"),
            )
            session.add(
                Injury(
                    team_id=team.id,
                    player_id=player.id if player is not None else None,
                    injury_date=row.get("injury_date"),
                    severity=row.get("severity"),
                    description=row.get("description"),
                    source="football_api",
                )
            )
        session.commit()

    def _get_or_create_team(
        self,
        session: Session,
        team_name: str,
        external_id: int | str | None,
    ) -> Team:
        """Fetch or create a team row."""
        query = session.query(Team).filter(Team.name == team_name)
        team = query.one_or_none()
        if team is not None:
            if external_id is not None and not team.fifa_code:
                team.fifa_code = str(external_id)
            return team
        team = Team(name=team_name, fifa_code=str(external_id) if external_id is not None else None)
        session.add(team)
        session.flush()
        return team

    def _get_or_create_player(
        self,
        session: Session,
        team: Team,
        player_name: str | None,
        external_id: int | str | None,
    ) -> Player | None:
        """Fetch or create a player row."""
        if not player_name:
            return None
        player = (
            session.query(Player)
            .filter(Player.full_name == player_name, Player.team_id == team.id)
            .one_or_none()
        )
        if player is not None:
            if external_id is not None and not player.external_id:
                player.external_id = str(external_id)
            return player
        player = Player(
            team_id=team.id,
            external_id=str(external_id) if external_id is not None else None,
            full_name=player_name,
        )
        session.add(player)
        session.flush()
        return player


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch live football data from the configured Football API.")
    parser.add_argument(
        "--mode",
        choices=("fixtures", "lineups", "squad"),
        default="fixtures",
        help="Whether to fetch fixtures for a date, lineups for a fixture, or refresh player-reference squads.",
    )
    parser.add_argument("--date", default=None, help="Fixture date in YYYY-MM-DD format.")
    parser.add_argument("--fixture-id", type=int, default=None, help="Fixture identifier for lineup fetches.")
    parser.add_argument("--league-id", type=int, default=None, help="Optional league identifier.")
    parser.add_argument("--season", type=int, default=None, help="Optional season year.")
    parser.add_argument("--team-id", type=int, default=None, help="Optional team identifier.")
    parser.add_argument(
        "--team-ids",
        default=None,
        help="Comma-separated team identifiers to refresh player-reference squads for (squad mode).",
    )
    parser.add_argument("--last", type=int, default=None, help="Fetch the last N fixtures.")
    parser.add_argument("--next-count", type=int, default=None, help="Fetch the next N fixtures.")
    arguments = parser.parse_args()

    source = FootballApiSource()
    output_directory = Path("football_ai/data/raw/football_api")
    if arguments.mode == "fixtures":
        if not arguments.date:
            raise ValueError("--date is required when --mode fixtures is used.")
        fixtures = source.fetch_fixtures(
            fixture_date=arguments.date,
            league_id=arguments.league_id,
            season=arguments.season,
            team_id=arguments.team_id,
            last=arguments.last,
            next_count=arguments.next_count,
        )
        print(f"Fetched {len(fixtures)} normalized fixtures.")
        print(fixtures.head(10).to_string(index=False))
        print(f"Raw payloads are stored in {output_directory}.")
    elif arguments.mode == "lineups":
        if arguments.fixture_id is None:
            raise ValueError("--fixture-id is required when --mode lineups is used.")
        lineups = source.fetch_fixture_lineups(arguments.fixture_id)
        print(f"Fetched {len(lineups)} lineup rows.")
        if lineups.empty:
            print("No lineup rows returned for this fixture.")
        else:
            print(lineups.head(30).to_string(index=False))
        print(f"Raw payloads are stored in {output_directory}.")
    else:
        team_ids_argument = arguments.team_ids or (
            str(arguments.team_id) if arguments.team_id is not None else None
        )
        if not team_ids_argument:
            raise ValueError("--team-ids (or --team-id) is required when --mode squad is used.")
        team_ids = [int(value) for value in team_ids_argument.split(",") if value.strip()]
        result = source.refresh_player_reference(team_ids)
        print(
            f"Refreshed player reference from {len(team_ids)} team squad(s): "
            f"+{result.added_players} added, {result.updated_players} updated, "
            f"{len(result.frame)} total players in {source.settings.player_reference_path}."
        )
