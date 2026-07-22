"""StatsBomb Open Data importer."""

from __future__ import annotations

from datetime import UTC, datetime
import argparse
import logging
from typing import Any

import pandas as pd
from sqlalchemy.orm import Session

from football_ai.data.io import write_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.database.models import Match, Team, TeamStatistic
from football_ai.scrapers.base import BaseDataSource


LOGGER = logging.getLogger(__name__)

# Verified via StatsBomb's public `competitions.json` -- every international/
# national-team competition (men's + women's) they publish open data for.
# Free, no auth, no rate limit (static JSON on GitHub raw). Deliberately
# excludes their club competitions (La Liga, Champions League, FAWSL, ...).
INTERNATIONAL_COMPETITIONS: tuple[tuple[int, int, str], ...] = (
    (43, 106, "FIFA World Cup 2022"),
    (43, 3, "FIFA World Cup 2018"),
    (43, 55, "FIFA World Cup 1990"),
    (43, 54, "FIFA World Cup 1986"),
    (43, 51, "FIFA World Cup 1974"),
    (43, 272, "FIFA World Cup 1970"),
    (43, 270, "FIFA World Cup 1962"),
    (43, 269, "FIFA World Cup 1958"),
    (55, 282, "UEFA Euro 2024"),
    (55, 43, "UEFA Euro 2020"),
    (223, 282, "Copa America 2024"),
    (1267, 107, "Africa Cup of Nations 2023"),
    (72, 107, "Women's World Cup 2023"),
    (72, 30, "Women's World Cup 2019"),
    (53, 315, "UEFA Women's Euro 2025"),
    (53, 106, "UEFA Women's Euro 2022"),
)

_TEAM_STAT_COLUMNS = (
    "xg",
    "shots",
    "shots_on_target",
    "corners",
    "yellow_cards",
    "red_cards",
    "pressing_intensity",
)


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

    def build_wide_matches_frame(self, competition_id: int, season_id: int) -> pd.DataFrame:
        """Build a wide matches frame with real per-match stats for one competition/season.

        Reuses `fetch_matches`/`normalize_matches` for match identity and
        `fetch_events`/`aggregate_team_statistics` for the stats -- the only
        new logic here is pivoting the per-team stats onto the match row as
        `home_*`/`away_*` columns, matching the wide `matches` schema
        `features.builder.build_features` expects.
        """
        matches = self.fetch_matches(competition_id, season_id)
        normalized = self.normalize_matches(matches)
        stat_rows: list[dict[str, Any]] = []
        for row in normalized.to_dict(orient="records"):
            stat_row: dict[str, Any] = {"external_id": row["external_id"]}
            events = self.fetch_events(int(row["external_id"]))
            team_statistics = self.aggregate_team_statistics(events)
            for _, team_row in team_statistics.iterrows():
                is_home = team_row["team_name"] == row["home_team_name"]
                prefix = "home_" if is_home else "away_"
                for column in _TEAM_STAT_COLUMNS:
                    stat_row[f"{prefix}{column}"] = team_row.get(column)
            stat_rows.append(stat_row)
        stats_frame = pd.DataFrame(stat_rows)
        if stats_frame.empty:
            return normalized
        return normalized.merge(stats_frame, on="external_id", how="left")

    def build_all_international_matches(self) -> pd.DataFrame:
        """Build a combined wide matches frame across every curated international competition.

        A competition that fails to fetch (network hiccup, StatsBomb layout
        change) is logged and skipped rather than aborting the whole batch --
        this is an offline enrichment job, not a user-facing request, so a
        partial result is far more useful than none at all.
        """
        frames: list[pd.DataFrame] = []
        for competition_id, season_id, label in INTERNATIONAL_COMPETITIONS:
            LOGGER.info(
                "Fetching %s (competition_id=%s, season_id=%s)", label, competition_id, season_id
            )
            try:
                frames.append(self.build_wide_matches_frame(competition_id, season_id))
            except Exception:
                LOGGER.warning("Failed to build matches for %s", label, exc_info=True)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("match_date").reset_index(drop=True)

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
    parser = argparse.ArgumentParser(description="Fetch StatsBomb Open Data for international competitions.")
    parser.add_argument(
        "--mode",
        choices=("competitions", "enrich"),
        default="competitions",
        help="List available competitions, or build an enriched international matches file.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the enriched matches file (parquet/csv/json). Required for --mode enrich.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source = StatsBombOpenDataSource()
    if arguments.mode == "competitions":
        print(source.fetch_competitions()[:3])
    else:
        if not arguments.output_path:
            raise ValueError("--output-path is required when --mode enrich is used.")
        enriched_matches = source.build_all_international_matches()
        write_table(enriched_matches, arguments.output_path)
        print(f"Wrote {len(enriched_matches)} enriched international matches to {arguments.output_path}.")
