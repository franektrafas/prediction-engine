"""ESPN's undocumented site API as a supplemental match-stats source.

Not an official/published API -- URLs and payload shapes were found by
inspecting espn.com's own network traffic (`site.api.espn.com/apis/site/v2/
sports/soccer/...`), so they could change without notice. Free, no auth
key, no published rate limit.

Live-tested 2026-08-01: per-match boxscore stats (cards, corners, shots,
possession, passes, tackles, ...) are only populated for a subset of
competitions. UEFA-region qualifiers/finals and flagship tournament finals
have them; other confederations' World Cup qualifiers and FIFA friendlies
come back with an empty `boxscore.teams[].statistics` list. See
`INTERNATIONAL_COMPETITIONS` for exactly which competition/date-range
combinations were confirmed to have real stats.
"""

from __future__ import annotations

from datetime import datetime
import argparse
import logging
from typing import Any

import pandas as pd

from football_ai.data.io import write_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.scrapers.base import BaseDataSource


LOGGER = logging.getLogger(__name__)

# Each tuple is (league_slug, start_date, end_date, label); dates are
# inclusive YYYYMMDD, passed straight through to ESPN's own `dates=` range
# query. Unlike StatsBomb's `competition_id`/`season_id` pair, an ESPN league
# slug is tied to one specific competition instance (e.g. "uefa.euroq" is
# Euro 2024 qualifying specifically, not "Euro qualifying" in general) --
# there's no discovery endpoint for past instances, so each campaign needs
# its own curated date window, verified live to actually carry boxscore
# stats before being added here.
INTERNATIONAL_COMPETITIONS: tuple[tuple[str, str, str, str], ...] = (
    ("fifa.world", "20220101", "20221231", "FIFA World Cup 2022"),
    ("fifa.world", "20180101", "20181231", "FIFA World Cup 2018"),
    ("uefa.euroq", "20230101", "20240401", "UEFA Euro 2024 Qualifying"),
    ("fifa.worldq.uefa", "20250101", "20260401", "UEFA World Cup 2026 Qualifying"),
    ("uefa.nations", "20240101", "20250701", "UEFA Nations League 2024/25"),
    ("caf.nations", "20240101", "20240301", "Africa Cup of Nations 2023"),
    ("conmebol.america", "20240601", "20240801", "Copa America 2024"),
)

# ESPN boxscore stat `name` -> this project's wide-matches column name.
# Deliberately broader than StatsBomb's `_TEAM_STAT_COLUMNS` in
# `statsbomb.py` -- ESPN doesn't expose xg or a pressing metric, but does
# expose possession/fouls/passing that StatsBomb's event-stream aggregation
# doesn't currently compute. `merge_advanced_stats` only fills columns the
# base frame doesn't already have, so the extra columns are a free bonus,
# not a conflict.
_STAT_NAME_MAP: dict[str, str] = {
    "totalShots": "shots",
    "shotsOnTarget": "shots_on_target",
    "wonCorners": "corners",
    "yellowCards": "yellow_cards",
    "redCards": "red_cards",
    "possessionPct": "possession_pct",
    "foulsCommitted": "fouls_committed",
    "offsides": "offsides",
    "totalPasses": "total_passes",
    "passPct": "pass_pct",
    "saves": "saves",
}
_STAT_COLUMNS: tuple[str, ...] = tuple(_STAT_NAME_MAP.values())


class EspnSource(BaseDataSource):
    """Importer for ESPN's undocumented site API."""

    BASE_URL = "https://site.api.espn.com/apis/site/v2/sports/soccer"

    def __init__(self) -> None:
        """Initialize the data source."""
        super().__init__(source_name="espn")

    def fetch_scoreboard(self, league_slug: str, start_date: str, end_date: str) -> list[dict[str, Any]]:
        """Fetch every event ESPN lists for a league slug within an inclusive date range.

        `limit=1000` is required -- CONFIRMED REAL (live-tested 2026-08-01):
        the scoreboard endpoint silently caps a date-range query at 100
        events without it, which is below a full qualifying campaign's real
        match count (a full 2023 calendar year of Euro 2024 qualifying alone
        is 230 matches).
        """
        payload = self.fetch_json(
            f"{self.BASE_URL}/{league_slug}/scoreboard",
            params={"dates": f"{start_date}-{end_date}", "limit": 1000},
        )
        self.persist_raw_json(payload, f"scoreboard_{league_slug}_{start_date}_{end_date}.json")
        return payload.get("events", [])

    def fetch_summary(self, league_slug: str, event_id: str) -> dict[str, Any]:
        """Fetch the full match summary (incl. boxscore stats) for one event."""
        payload = self.fetch_json(f"{self.BASE_URL}/{league_slug}/summary", params={"event": event_id})
        self.persist_raw_json(payload, f"summary_{league_slug}_{event_id}.json")
        return payload

    def normalize_matches(self, events: list[dict[str, Any]]) -> pd.DataFrame:
        """Normalize ESPN scoreboard event payloads to the shared wide-matches identity schema.

        Only finished matches are kept -- an unplayed fixture has no real
        score and would otherwise merge a null score into the base frame.
        """
        rows: list[dict[str, Any]] = []
        for event in events:
            competition = (event.get("competitions") or [{}])[0]
            status_name = ((competition.get("status") or {}).get("type") or {}).get("name")
            if status_name != "STATUS_FULL_TIME":
                continue
            competitors = competition.get("competitors", [])
            home = next((c for c in competitors if c.get("homeAway") == "home"), {})
            away = next((c for c in competitors if c.get("homeAway") == "away"), {})
            if not home or not away:
                continue
            venue = competition.get("venue") or {}
            rows.append(
                {
                    "external_id": str(event["id"]),
                    "source": "espn",
                    "match_date": datetime.fromisoformat(event["date"].replace("Z", "+00:00")),
                    "stadium_name": venue.get("fullName"),
                    "home_team_name": canonicalize_team_name(home.get("team", {}).get("displayName", "")),
                    "away_team_name": canonicalize_team_name(away.get("team", {}).get("displayName", "")),
                    "home_score": int(home["score"]) if home.get("score") not in (None, "") else None,
                    "away_score": int(away["score"]) if away.get("score") not in (None, "") else None,
                }
            )
        frame = pd.DataFrame(rows)
        if not frame.empty:
            frame = frame.sort_values("match_date")
        self.persist_frame(frame, "normalized_matches.parquet")
        return frame

    def extract_team_statistics(self, summary: dict[str, Any]) -> pd.DataFrame:
        """Extract per-team boxscore statistics from a match summary payload.

        Returns an empty frame when ESPN has no boxscore for this match --
        CONFIRMED REAL (live-tested 2026-08-01): non-UEFA World Cup
        qualifiers and FIFA friendlies come back with an empty `statistics`
        list per team, not an error, so this is a normal/expected outcome
        for competitions outside `INTERNATIONAL_COMPETITIONS`, not a bug.
        """
        teams = summary.get("boxscore", {}).get("teams", [])
        records: list[dict[str, Any]] = []
        for team in teams:
            stats = {stat["name"]: stat.get("displayValue") for stat in team.get("statistics", [])}
            if not stats:
                continue
            record: dict[str, Any] = {
                "team_name": canonicalize_team_name(team.get("team", {}).get("displayName", ""))
            }
            for espn_name, column in _STAT_NAME_MAP.items():
                value = stats.get(espn_name)
                record[column] = float(value) if value not in (None, "") else None
            records.append(record)
        return pd.DataFrame(records)

    def build_wide_matches_frame(self, league_slug: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Build a wide matches frame with real per-match stats for one league/date-range.

        Reuses `fetch_scoreboard`/`normalize_matches` for match identity and
        `fetch_summary`/`extract_team_statistics` for the stats -- the same
        two-pass shape as StatsBomb's `build_wide_matches_frame` in
        `statsbomb.py`, so both sources plug into `merge_advanced_stats` the
        same way.
        """
        events = self.fetch_scoreboard(league_slug, start_date, end_date)
        normalized = self.normalize_matches(events)
        if normalized.empty:
            return normalized
        stat_rows: list[dict[str, Any]] = []
        for row in normalized.to_dict(orient="records"):
            stat_row: dict[str, Any] = {"external_id": row["external_id"]}
            summary = self.fetch_summary(league_slug, row["external_id"])
            header_competition = (summary.get("header", {}).get("competitions") or [{}])[0]
            stat_row["neutral_venue"] = bool(header_competition.get("neutralSite", False))
            team_statistics = self.extract_team_statistics(summary)
            for _, team_row in team_statistics.iterrows():
                is_home = team_row["team_name"] == row["home_team_name"]
                prefix = "home_" if is_home else "away_"
                for column in _STAT_COLUMNS:
                    stat_row[f"{prefix}{column}"] = team_row.get(column)
            stat_rows.append(stat_row)
        stats_frame = pd.DataFrame(stat_rows)
        if stats_frame.empty:
            return normalized
        return normalized.merge(stats_frame, on="external_id", how="left")

    def build_all_international_matches(self) -> pd.DataFrame:
        """Build a combined wide matches frame across every curated competition/date-range.

        A competition that fails to fetch (network hiccup, ESPN layout
        change -- this is an undocumented API) is logged and skipped rather
        than aborting the whole batch, same reasoning as StatsBomb's
        equivalent in `statsbomb.py`.
        """
        frames: list[pd.DataFrame] = []
        for league_slug, start_date, end_date, label in INTERNATIONAL_COMPETITIONS:
            LOGGER.info("Fetching %s (%s, %s-%s)", label, league_slug, start_date, end_date)
            try:
                frames.append(self.build_wide_matches_frame(league_slug, start_date, end_date))
            except Exception:
                LOGGER.warning("Failed to build matches for %s", label, exc_info=True)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True).sort_values("match_date").reset_index(drop=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch ESPN international match stats.")
    parser.add_argument(
        "--mode",
        choices=("competitions", "enrich"),
        default="competitions",
        help="List the curated competition/date-range entries, or build an enriched matches file.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Where to write the enriched matches file (parquet/csv/json). Required for --mode enrich.",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source = EspnSource()
    if arguments.mode == "competitions":
        for entry in INTERNATIONAL_COMPETITIONS:
            print(entry)
    else:
        if not arguments.output_path:
            raise ValueError("--output-path is required when --mode enrich is used.")
        enriched_matches = source.build_all_international_matches()
        write_table(enriched_matches, arguments.output_path)
        print(f"Wrote {len(enriched_matches)} enriched international matches to {arguments.output_path}.")
