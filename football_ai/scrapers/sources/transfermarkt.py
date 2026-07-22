"""Transfermarkt scraper for national team squads, market values, and clubs."""

from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from football_ai.data.io import write_table
from football_ai.data.normalization import canonicalize_team_name
from football_ai.predictions.player_reference_store import (
    PlayerReferenceUpdateResult,
    update_player_reference_store,
)
from football_ai.scrapers.base import BaseDataSource


LOGGER = logging.getLogger(__name__)

# transfermarkt.com blocks the default `requests` User-Agent; a realistic
# browser header is all that's needed -- verified against real pages during
# planning, no Cloudflare challenge, no proxy/TLS-fingerprint spoofing.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_AGE_GROUP_SLUG = re.compile(r"-(u1[5-9]|u2[0-3]|olympia|frauen|women)$", re.IGNORECASE)
_MARKET_VALUE = re.compile(r"€\s*([\d.]+)\s*(m|k|th\.?)?", re.IGNORECASE)
_AGE_SUFFIX = re.compile(r"\((\d+)\)")

# Verified individually via Transfermarkt's own quick-search (exact link-text
# match against the canonical English country name), not guessed -- see the
# roadmap plan for the verification method. (team_id, url_slug, canonical_name)
NATIONAL_TEAM_IDS: tuple[tuple[int, str, str], ...] = (
    (3442, "polen", "Poland"),
    (3377, "frankreich", "France"),
    (3262, "deutschland", "Germany"),
    (3375, "spanien", "Spain"),
    (3299, "england", "England"),
    (3376, "italien", "Italy"),
    (3300, "portugal", "Portugal"),
    (3379, "niederlande", "Netherlands"),
    (3382, "belgien", "Belgium"),
    (3556, "kroatien", "Croatia"),
    (3436, "danemark", "Denmark"),
    (3557, "schweden", "Sweden"),
    (3384, "schweiz", "Switzerland"),
    (3383, "osterreich", "Austria"),
    (3438, "serbien", "Serbia"),
    (3699, "ukraine", "Ukraine"),
    (3381, "turkei", "Turkiye"),
    (3864, "wales", "Wales"),
    (3380, "schottland", "Scotland"),
    (3437, "argentinien", "Argentina"),
    (3439, "brasilien", "Brazil"),
    (3449, "uruguay", "Uruguay"),
    (3816, "kolumbien", "Colombia"),
    (3700, "chile", "Chile"),
    (5750, "ecuador", "Ecuador"),
    (3584, "peru", "Peru"),
    (3505, "vereinigte-staaten", "United States"),
    (6303, "mexiko", "Mexico"),
    (3510, "kanada", "Canada"),
    (3435, "japan", "Japan"),
    (3589, "sudkorea", "South Korea"),
    (3807, "saudi-arabien", "Saudi Arabia"),
    (3582, "iran", "Iran"),
    (3433, "australien", "Australia"),
    (3575, "marokko", "Morocco"),
    (3499, "senegal", "Senegal"),
    (3444, "nigeria", "Nigeria"),
    (3672, "agypten", "Egypt"),
    (3591, "elfenbeinkuste", "Ivory Coast"),
    (3441, "ghana", "Ghana"),
    (3434, "kamerun", "Cameroon"),
)


def _parse_market_value(text: str | None) -> float | None:
    """Parse a Transfermarkt market-value string into euros.

    `"€12.00m"` -> `12_000_000.0`, `"€500k"`/`"€500Th."` -> `500_000.0`,
    `"-"`/empty/unmatched -> `None`.
    """
    if not text:
        return None
    match = _MARKET_VALUE.search(text)
    if not match:
        return None
    amount = float(match.group(1))
    suffix = (match.group(2) or "").lower()
    if suffix == "m":
        return amount * 1_000_000.0
    if suffix in {"k", "th", "th."}:
        return amount * 1_000.0
    return amount


def _parse_age(text: str | None) -> float | None:
    """Extract the `(NN)` age suffix Transfermarkt appends to a date of birth."""
    if not text:
        return None
    match = _AGE_SUFFIX.search(text)
    return float(match.group(1)) if match else None


class TransfermarktSource(BaseDataSource):
    """Scraper for national team squads, market values, and clubs."""

    BASE_URL = "https://www.transfermarkt.com"

    def __init__(self) -> None:
        """Initialize the Transfermarkt source."""
        super().__init__(source_name="transfermarkt")

    def search_teams(self, query: str) -> list[dict[str, Any]]:
        """Search Transfermarkt for team pages matching a name query.

        A discovery/verification helper for building `NATIONAL_TEAM_IDS`
        entries -- not something called blindly at prediction time. Results
        are noisy (clubs with the query as a substring, youth/reserve teams
        of the same country); callers should filter, e.g. by exact link text
        matching the expected country name, the way `NATIONAL_TEAM_IDS` was
        built.
        """
        html = self.fetch_text_with_headers(
            f"{self.BASE_URL}/schnellsuche/ergebnis/schnellsuche",
            headers=_REQUEST_HEADERS,
            params={"query": query},
        )
        soup = BeautifulSoup(html, "lxml")
        results: list[dict[str, Any]] = []
        for link in soup.select("a[href*='/startseite/verein/']"):
            href = link.get("href", "")
            match = re.match(r"^/([a-z0-9\-]+)/startseite/verein/(\d+)$", href)
            if not match:
                continue
            results.append(
                {
                    "slug": match.group(1),
                    "team_id": int(match.group(2)),
                    "text": link.get_text(strip=True),
                    "is_age_group_variant": bool(_AGE_GROUP_SLUG.search(match.group(1))),
                }
            )
        return results

    def fetch_national_team_squad(
        self,
        team_id: int,
        team_slug: str,
        team_name: str,
    ) -> pd.DataFrame:
        """Fetch and parse a national team's current squad with market values.

        `team_name` is caller-supplied (from `NATIONAL_TEAM_IDS`) rather than
        scraped from the page, to avoid a second fragile HTML-title
        extraction on top of the squad table itself.
        """
        html = self.fetch_text_with_headers(
            f"{self.BASE_URL}/{team_slug}/startseite/verein/{team_id}",
            headers=_REQUEST_HEADERS,
        )
        return self._parse_squad_table(html, team_id=team_id, team_name=team_name)

    def _parse_squad_table(self, html: str, team_id: int, team_name: str) -> pd.DataFrame:
        """Parse the real squad table structure into player-reference-shaped rows.

        Plain `pd.read_html` (used by `BaseDataSource.fetch_table` elsewhere
        in this project) triples every row here -- Transfermarkt nests photo/
        name/position in one cell, which `read_html` flattens badly -- and
        loses the club column entirely, since the club name only exists in
        an `<img>`/`<a>` `title` attribute, not as visible text. This walks
        the real `table.items > tbody > tr` structure directly instead.
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one("table.items")
        if table is None:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        canonical_team_name = canonicalize_team_name(team_name)
        for row in table.select("tbody > tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) < 5:
                continue
            name_link = cells[1].select_one("td.hauptlink a")
            if name_link is None:
                continue
            player_href = name_link.get("href", "")
            player_id_match = re.search(r"/spieler/(\d+)", player_href)
            club_link = cells[3].select_one("a[title]")
            rows.append(
                {
                    "team_name": canonical_team_name,
                    "team_external_id": team_id,
                    "player_name": name_link.get_text(strip=True),
                    "player_external_id": (
                        int(player_id_match.group(1)) if player_id_match else None
                    ),
                    "position": cells[0].get("title"),
                    "age": _parse_age(cells[2].get_text(strip=True)),
                    "club_name": club_link.get("title") if club_link is not None else None,
                    "market_value_eur": _parse_market_value(cells[4].get_text(strip=True)),
                }
            )
        return pd.DataFrame(rows)

    def fetch_national_team_injuries(
        self,
        team_id: int,
        team_slug: str,
        team_name: str,
    ) -> pd.DataFrame:
        """Fetch and parse a national team's current injuries/suspensions.

        Unlike `FootballApiSource.fetch_injuries()`, this needs no
        `league_id`/`season` and has no free-plan season cutoff -- verified
        against the real, current page (a July-2026-dated injury showed up
        for Poland during planning) rather than a stale 2022-2024 snapshot.
        """
        html = self.fetch_text_with_headers(
            f"{self.BASE_URL}/{team_slug}/sperrenundverletzungen/verein/{team_id}",
            headers=_REQUEST_HEADERS,
        )
        return self._parse_injuries_table(html, team_id=team_id, team_name=team_name)

    def _parse_injuries_table(self, html: str, team_id: int, team_name: str) -> pd.DataFrame:
        """Parse the real injuries/suspensions table structure into player rows.

        Same `table.items > tbody > tr` family as the squad table, but rows
        are grouped under single-cell category headers (`"Injuries"`,
        `"Suspension"`, ...) that apply to every player row until the next
        one -- tracked as `current_category` while walking the table.
        """
        soup = BeautifulSoup(html, "lxml")
        table = soup.select_one("table.items")
        if table is None:
            return pd.DataFrame()

        rows: list[dict[str, Any]] = []
        canonical_team_name = canonicalize_team_name(team_name)
        current_category: str | None = None
        for row in table.select("tbody > tr"):
            cells = row.find_all("td", recursive=False)
            if len(cells) == 1:
                current_category = cells[0].get_text(strip=True)
                continue
            if len(cells) < 6:
                continue
            name_link = cells[0].select_one("td.hauptlink a")
            if name_link is None:
                continue
            player_href = name_link.get("href", "")
            player_id_match = re.search(r"/spieler/(\d+)", player_href)
            rows.append(
                {
                    "team_name": canonical_team_name,
                    "team_external_id": team_id,
                    "player_name": name_link.get_text(strip=True),
                    "player_external_id": (
                        int(player_id_match.group(1)) if player_id_match else None
                    ),
                    "injury_category": current_category,
                    "injury_reason": cells[2].get_text(strip=True) or None,
                    "injury_since": cells[3].get_text(strip=True) or None,
                    "injury_expected_return": cells[4].get_text(strip=True) or None,
                }
            )
        return pd.DataFrame(rows)

    def refresh_injuries(
        self,
        team_ids: list[tuple[int, str, str]] | None = None,
        injuries_path: str | None = None,
        request_delay_seconds: float = 1.0,
    ) -> pd.DataFrame:
        """Fetch current injuries/suspensions for the given teams and replace the injuries store.

        Unlike `refresh_player_reference()`, this **overwrites**
        `injuries_path` rather than merging -- an injury list is a current
        snapshot, not an accumulating reference, so a recovered player
        should disappear on the next refresh instead of being stuck forever
        the way a null-safe column merge would leave them. As a low-cost
        bonus, also flags the same players' `injury_risk_score` in the
        player reference store (a coarser fallback signal if this dedicated
        store isn't refreshed for some reason) -- that part reuses the
        existing merge, so it inherits the same graceful degradation.
        """
        resolved_path = injuries_path if injuries_path else self.settings.injuries_path
        teams = team_ids if team_ids is not None else list(NATIONAL_TEAM_IDS)
        injury_frames: list[pd.DataFrame] = []
        for index, (team_id, team_slug, team_name) in enumerate(teams):
            if index > 0 and request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            try:
                injuries = self.fetch_national_team_injuries(team_id, team_slug, team_name)
            except Exception:
                LOGGER.warning("Failed to fetch Transfermarkt injuries for %s", team_name, exc_info=True)
                continue
            if not injuries.empty:
                injury_frames.append(injuries)
        combined = (
            pd.concat(injury_frames, ignore_index=True) if injury_frames else pd.DataFrame()
        )
        write_table(combined, resolved_path)
        if not combined.empty:
            risk_scores = combined[["team_name", "player_name", "player_external_id"]].drop_duplicates()
            risk_scores["injury_risk_score"] = 1.0
            try:
                update_player_reference_store(
                    self.settings.player_reference_path,
                    observed_players=risk_scores,
                    source_name="transfermarkt",
                )
            except Exception:
                LOGGER.warning("Failed to flag injury_risk_score in player reference", exc_info=True)
        return combined

    def refresh_player_reference(
        self,
        team_ids: list[tuple[int, str, str]] | None = None,
        player_reference_path: str | None = None,
        request_delay_seconds: float = 1.0,
    ) -> PlayerReferenceUpdateResult:
        """Fetch squads (with market values) for the given teams and merge into the player reference store.

        Resilient per team, same shape as
        `FootballApiSource.refresh_player_reference`: a single team's fetch
        failing is logged and skipped rather than aborting the whole
        refresh, and a total failure degrades gracefully (existing reference
        unchanged) instead of raising. `request_delay_seconds` is
        self-imposed courtesy rate limiting -- this is unauthenticated
        scraping of a real site, not a documented API.
        """
        resolved_path = (
            player_reference_path if player_reference_path else self.settings.player_reference_path
        )
        teams = team_ids if team_ids is not None else list(NATIONAL_TEAM_IDS)
        squads: list[pd.DataFrame] = []
        for index, (team_id, team_slug, team_name) in enumerate(teams):
            if index > 0 and request_delay_seconds > 0:
                time.sleep(request_delay_seconds)
            try:
                squad = self.fetch_national_team_squad(team_id, team_slug, team_name)
            except Exception:
                LOGGER.warning("Failed to fetch Transfermarkt squad for %s", team_name, exc_info=True)
                continue
            if not squad.empty:
                squads.append(squad)
        combined = pd.concat(squads, ignore_index=True) if squads else pd.DataFrame()
        return update_player_reference_store(
            resolved_path,
            observed_players=combined,
            source_name="transfermarkt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch Transfermarkt squad data for national teams.")
    parser.add_argument(
        "--mode",
        choices=("search", "squad", "injuries", "refresh-all", "refresh-injuries-all"),
        default="refresh-all",
        help=(
            "Discover a team's id/slug, fetch one team's squad or injuries, or refresh "
            "the full curated list (squads or injuries)."
        ),
    )
    parser.add_argument("--query", default=None, help="Team name to search for (search mode).")
    parser.add_argument("--team-id", type=int, default=None, help="Transfermarkt team id (squad mode).")
    parser.add_argument("--team-slug", default=None, help="Transfermarkt URL slug (squad mode).")
    parser.add_argument("--team-name", default=None, help="Canonical team name (squad mode).")
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    source = TransfermarktSource()
    if arguments.mode == "search":
        if not arguments.query:
            raise ValueError("--query is required when --mode search is used.")
        for result in source.search_teams(arguments.query):
            print(result)
    elif arguments.mode == "squad":
        if not (arguments.team_id and arguments.team_slug and arguments.team_name):
            raise ValueError("--team-id, --team-slug, and --team-name are required when --mode squad is used.")
        squad_frame = source.fetch_national_team_squad(
            arguments.team_id, arguments.team_slug, arguments.team_name
        )
        print(f"Fetched {len(squad_frame)} squad rows for {arguments.team_name}.")
        print(squad_frame.head(30).to_string(index=False))
    elif arguments.mode == "injuries":
        if not (arguments.team_id and arguments.team_slug and arguments.team_name):
            raise ValueError("--team-id, --team-slug, and --team-name are required when --mode injuries is used.")
        injuries_frame = source.fetch_national_team_injuries(
            arguments.team_id, arguments.team_slug, arguments.team_name
        )
        print(f"Fetched {len(injuries_frame)} injury rows for {arguments.team_name}.")
        print(injuries_frame.head(30).to_string(index=False))
    elif arguments.mode == "refresh-injuries-all":
        injuries_result = source.refresh_injuries()
        print(
            f"Refreshed injuries from {len(NATIONAL_TEAM_IDS)} national team(s): "
            f"{len(injuries_result)} current injuries/suspensions in {source.settings.injuries_path}."
        )
    else:
        result = source.refresh_player_reference()
        print(
            f"Refreshed player reference from {len(NATIONAL_TEAM_IDS)} national team squad(s): "
            f"+{result.added_players} added, {result.updated_players} updated, "
            f"{len(result.frame)} total players in {source.settings.player_reference_path}."
        )
