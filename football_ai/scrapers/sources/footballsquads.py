"""Historical tournament squads importer (footballcsv/cache.footballsquads).

CC0-licensed cache of footballsquads.co.uk, confirmed 2026-08-20 (see
`PROMPT_FOOTBALLSQUADS.md`). Fills a real, documented gap: `features/builder.py`
has always zero-filled `home_lineup_average_age`/`home_lineup_market_value`
for historical rows because no point-in-time historical squad record existed
-- only current live snapshots. This is the first source giving real,
historical, per-tournament squads.

Phase 1 scope only: national-team squads at five tournaments (World Cup,
UEFA Euro, Copa América, Africa Cup of Nations, Confederations Cup) --
`national/{competition}/{prefix}{year}/{team}.txt`. Club squads
(`squads.htm`/`archive.htm`) are a deliberately separate, larger Phase 2,
per the plan.

A squad file is a snapshot of ONE moment (tournament start), not a
per-match lineup -- this only has coverage for matches at or near a covered
tournament, never for ordinary qualifiers/friendlies between tournaments.
Not wired into `features/builder.py` here -- that's a separate, explicit
decision, same as the neutral-venue fix this week.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import time
from dataclasses import dataclass
from io import StringIO

import pandas as pd

from football_ai.data.io import read_table, write_table
from football_ai.data.normalization import canonicalize_team_name, normalize_text
from football_ai.scrapers.base import BaseDataSource

LOGGER = logging.getLogger(__name__)

_REPO = "footballcsv/cache.footballsquads"
_TREE_URL = f"https://api.github.com/repos/{_REPO}/git/trees/master?recursive=1"
_RAW_BASE = f"https://raw.githubusercontent.com/{_REPO}/master"

# competition slug (footballsquads folder name) -> the exact `tournament`
# string this project's martj42-derived records_frame uses for the FINALS
# (never the "... qualification" variant) -- verified 2026-08-20 against
# `international_results_full_history.parquet`'s real distinct values.
_COMPETITION_TOURNAMENT_NAMES: dict[str, str] = {
    "worldcup": "FIFA World Cup",
    "eurocham": "UEFA Euro",
    "copaamer": "Copa América",
    "africacn": "African Cup of Nations",
    "confed": "Confederations Cup",
}

_HEADER_LINE_PATTERN = re.compile(
    r"^=\s*(?P<team>.+?)\s*-\s*(?P<label>.+?)\s+(?P<year>\d{4})\s*$"
)
_METRIC_HEIGHT_PATTERN = re.compile(r"^\d\.\d{2}$")
_IMPERIAL_HEIGHT_PATTERN = re.compile(r"^(?P<feet>\d)'(?P<inches>\d{1,2})\"?$")
_MIN_PLAUSIBLE_AGE = 14
_MAX_PLAUSIBLE_AGE = 50


@dataclass(slots=True)
class SquadFileRef:
    """One `national/{competition}/{prefix}{year}/{team}.txt` file."""

    competition: str
    year: int
    team_slug: str
    path: str


def _parse_year_folder(competition: str, year_folder: str) -> int | None:
    """Extract the 4-digit year from a year-folder name (e.g. "wc2022", "acn2023")."""
    match = re.search(r"(\d{4})$", year_folder)
    return int(match.group(1)) if match else None


def _parse_height_meters(raw_value: str) -> float | None:
    r"""Parse a height cell into meters, handling both eras' real formats.

    Recent squads (~1994+) use decimal metres ("1.95"); older squads (e.g.
    the 1950 files) use feet/inches ("5'09\""). Confirmed both formats for
    real by fetching wc1950/brazil.txt and wc2022/poland.txt directly.
    Anything else (blank, unparseable) returns `None` -- never guessed.
    """
    value = raw_value.strip()
    if not value:
        return None
    if _METRIC_HEIGHT_PATTERN.match(value):
        return float(value)
    imperial_match = _IMPERIAL_HEIGHT_PATTERN.match(value)
    if imperial_match:
        feet = int(imperial_match.group("feet"))
        inches = int(imperial_match.group("inches"))
        return round(feet * 0.3048 + inches * 0.0254, 2)
    return None


def _parse_birth_date(raw_value: str, tournament_year: int) -> pd.Timestamp | None:
    """Parse a "DD-MM-YY" birth date, resolving the 2-digit year's century.

    `pandas`'/`strptime`'s default pivot-year heuristic (roughly 69/70) is
    wrong here: e.g. euro2024 squads include players born in the 2000s
    ("YY" like "03"), which a naive pivot would read correctly, but wc1950
    squads have players born in the 1920s ("YY" like "21") which a naive
    pivot reads as 2021 -- decades in the future relative to the 1950
    tournament. Resolved instead by picking whichever century puts the
    player's age at the tournament in a plausible human-footballer range
    (14-50); this is well-defined because the two candidate centuries are
    always 100 years apart, so at most one can ever land in that range for
    a real footballer.
    """
    value = raw_value.strip()
    if not value:
        return None
    day_month_year = re.match(r"^(\d{1,2})-(\d{1,2})-(\d{2})$", value)
    if not day_month_year:
        return None
    day, month, two_digit_year = (int(part) for part in day_month_year.groups())
    for century_base in (1900, 2000):
        birth_year = century_base + two_digit_year
        age_at_tournament = tournament_year - birth_year
        if _MIN_PLAUSIBLE_AGE <= age_at_tournament <= _MAX_PLAUSIBLE_AGE:
            try:
                return pd.Timestamp(year=birth_year, month=month, day=day, tz="UTC")
            except ValueError:
                return None
    return None


def _parse_squad_text(
    raw_text: str, competition: str, year: int, team_slug: str
) -> pd.DataFrame:
    """Parse one squad file's raw text into a per-player DataFrame.

    Column presence varies by era (confirmed for real: wc1950 files have no
    "Nat" column, wc2022 files do) -- read the actual header line rather
    than assuming a fixed column layout, per the source prompt's explicit
    warning not to assume the schema.
    """
    lines = raw_text.splitlines()
    header_match = None
    header_row_index = None
    for index, line in enumerate(lines):
        candidate = _HEADER_LINE_PATTERN.match(line)
        if candidate:
            header_match = candidate
            header_row_index = index
            continue
        if header_row_index is not None and "Name" in line and "," in line:
            column_row_index = index
            break
    else:
        column_row_index = None

    if header_match is None or column_row_index is None:
        LOGGER.warning(
            "Could not parse header for %s/%s/%s -- skipping.",
            competition,
            year,
            team_slug,
        )
        return pd.DataFrame()

    columns = [c.strip() for c in lines[column_row_index].split(",")]
    data_lines = [line for line in lines[column_row_index + 1 :] if line.strip()]
    reader = csv.reader(StringIO("\n".join(data_lines)), skipinitialspace=True)

    rows: list[dict[str, str]] = []
    for record in reader:
        if len(record) != len(columns):
            LOGGER.warning(
                "Skipping malformed row in %s/%s/%s (expected %d fields, got %d).",
                competition,
                year,
                team_slug,
                len(columns),
                len(record),
            )
            continue
        rows.append(dict(zip(columns, record, strict=True)))

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame

    frame["competition"] = competition
    frame["tournament_year"] = year
    frame["team_slug"] = team_slug
    frame["team_name"] = canonicalize_team_name(header_match.group("team"))
    frame["height_m"] = frame.get("Height", pd.Series(dtype=object)).map(
        _parse_height_meters
    )
    frame["birth_date"] = frame.get("Date of Birth", pd.Series(dtype=object)).map(
        lambda value: _parse_birth_date(value, year)
    )
    frame["position"] = frame.get("Pos", pd.Series(dtype=object)).str.strip()
    frame["player_name"] = frame.get("Name", pd.Series(dtype=object)).str.strip()
    return frame


# (competition, branding_year) -> the real year matches were actually played.
# Three tournaments kept their original marketing name after being postponed
# (COVID-19 for the first, scheduling/weather for the other two) -- verified
# directly against `international_results_full_history.parquet` (martj42),
# not guessed: "eurocham/euro2020" squad files have no "UEFA Euro" matches
# with match_date.year==2020 at all (the real Euro 2020 was played
# 2021-06-11 to 2021-07-11); "africacn/acn2021" likewise has no matches in
# 2021 (real dates 2022-01-09 to 2022-02-06); "africacn/acn2023" likewise
# (real dates 2024-01-13 to 2024-02-11). `tournament_year` in this module's
# output frames stays the branding year throughout (matching the source's
# own folder names and how these tournaments are colloquially referenced)
# -- only the internal records_frame lookup below needs the real year.
_BRANDING_YEAR_TO_ACTUAL_YEAR: dict[tuple[str, int], int] = {
    ("eurocham", 2020): 2021,
    ("africacn", 2021): 2022,
    ("africacn", 2023): 2024,
}


def _tournament_start_date(
    records_frame: pd.DataFrame, competition: str, year: int
) -> pd.Timestamp | None:
    """Look up a tournament's real start date from the already-owned records_frame.

    Derived from data already fetched (martj42), not a hand-maintained
    date table -- per the source prompt's suggestion. `records_frame` must
    already be UTC-aware (see `international_results.py`'s
    `normalize_results`); comparing naive/aware timestamps would raise.
    """
    tournament_name = _COMPETITION_TOURNAMENT_NAMES[competition]
    actual_year = _BRANDING_YEAR_TO_ACTUAL_YEAR.get((competition, year), year)
    matches = records_frame.loc[
        (records_frame["tournament"] == tournament_name)
        & (records_frame["match_date"].dt.year == actual_year)
    ]
    if matches.empty:
        return None
    return matches["match_date"].min()


def _position_bucket(position: str) -> str | None:
    """Map a raw position code onto one of the four standard buckets, or None."""
    code = position.strip().upper()
    if code in {"G", "GK"}:
        return "goalkeepers"
    if code in {"D", "DF"}:
        return "defenders"
    if code in {"M", "MF"}:
        return "midfielders"
    if code in {"F", "FW", "S"}:
        return "forwards"
    return None


_LOAN_SUFFIX_PATTERN = re.compile(r"\s*\((?:on loan|loan)\)\s*$", re.IGNORECASE)


def _normalize_club_name(raw_club: str) -> str | None:
    """Normalize a "Current Club" cell for cohesion-index grouping.

    Minimal, not a full alias table (no club equivalent of
    `config.team_aliases` exists in this project yet, and one club-name
    task doesn't warrant building one) -- strips the real, high-volume
    "(On Loan)" suffix confirmed in the actual data (820 of 2,887 unique
    raw club strings, 2026-08-25: e.g. "Aston Villa" and "Aston Villa (On
    Loan)" are the same club, appearing as two different strings only
    because one player's row happens to be a loan spell), then reuses
    `normalize_text` (accent/case) already used elsewhere in this module
    for continuity matching. Doesn't catch every real-world variant (e.g.
    "Tottenham H" vs "Tottenham Hotspur") -- a genuine limitation, not
    silently pretended away.
    """
    without_loan_suffix = _LOAN_SUFFIX_PATTERN.sub("", raw_club)
    normalized = normalize_text(without_loan_suffix)
    return normalized or None


def _club_cohesion_index(players: pd.DataFrame) -> dict[str, object]:
    """Club Cohesion Index: how concentrated a squad is at its largest shared club.

    Per `PROMPT_FOOTBALLSQUADS.md`'s spec: not a flat pair count -- a squad
    with one 5-player cluster at the same club plays differently than one
    with five unrelated pairs, even at the same total pair count, so the
    headline number is the *largest* cluster's size (converted to
    C(n,2) connected pairs), with the sum across every 2+ cluster kept as
    a secondary total. A fact for `insights/`+`narrative/` first, per that
    spec -- deliberately not merged into `features/builder.py` here.
    """
    raw_clubs = players.get("Current Club", pd.Series(dtype=object))
    normalized_clubs = raw_clubs.dropna().map(_normalize_club_name).dropna()
    cluster_sizes = normalized_clubs.value_counts()
    shared_clusters = cluster_sizes[cluster_sizes >= 2]

    if shared_clusters.empty:
        return {
            "largest_club_cluster_size": 0,
            "largest_club_cluster_name": None,
            "largest_club_cluster_pairs": 0,
            "total_intra_club_pairs": 0,
        }

    largest_size = int(shared_clusters.iloc[0])
    largest_name = str(shared_clusters.index[0])
    total_pairs = int(sum(size * (size - 1) // 2 for size in shared_clusters))
    return {
        "largest_club_cluster_size": largest_size,
        "largest_club_cluster_name": largest_name,
        "largest_club_cluster_pairs": largest_size * (largest_size - 1) // 2,
        "total_intra_club_pairs": total_pairs,
    }


def _squad_summary(
    players: pd.DataFrame,
    competition: str,
    year: int,
    team_slug: str,
    tournament_start: pd.Timestamp | None,
) -> dict[str, object]:
    """Aggregate one team's squad file into the per-team-per-tournament summary row."""
    ages_years: list[float] = []
    if tournament_start is not None:
        birth_dates = players["birth_date"].dropna()
        ages_years = [
            (tournament_start - birth_date).days / 365.25 for birth_date in birth_dates
        ]

    heights = players["height_m"].dropna()
    position_counts = {
        "goalkeepers": 0,
        "defenders": 0,
        "midfielders": 0,
        "forwards": 0,
    }
    for position in players["position"].dropna():
        bucket = _position_bucket(position)
        if bucket is not None:
            position_counts[bucket] += 1

    average_age = (sum(ages_years) / len(ages_years)) if ages_years else None
    average_height = float(heights.mean()) if not heights.empty else None
    return {
        "competition": competition,
        "tournament_year": year,
        "team_slug": team_slug,
        "team_name": players["team_name"].iloc[0] if not players.empty else None,
        "tournament_start_date": tournament_start,
        "squad_size": len(players),
        "average_age_years": average_age,
        "age_known_count": len(ages_years),
        "average_height_m": average_height,
        "height_known_count": int(len(heights)),
        **position_counts,
        **_club_cohesion_index(players),
    }


def _continuity_ratio(
    current_names: set[str], previous_names: set[str]
) -> float | None:
    """Best-effort continuity: fraction of the current squad also in the previous one.

    Matched on `normalize_text`-normalized names (accent/case-insensitive
    exact match) -- spelling isn't perfectly consistent across tournament
    years in this source, so this is explicitly an approximation, not an
    exact headcount. Returns `None` when there's no previous squad to
    compare against (first tournament a team appears in this source).
    """
    if not previous_names:
        return None
    if not current_names:
        return 0.0
    overlap = len(current_names & previous_names)
    return overlap / len(current_names)


def summarize_squads(
    players_frame: pd.DataFrame, records_frame: pd.DataFrame
) -> pd.DataFrame:
    """Aggregate an already-parsed `players_frame` into per-team-per-tournament rows.

    Shared by `FootballSquadsSource.build_squads()` (fresh fetch) and any
    offline recomputation from the cached `tournament_squads_players.parquet`
    -- e.g. after adding a new summary field, there's no need to re-fetch
    every file from GitHub, only to re-run this aggregation over what's
    already been fetched and parsed. Groups are visited in the same
    (competition, tournament_year, team_slug) order `build_squads()` used to
    walk `refs_sorted`, so `squad_continuity_ratio` (which depends on visit
    order, comparing each squad against the chronologically previous one for
    that team+competition) matches what a live re-fetch would produce.
    """
    if players_frame.empty:
        return pd.DataFrame()

    summary_rows: list[dict[str, object]] = []
    previous_squad_names: dict[tuple[str, str], set[str]] = {}
    group_keys = (
        players_frame[["competition", "tournament_year", "team_slug"]]
        .drop_duplicates()
        .sort_values(["competition", "tournament_year", "team_slug"])
    )
    for _, key_row in group_keys.iterrows():
        competition = key_row["competition"]
        year = int(key_row["tournament_year"])
        team_slug = key_row["team_slug"]
        players = players_frame.loc[
            (players_frame["competition"] == competition)
            & (players_frame["tournament_year"] == year)
            & (players_frame["team_slug"] == team_slug)
        ]

        tournament_start = _tournament_start_date(records_frame, competition, year)
        summary_rows.append(
            _squad_summary(players, competition, year, team_slug, tournament_start)
        )

        current_names = {normalize_text(name) for name in players["player_name"]}
        team_key = (competition, team_slug)
        summary_rows[-1]["squad_continuity_ratio"] = _continuity_ratio(
            current_names, previous_squad_names.get(team_key, set())
        )
        previous_squad_names[team_key] = current_names

    return pd.DataFrame(summary_rows)


class FootballSquadsSource(BaseDataSource):
    """Fetches and normalizes footballcsv/cache.footballsquads' national-team squads."""

    def __init__(self, request_delay_seconds: float = 0.15) -> None:
        """Initialize the source. `request_delay_seconds` throttles per-file fetches."""
        super().__init__(source_name="footballsquads")
        self.request_delay_seconds = request_delay_seconds

    def list_squad_files(self) -> list[SquadFileRef]:
        """List every national-team squad file via a single GitHub Trees API call.

        One recursive-tree call instead of walking directories one call per
        competition/year (which would cost 90+ GitHub API requests and risk
        the unauthenticated 60/hour rate limit) -- confirmed 2026-08-20 that
        this repo's tree isn't large enough to hit the API's truncation limit.
        """
        payload = self.fetch_json(_TREE_URL)
        refs: list[SquadFileRef] = []
        for entry in payload.get("tree", []):
            path = entry["path"]
            if not path.startswith("national/") or not path.endswith(".txt"):
                continue
            parts = path.split("/")
            if len(parts) != 4:
                continue
            _, competition, year_folder, file_name = parts
            if competition not in _COMPETITION_TOURNAMENT_NAMES:
                continue
            year = _parse_year_folder(competition, year_folder)
            if year is None:
                continue
            refs.append(
                SquadFileRef(
                    competition=competition,
                    year=year,
                    team_slug=file_name.removesuffix(".txt"),
                    path=path,
                )
            )
        return refs

    def fetch_squad_text(self, path: str) -> str:
        """Fetch one squad file's raw text."""
        return self.fetch_text(f"{_RAW_BASE}/{path}")

    def build_squads(
        self, records_frame: pd.DataFrame, refs: list[SquadFileRef] | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch and parse every squad file, returning (players_frame, summary_frame).

        `players_frame` is one row per player per tournament (raw). The
        aggregation into `summary_frame` is delegated to `summarize_squads()`
        so a later re-aggregation (e.g. after adding a new summary field)
        can run offline against an already-fetched `players_frame` without
        re-fetching every file from GitHub.
        """
        if refs is None:
            refs = self.list_squad_files()

        player_frames: list[pd.DataFrame] = []
        refs_sorted = sorted(
            refs, key=lambda ref: (ref.competition, ref.year, ref.team_slug)
        )
        for index, ref in enumerate(refs_sorted):
            if index > 0:
                time.sleep(self.request_delay_seconds)
            try:
                raw_text = self.fetch_squad_text(ref.path)
            except Exception as error:  # noqa: BLE001 -- one bad fetch shouldn't abort the run
                LOGGER.warning("Failed to fetch %s: %s", ref.path, error)
                continue

            players = _parse_squad_text(
                raw_text, ref.competition, ref.year, ref.team_slug
            )
            if players.empty:
                continue
            player_frames.append(players)

        players_frame = (
            pd.concat(player_frames, ignore_index=True)
            if player_frames
            else pd.DataFrame()
        )
        summary_frame = summarize_squads(players_frame, records_frame)
        return players_frame, summary_frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fetch and normalize footballcsv/cache.footballsquads' squads."
    )
    parser.add_argument(
        "--records-frame",
        default="football_ai/data/processed/international_results_full_history.parquet",
        help="Path to the full-history records_frame (for tournament start dates).",
    )
    parser.add_argument(
        "--players-output",
        default="football_ai/data/processed/tournament_squads_players.parquet",
        help="Output path for the per-player-per-tournament raw frame.",
    )
    parser.add_argument(
        "--summary-output",
        default="football_ai/data/processed/tournament_squads_summary.parquet",
        help="Output path for the per-team-per-tournament aggregated frame.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.15,
        help="Seconds to sleep between per-file fetches (be polite to a hobby site).",
    )
    arguments = parser.parse_args()

    log_format = "%(levelname)s %(name)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_format)
    records = read_table(arguments.records_frame)

    source = FootballSquadsSource(request_delay_seconds=arguments.request_delay)
    refs = source.list_squad_files()
    n_competitions = len(_COMPETITION_TOURNAMENT_NAMES)
    print(f"Found {len(refs)} squad files across {n_competitions} competitions.")

    players_frame, summary_frame = source.build_squads(records, refs=refs)
    write_table(players_frame, arguments.players_output)
    write_table(summary_frame, arguments.summary_output)
    print(f"players_frame: {len(players_frame)} rows -> {arguments.players_output}")
    print(f"summary_frame: {len(summary_frame)} rows -> {arguments.summary_output}")
