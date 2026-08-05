"""UEFA Nations League fixtures/results, parsed from Wikipedia's raw wikitext.

Built to close a real gap found 2026-07-30: api-football.com's `fixtures`
endpoint returns zero rows for the 2026/27 UEFA Nations League (league_id=5,
season=2026) even though the real-world league phase draw already happened
(12 February 2026, Brussels, confirmed via a real web search) -- a
data-provider sync gap, not a "the schedule doesn't exist yet" problem.

Wikipedia's Nations League articles embed every match in a structured
`{{Football box|...}}` template (verified directly, 2026-07-30, against
`2026–27 UEFA Nations League B`) -- fetched via MediaWiki's own raw-wikitext
endpoint (`action=raw`), not by scraping rendered HTML, so this doesn't
depend on any particular page layout/CSS surviving a redesign. Same
`goals1`/`goals2` fields go from empty to real numbers once a match is
played -- one consistent source for both the schedule and, as the
tournament progresses, results, unlike stitching together two different
providers for "what's left to play" vs "what already happened".

Deliberately competition-agnostic where the parsing itself is concerned
(`parse_football_box_matches` doesn't know or care it's being pointed at a
Nations League page) -- only `_page_title`/`find_team_group` below encode
anything UEFA Nations League-specific, so extending this to another
Wikipedia-tracked competition later is a new small orchestration function,
not a new parser.
"""

from __future__ import annotations

import re
from datetime import date

import pandas as pd

from football_ai.config.fifa_country_codes import resolve_fifa_code
from football_ai.data.normalization import canonicalize_team_name
from football_ai.scrapers.base import BaseDataSource


_WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/index.php"

_TEAM_CODE_PATTERN = re.compile(r"\{\{fb(?:-rt)?\|([A-Za-z]{3})")
_START_DATE_PATTERN = re.compile(r"\{\{Start date\|(\d{4})\|(\d{1,2})\|(\d{1,2})")
_FIELD_LINE_PATTERN = re.compile(r"^\|\s*(\w+)\s*=\s*(.*)$")
_GROUP_HEADER_PATTERN = re.compile(r"^===\s*Group\s+(\d+)\s*===\s*$", re.MULTILINE)
_SECTION_HEADER_PATTERN = re.compile(r"^===.*===\s*$", re.MULTILINE)


class UefaNationsLeagueSource(BaseDataSource):
    """Fetches and parses UEFA Nations League group fixtures/results from Wikipedia."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="uefa_nations_league")

    def fetch_wikitext(self, page_title: str) -> str:
        """Fetch a Wikipedia page's raw wikitext (not rendered HTML).

        Needs an explicit User-Agent -- Wikimedia's API etiquette policy
        rejects the default `python-requests/...` one with a 403 (confirmed
        for real, 2026-07-30 -- the same shape of problem Transfermarkt had
        earlier this session, different site).
        """
        return self.fetch_text_with_headers(
            _WIKIPEDIA_API_URL,
            headers={"User-Agent": "football_ai/1.0 (research project; contact via GitHub repo)"},
            params={"title": page_title, "action": "raw"},
        )

    def fetch_group_matches(
        self,
        division: str,
        group_number: int,
        season: str = "2026–27",
    ) -> pd.DataFrame:
        """Fetch and parse one division's one group -- e.g. `("B", 4, "2026–27")` for Poland's real group.

        `division` is `"A"`, `"B"`, `"C"`, or `"D"`. Returns the project's
        standard wide-match columns plus `stadium`; `home_score`/
        `away_score` are `pd.NA` for not-yet-played matches, real integers
        once Wikipedia has been updated after the match.
        """
        page_title = f"{season} UEFA Nations League {division}"
        wikitext = self.fetch_wikitext(page_title)
        group_wikitext = _extract_group_section(wikitext, group_number)
        return _matches_frame(parse_football_box_matches(group_wikitext))

    def find_team_group(self, team_name: str, season: str = "2026–27") -> tuple[str, int]:
        """Search League A/B/C/D for which division and group number a team is in.

        Four page fetches in the worst case (one per division) -- meant for
        occasional lookup/orchestration, not a hot path.
        """
        canonical = canonicalize_team_name(team_name)
        for division in ("A", "B", "C", "D"):
            page_title = f"{season} UEFA Nations League {division}"
            wikitext = self.fetch_wikitext(page_title)
            for group_number in range(1, 5):
                group_wikitext = _extract_group_section(wikitext, group_number)
                if group_wikitext is None:
                    continue
                matches = parse_football_box_matches(group_wikitext)
                teams_in_group = {
                    canonicalize_team_name(resolve_fifa_code(match["team1_code"]))
                    for match in matches
                } | {
                    canonicalize_team_name(resolve_fifa_code(match["team2_code"]))
                    for match in matches
                }
                if canonical in teams_in_group:
                    return division, group_number
        raise ValueError(f"Could not find {team_name!r} in any UEFA Nations League {season} division/group.")


def _extract_group_section(wikitext: str, group_number: int) -> str | None:
    """Slice out just one `===Group N===` section's wikitext, up to the next `===...===` header of ANY kind.

    Must stop at the next section header regardless of what it's titled --
    a naive "next ===Group N=== header" cut would, for the *last* group on
    the page, swallow everything after it (e.g. a trailing
    `===Overall ranking===` section) since there's no further group header
    to bound it.
    """
    all_headers = list(_SECTION_HEADER_PATTERN.finditer(wikitext))
    for header_match in _GROUP_HEADER_PATTERN.finditer(wikitext):
        if int(header_match.group(1)) != group_number:
            continue
        start = header_match.end()
        end = next((h.start() for h in all_headers if h.start() >= start), len(wikitext))
        return wikitext[start:end]
    return None


def _find_balanced_template_blocks(
    wikitext: str, template_name: str, case_sensitive: bool = True
) -> list[str]:
    """Extract every `{{template_name ... }}` block, tracking brace depth so nested templates don't truncate it early.

    `case_sensitive=False` is for pages that spell the same template both
    ways -- the per-team results pages mix `{{Football box collapsible}}`
    and `{{football box collapsible}}` in one article (measured on Poland's
    2020-present page, 2026-08-04: 56 capitalised, 26 not). Default stays
    case-sensitive so the Nations League parser above is untouched.

    Matching is done with a case-insensitive regex rather than by searching
    a lowercased copy, because `str.lower()` is not length-preserving:
    Japan's results page contains `İ` (U+0130, in a Turkish referee's
    name), which lowercases to two codepoints and shifted every subsequent
    offset by one -- slicing blocks a character late, so they began `{F`
    instead of `{{F` and the brace walk truncated them mid-template. Found
    for real, 2026-08-04.
    """
    blocks: list[str] = []
    marker = "{{" + template_name
    pattern = re.compile(re.escape(marker), 0 if case_sensitive else re.IGNORECASE)
    search_start = 0
    while True:
        found = pattern.search(wikitext, search_start)
        if found is None:
            break
        start = found.start()
        depth = 0
        index = start
        end = None
        while index < len(wikitext):
            if wikitext.startswith("{{", index):
                depth += 1
                index += 2
                continue
            if wikitext.startswith("}}", index):
                depth -= 1
                index += 2
                if depth == 0:
                    end = index
                    break
                continue
            index += 1
        if end is None:
            break  # unbalanced template -- stop rather than guess
        blocks.append(wikitext[start:end])
        search_start = end
    return blocks


def parse_football_box_matches(wikitext: str) -> list[dict[str, object]]:
    """Parse every `{{Football box|...}}` template in `wikitext` into a structured record.

    Returns raw field values (FIFA codes, not canonical names; `date.date`
    objects; `None` for not-yet-played goals) -- callers resolve team codes
    and build a project-schema frame (see `_matches_frame`).
    """
    records: list[dict[str, object]] = []
    for block in _find_balanced_template_blocks(wikitext, "Football box"):
        fields: dict[str, str] = {}
        for line in block.splitlines():
            field_match = _FIELD_LINE_PATTERN.match(line.strip())
            if field_match:
                fields[field_match.group(1)] = field_match.group(2).strip()

        date_match = _START_DATE_PATTERN.search(fields.get("date", ""))
        if date_match is None:
            continue  # not a real match block (e.g. a template doc example) -- skip, don't guess a date
        match_date = date(int(date_match.group(1)), int(date_match.group(2)), int(date_match.group(3)))

        team1_match = _TEAM_CODE_PATTERN.search(fields.get("team1", ""))
        team2_match = _TEAM_CODE_PATTERN.search(fields.get("team2", ""))
        if team1_match is None or team2_match is None:
            continue

        records.append(
            {
                "match_date": match_date,
                "team1_code": team1_match.group(1).upper(),
                "team2_code": team2_match.group(1).upper(),
                "goals1": _parse_goals(fields.get("goals1", "")),
                "goals2": _parse_goals(fields.get("goals2", "")),
                "stadium": _strip_wikilinks(fields.get("stadium", "")),
            }
        )
    return records


def _parse_goals(value: str) -> int | None:
    """A goals field is either a plain integer (played) or blank (not yet played)."""
    stripped = value.strip()
    if not stripped:
        return None
    digits = re.match(r"\d+", stripped)
    return int(digits.group(0)) if digits else None


def _strip_wikilinks(value: str) -> str:
    """Best-effort plain text from a wikitext value -- strips `[[...]]`/`<ref>...`/`{{...}}` markup.

    Not correctness-critical (stadium is a display field, not a merge key),
    so this doesn't need to be perfect -- just readable.
    """
    without_refs = re.sub(r"<ref[^/]*?(/>|>.*?</ref>)", "", value, flags=re.DOTALL)
    without_templates = re.sub(r"\{\{.*?\}\}", "", without_refs)
    without_links = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]", r"\1", without_templates)
    return without_links.strip()


def _matches_frame(records: list[dict[str, object]]) -> pd.DataFrame:
    """Resolve FIFA codes to canonical names and build the project's standard wide-match columns."""
    rows = []
    for record in records:
        rows.append(
            {
                "match_date": pd.Timestamp(record["match_date"]),
                "home_team_name": canonicalize_team_name(resolve_fifa_code(record["team1_code"])),
                "away_team_name": canonicalize_team_name(resolve_fifa_code(record["team2_code"])),
                "home_score": record["goals1"] if record["goals1"] is not None else pd.NA,
                "away_score": record["goals2"] if record["goals2"] is not None else pd.NA,
                "stadium": record["stadium"],
            }
        )
    return pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
