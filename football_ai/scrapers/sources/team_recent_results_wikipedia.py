"""Attendance and referee for a team's recent matches, from Wikipedia.

Deliberately narrow: this is *not* another feed into `records_frame` or
`training_frame`, and it does not try to cover history. It enriches the last
handful of matches the app already shows, with two fields no other source
here carries -- how many people were in the ground, and who refereed.

MEASURED, not assumed (2026-08-04, live against en.wikipedia.org). Pages
are titled `"<Team> national football team results (2020-present)"` and
embed each match in a `{{Football box collapsible}}` template. Real numbers
from that check:

- 15 of 18 sampled teams resolve to a parseable page (83%). Poland 82
  matches, England 94, France 93, Germany 89, Sweden 86, Malta 56.
- Brazil, Italy and Argentina have pages but lay their results out as a
  plain wikitable instead of this template -- they parse to zero rows.
  That is the uneven coverage this module is designed around, not a bug to
  fix: callers show what exists and stay silent otherwise.
- `attendance` and `referee` are each filled on 73 of Poland's 83 blocks.
  Eight of those attendances are a real, meaningful `0` -- September and
  October 2020 behind closed doors under COVID -- so a falsy attendance
  must never be treated as missing data.

Three title shapes are needed to reach that 83%, hence `_candidate_titles`:
the plain `(2020-present)` form; a `#REDIRECT` hop (Sweden's page lives at
`"Sweden men's national football team results (2020-present)"`, Brazil's at
`(2010-present)`); and a suffix-less `"<Team> national football team
results"` (San Marino, Gibraltar).

Structurally this template is close enough to `{{Football box}}` to share
`fetch_wikitext`/`_strip_wikilinks`/`_find_balanced_template_blocks` with
`uefa_nations_league`, but not close enough to reuse its parser: dates here
are plain text (`date = 4 September 2020`, not `{{Start date|...}}`), the
score is one `1-2` string rather than separate `goals1`/`goals2` counts,
and teams appear four different ways (`{{fb|POL}}`, `{{fb-rt|POL}}`,
`{{flagdeco|POL}} Poland`, `Poland {{flagdeco|POL}}`).
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from football_ai.config.fifa_country_codes import resolve_fifa_code
from football_ai.data.normalization import canonicalize_team_name
from football_ai.scrapers.base import BaseDataSource
from football_ai.scrapers.sources.uefa_nations_league import (
    _find_balanced_template_blocks,
    _strip_wikilinks,
)

LOGGER = logging.getLogger(__name__)

_WIKIPEDIA_API_URL = "https://en.wikipedia.org/w/index.php"
_USER_AGENT = "football_ai/1.0 (research project; contact via GitHub repo)"
_TEMPLATE_NAME = "Football box collapsible"

# `[^\S\n]` not `\s`: `\s` eats the newline and runs a blank field into the
# next line's value (found while checking this page's real markup).
_FIELD_LINE_PATTERN = re.compile(
    r"^\|[^\S\n]*([\w-]+)[^\S\n]*=[^\S\n]*(.*)$", re.MULTILINE
)
_REDIRECT_PATTERN = re.compile(r"^#REDIRECT\s*\[\[([^\]]+)\]\]", re.IGNORECASE)
_TEAM_CODE_PATTERN = re.compile(r"\{\{(?:fb|fb-rt|flagdeco|flagicon)\|([^}|]+)")
_ATTENDANCE_PATTERN = re.compile(r"^([\d,]+)")
# En-dash is what Wikipedia actually uses in scores; hyphen accepted anyway.
_SCORE_PATTERN = re.compile(r"^(\d+)\s*[-–—]\s*(\d+)")
_H2H_ROW_PATTERN = re.compile(
    r"^!scope=row[^|]*\|\s*\{\{fb\|([^}|]+)\}\}\s*\n\|(.+)$", re.MULTILINE
)
_YEAR_HEADER_PATTERN = re.compile(r"^==+\s*(\d{4})\s*==+\s*$", re.MULTILINE)

_RECENT_RESULTS_COLUMNS = [
    "match_date",
    "opponent",
    "is_home",
    "goals_for",
    "goals_against",
    "attendance",
    "referee",
    "competition",
]
_HEAD_TO_HEAD_COLUMNS = [
    "opponent",
    "played",
    "wins",
    "draws",
    "losses",
    "goals_for",
    "goals_against",
]


def empty_recent_results() -> pd.DataFrame:
    """The correctly-shaped empty frame every failure path returns."""
    return pd.DataFrame(
        {column: pd.Series(dtype="object") for column in _RECENT_RESULTS_COLUMNS}
    )


class TeamRecentResultsWikipediaSource(BaseDataSource):
    """Fetches attendance/referee for a national team's most recent matches."""

    def __init__(self) -> None:
        """Initialize the source."""
        super().__init__(source_name="team_recent_results_wikipedia")

    def fetch_wikitext(self, page_title: str) -> str:
        """Fetch a page's raw wikitext.

        Wikimedia's API etiquette policy 403s the default
        `python-requests/...` User-Agent, so an explicit one is required --
        same reason `uefa_nations_league` sets it.
        """
        return self.fetch_text_with_headers(
            _WIKIPEDIA_API_URL,
            headers={"User-Agent": _USER_AGENT},
            params={"title": page_title, "action": "raw"},
        )

    def fetch_page(self, team_name: str) -> str | None:
        """Resolve a team to its results page's wikitext, following one redirect hop.

        Returns `None` rather than raising when no candidate title exists --
        for most of the world's teams there simply is no such page, which is
        an ordinary outcome here, not an error.
        """
        canonical = canonicalize_team_name(team_name)
        for title in _candidate_titles(canonical):
            try:
                wikitext = self.fetch_wikitext(title)
            except Exception:  # noqa: BLE001 -- 404s are the common case, not a fault
                continue
            redirect = _REDIRECT_PATTERN.match(wikitext.strip())
            if redirect:
                try:
                    wikitext = self.fetch_wikitext(redirect.group(1))
                except Exception:  # noqa: BLE001
                    continue
            if _TEMPLATE_NAME.lower() in wikitext.lower():
                return wikitext
        LOGGER.warning("No parseable Wikipedia results page for %r", canonical)
        return None

    def fetch_recent_results(self, team_name: str, limit: int = 15) -> pd.DataFrame:
        """The team's `limit` most recent matches, newest last, with attendance/referee.

        Never raises on a missing page, a network failure or an unparseable
        layout -- returns `empty_recent_results()` instead, the same
        degrade-to-empty contract `_load_records_frame()` and
        `WorldFootballEloSource` follow.
        """
        try:
            wikitext = self.fetch_page(team_name)
        except Exception:  # noqa: BLE001 -- network faults degrade, they don't propagate
            LOGGER.warning("Wikipedia fetch failed for %r", team_name, exc_info=True)
            return empty_recent_results()
        if wikitext is None:
            return empty_recent_results()
        return parse_recent_results(wikitext, team_name, limit=limit)

    def fetch_head_to_head_table(self, team_name: str) -> pd.DataFrame:
        """The page's own head-to-head table -- an independent cross-check.

        Covers only the period the page spans (2020-present for most
        teams), so it is a reference point for that window, not a
        replacement for `insights.h2h`.
        """
        try:
            wikitext = self.fetch_page(team_name)
        except Exception:  # noqa: BLE001
            LOGGER.warning("Wikipedia fetch failed for %r", team_name, exc_info=True)
            return pd.DataFrame(columns=_HEAD_TO_HEAD_COLUMNS)
        if wikitext is None:
            return pd.DataFrame(columns=_HEAD_TO_HEAD_COLUMNS)
        return parse_head_to_head_table(wikitext)


def _candidate_titles(canonical_team: str) -> list[str]:
    """Every real title shape observed, most common first."""
    return [
        f"{canonical_team} national football team results (2020–present)",
        f"{canonical_team} men's national football team results (2020–present)",
        f"{canonical_team} national football team results",
    ]


def _parse_fields(block: str) -> dict[str, str]:
    """Flatten one template block's `|name = value` lines into a dict."""
    return {
        match.group(1): match.group(2).strip()
        for match in _FIELD_LINE_PATTERN.finditer(block)
    }


def _team_token(value: str) -> str | None:
    """A team field's raw token, unresolved -- `"POL"`, `"Bosnia and Herzegovina"`.

    Used to tell the two sides of a block apart without needing either to be
    resolvable, which matters: the FIFA code table here is UEFA-shaped (55
    codes), and measured against these pages 77 distinct codes fall outside
    it -- including `MAR` and `JPN`, the subject teams of their own pages.
    """
    code_match = _TEAM_CODE_PATTERN.search(value)
    if code_match:
        return code_match.group(1).strip()
    plain = _strip_wikilinks(value)
    return plain or None


def _resolve_team(token: str | None) -> str | None:
    """Best-effort canonical name, keeping the raw token when unmappable.

    A three-letter token is a FIFA code on match blocks (`POL`); the
    head-to-head table instead spells opponents out
    (`{{fb|Bosnia and Herzegovina}}`). An unresolved code is returned as-is
    rather than dropped -- callers key on match date, so an opaque opponent
    label costs nothing, while discarding the row would lose a real match.
    """
    if token is None:
        return None
    if len(token) == 3 and token.isalpha():
        try:
            return canonicalize_team_name(resolve_fifa_code(token.upper()))
        except KeyError:
            return token.upper()
    return canonicalize_team_name(token)


def _parse_attendance(value: str) -> int | None:
    """`"19,297"` -> 19297, `"0{{refn|...}}"` -> 0, `""` -> None.

    Zero is a real, deliberately-recorded figure on these pages (COVID
    behind-closed-doors matches), so it must survive as 0 and never
    collapse into "no data".
    """
    match = _ATTENDANCE_PATTERN.match(value.strip())
    return int(match.group(1).replace(",", "")) if match else None


def _parse_referee(value: str) -> str | None:
    """`"[[Szymon Marciniak]] ([[Polish FA|Poland]])"` -> `"Szymon Marciniak (Poland)"`.

    Runs of whitespace are collapsed because the source really does contain
    them (`"[[Maurizio Mariani]]  ([[...|Italy]])"` on Poland's page), and a
    double space is visible in the rendered caption.
    """
    without_refs = re.sub(r"<ref.*?(/>|>.*?</ref>)", "", value, flags=re.DOTALL)
    text = _strip_wikilinks(without_refs)
    return re.sub(r"\s+", " ", text).strip() or None


def _parse_date(value: str, default_year: int | None = None) -> pd.Timestamp | None:
    """`"4 September 2020"` -> Timestamp; anything unrecognised -> None.

    Some pages leave the year out of the field entirely and carry it in the
    enclosing `===2025===` heading instead (Malta writes `date = 17
    November`). Without `default_year` those parse to year 1, which is why
    the year is threaded down from the section rather than inferred.
    """
    text = _strip_wikilinks(value)
    if not text:
        return None
    if default_year is not None and not re.search(r"\b\d{4}\b", text):
        text = f"{text} {default_year}"
    parsed = pd.to_datetime(text, errors="coerce", dayfirst=True)
    if pd.isna(parsed) or pd.Timestamp(parsed).year < 1900:
        return None  # a yearless date with no section to borrow from -- don't guess
    return pd.Timestamp(parsed)


def _year_sections(wikitext: str) -> list[tuple[int | None, str]]:
    """Split a page into `(year, text)` chunks on its `===YYYY===` headings."""
    headers = list(_YEAR_HEADER_PATTERN.finditer(wikitext))
    if not headers:
        return [(None, wikitext)]
    sections: list[tuple[int | None, str]] = [(None, wikitext[: headers[0].start()])]
    for index, header in enumerate(headers):
        end = headers[index + 1].start() if index + 1 < len(headers) else len(wikitext)
        sections.append((int(header.group(1)), wikitext[header.end() : end]))
    return sections


def _subject_token(parsed_blocks: list[tuple[str | None, str | None]]) -> str | None:
    """The token on one side of nearly every block -- i.e. whose page this is.

    Inferred rather than matched against the requested name because the
    subject team's own code frequently isn't in the FIFA table (`MAR`,
    `JPN`), and a name-based match would silently yield zero rows for those
    teams. The page is by construction about one team, so the most frequent
    token is that team.
    """
    counts: dict[str, int] = {}
    for team1, team2 in parsed_blocks:
        for token in (team1, team2):
            if token:
                counts[token] = counts.get(token, 0) + 1
    return max(counts, key=lambda token: counts[token]) if counts else None


def parse_recent_results(
    wikitext: str, team_name: str, limit: int = 15
) -> pd.DataFrame:
    """Parse a results page into the subject team's most recent matches, newest last.

    `team_name` is used only to label the frame's own perspective; which
    side is the subject is inferred from the page itself (see
    `_subject_token`). Rows that aren't a real, played match -- a cancelled
    fixture, an unparseable date, a score that isn't `N-N` -- are dropped
    rather than guessed at.
    """
    parsed: list[tuple[int | None, dict[str, str]]] = []
    for year, section in _year_sections(wikitext):
        for block in _find_balanced_template_blocks(
            section, _TEMPLATE_NAME, case_sensitive=False
        ):
            parsed.append((year, _parse_fields(block)))
    tokens = [
        (_team_token(fields.get("team1", "")), _team_token(fields.get("team2", "")))
        for _, fields in parsed
    ]
    subject = _subject_token(tokens)
    if subject is None:
        return empty_recent_results()

    rows: list[dict[str, object]] = []
    for (year, fields), (team1, team2) in zip(parsed, tokens, strict=True):
        match_date = _parse_date(fields.get("date", ""), default_year=year)
        if match_date is None:
            continue
        if subject == team1:
            opponent_token, is_home = team2, True
        elif subject == team2:
            opponent_token, is_home = team1, False
        else:
            continue  # neither side is the subject team -- not this team's match

        score = _SCORE_PATTERN.match(_strip_wikilinks(fields.get("score", "")))
        if score is None:
            continue  # cancelled/awarded/not-yet-played -- no result to report
        team1_goals, team2_goals = int(score.group(1)), int(score.group(2))

        rows.append(
            {
                "match_date": match_date,
                "opponent": _resolve_team(opponent_token),
                "is_home": is_home,
                "goals_for": team1_goals if is_home else team2_goals,
                "goals_against": team2_goals if is_home else team1_goals,
                "attendance": _parse_attendance(fields.get("attendance", "")),
                "referee": _parse_referee(fields.get("referee", "")),
                "competition": _strip_wikilinks(fields.get("round", "")) or None,
            }
        )

    if not rows:
        return empty_recent_results()
    frame = pd.DataFrame(rows).sort_values("match_date").reset_index(drop=True)
    return frame.tail(limit).reset_index(drop=True)


def parse_head_to_head_table(wikitext: str) -> pd.DataFrame:
    """Parse the `==Head to head records==` wikitable into one row per opponent."""
    rows: list[dict[str, object]] = []
    for match in _H2H_ROW_PATTERN.finditer(wikitext):
        opponent = _resolve_team(match.group(1).strip())
        cells = [cell.strip() for cell in match.group(2).split("||")]
        if opponent is None or len(cells) < 6:
            continue
        if not all(cell.isdigit() for cell in cells[:6]):
            continue
        played, wins, draws, losses, goals_for, goals_against = (
            int(cell) for cell in cells[:6]
        )
        rows.append(
            {
                "opponent": opponent,
                "played": played,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "goals_for": goals_for,
                "goals_against": goals_against,
            }
        )
    return pd.DataFrame(rows, columns=_HEAD_TO_HEAD_COLUMNS)


if __name__ == "__main__":
    source = TeamRecentResultsWikipediaSource()
    print(source.fetch_recent_results("Poland").to_string())
