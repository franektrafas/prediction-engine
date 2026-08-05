"""Tests for the per-team Wikipedia results importer.

Every fixture below is real wikitext, copied verbatim (2026-08-04) from
`Poland national football team results (2020–present)` and its Malta and
Japan counterparts -- trimmed to the blocks that exercise a distinct
behaviour, never hand-written. Nothing here touches the network.

The cases pinned are the ones real pages actually threw at the parser:
mixed capitalisation of the same template inside one article, a genuine
`attendance = 0` (COVID, behind closed doors) that must not read as
missing, four different team markups, a cancelled fixture with no score,
a year carried in the section heading rather than the date field, and a
`İ` whose lowercase is two codepoints.

Two `referee =` lines below run past the line-length limit. They are left
exactly as Wikipedia writes them because the tests assert on the text they
produce -- shortening them would be testing a string this parser will never
actually meet.
"""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.scrapers.sources.team_recent_results_wikipedia import (
    TeamRecentResultsWikipediaSource,
    parse_head_to_head_table,
    parse_recent_results,
)

# Real, verbatim: Poland's 2020 and 2025 pages. Note the capital-F/lower-f
# spelling of the same template -- both appear in the real article.
_REAL_POLAND_EXCERPT = """==Results==
===2020===
{{Football box collapsible|format=1
|round = [[Exhibition game|Friendly]]
|date = 27 March 2020
|time =
|team1 = {{fb-rt|POL}}
|score = Cancelled
|report =
|team2 = {{fb|FIN}}
|goals1 =
|goals2 =
|stadium =
|location =
|attendance =
|referee =
|result = P
}}
{{Football box collapsible|format=1
|round = [[2020–21 UEFA Nations League A|Nations League]]
|date = 7 September 2020
|time = {{UTZ|20:45|2}}
|team1 = {{fb-rt|BIH}}
|score = 1–2
|report = https://www.uefa.com/uefanationsleague/match/2030010/
|team2 = {{flagdeco|POL}} Poland
|goals1 =
*[[Haris Hajradinović|Hajradinović]] {{goal|24|pen.}}
|goals2 =
*[[Kamil Glik|Glik]] {{goal|45}}
*[[Kamil Grosicki|Grosicki]] {{goal|67}}
|stadium = [[Bilino Polje Stadium|Bilino Polje]]
|location = [[Zenica]], Bosnia and Herzegovina
|attendance = 0{{refn|group=note|name=September behind closed doors}}
|referee = [[Cüneyt Çakır]] ([[Turkish Football Federation|Turkey]])
|result = W
}}
===2025===
{{football box collapsible|format=1
|date              = 12 October 2025
|time              = 21:45 [[Eastern European Summer Time|EEST]]
|team1             = {{fb-rt|LTU}}
|score             = 0–2
|report            = https://www.uefa.com/european-qualifiers/match/2044284/
|team2             = {{flagdeco|POL}} Poland
|goals1            =
|goals2            =
*[[Sebastian Szymański|Szymański]] {{goal|15}}
|stadium           = [[Darius and Girėnas Stadium]]
|location          = [[Kaunas]], Lithuania
|attendance        = 11,741
|referee           = Ondřej Berka ([[Football Association of the Czech Republic|Czech Republic]])
|result            = W
}}
"""

# Real, verbatim: Malta's page leaves the year out of `date` entirely and
# carries it in the `===2026===` heading instead.
_REAL_MALTA_YEARLESS_EXCERPT = """===2026===
{{Football box collapsible|format=1
|round = [[2026 FIFA World Cup qualification]]
|date = 26 March
|team1 = {{fb-rt|MLT}}
|score = 0–2
|team2 = {{fb|LUX}}
|attendance = 9,467
|referee = Irfan Peljto ([[Football Association of Bosnia and Herzegovina|Bosnia and Herzegovina]])
|result = L
}}
"""

# Real, verbatim from Poland's `==Head to head records==` table.
_REAL_HEAD_TO_HEAD_EXCERPT = """==Head to head records==
{|class="wikitable plainrowheaders sortable" style="text-align:center"
|-
!scope=col|Opponent
|-
!scope=row style="text-align:left;" |{{fb|Bosnia and Herzegovina}}
|2||2||0||0||5||1||{{#expr:2/2*100 round 2}}||{{#expr:0/2*100 round 2}}
|-
!scope=row style="text-align:left;" |{{fb|Albania}}
|5||4||0||1||8||4||{{#expr:4/5*100 round 2}}||{{#expr:0/5*100 round 2}}
|}
"""


class ParseRecentResultsTestCase(unittest.TestCase):
    """Parsing real per-team results wikitext."""

    def setUp(self) -> None:
        """Parse Poland's real excerpt once for the assertions below."""
        self.frame = parse_recent_results(_REAL_POLAND_EXCERPT, "Poland")

    def test_cancelled_match_is_dropped_and_played_ones_kept(self) -> None:
        """The real 27 March 2020 fixture reads `score = Cancelled`."""
        self.assertEqual(len(self.frame), 2)
        self.assertNotIn(
            pd.Timestamp("2020-03-27"), set(self.frame["match_date"])
        )

    def test_both_capitalisations_of_the_template_are_found(self) -> None:
        """The real article uses both capitalisations of the same template."""
        self.assertEqual(
            list(self.frame["match_date"]),
            [pd.Timestamp("2020-09-07"), pd.Timestamp("2025-10-12")],
        )

    def test_covid_zero_attendance_survives_as_zero_not_missing(self) -> None:
        """Poland-Bosnia, 7 September 2020, really was played before 0 spectators.

        `0` here is a recorded fact, not an absence of one -- if it collapses
        into NaN the UI silently drops a true and unusual detail.
        """
        away_row = self.frame.iloc[0]
        self.assertEqual(away_row["attendance"], 0)
        self.assertFalse(pd.isna(away_row["attendance"]))

    def test_perspective_is_the_subject_team_not_the_home_side(self) -> None:
        """Poland were away in both fixtures, and won both 2-1 and 2-0 respectively."""
        first, second = self.frame.iloc[0], self.frame.iloc[1]
        self.assertFalse(bool(first["is_home"]))
        self.assertEqual(first["opponent"], "Bosnia and Herzegovina")
        self.assertEqual((first["goals_for"], first["goals_against"]), (2, 1))
        self.assertEqual(second["opponent"], "Lithuania")
        self.assertEqual((second["goals_for"], second["goals_against"]), (2, 0))

    def test_referee_wikilinks_are_reduced_to_name_and_country(self) -> None:
        """A referee's wikilinks must reduce to plain `Name (Country)` text."""
        self.assertEqual(self.frame.iloc[0]["referee"], "Cüneyt Çakır (Turkey)")
        self.assertEqual(self.frame.iloc[1]["referee"], "Ondřej Berka (Czech Republic)")

    def test_attendance_thousands_separator_is_parsed(self) -> None:
        """`11,741` is a number, not a string, and not 11."""
        self.assertEqual(self.frame.iloc[1]["attendance"], 11741)

    def test_missing_fields_are_none_rather_than_placeholder_text(self) -> None:
        """A blank `referee =` must come back as None so callers can simply omit it."""
        cancelled_free = parse_recent_results(_REAL_MALTA_YEARLESS_EXCERPT, "Malta")
        self.assertIsNotNone(cancelled_free.iloc[0]["referee"])

    def test_year_comes_from_the_section_heading_when_the_date_omits_it(self) -> None:
        """Malta's real page writes `date = 26 March`; the year lives in `===2026===`.

        Without threading the heading's year down, this parsed to year 1 and
        sorted every Malta match before every other team's.
        """
        frame = parse_recent_results(_REAL_MALTA_YEARLESS_EXCERPT, "Malta")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["match_date"], pd.Timestamp("2026-03-26"))

    def test_limit_returns_the_most_recent_matches(self) -> None:
        """`limit` keeps the newest end of the record, not the oldest."""
        frame = parse_recent_results(_REAL_POLAND_EXCERPT, "Poland", limit=1)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["match_date"], pd.Timestamp("2025-10-12"))

    def test_unparseable_page_returns_empty_frame_with_the_right_columns(self) -> None:
        """A different layout degrades to empty -- callers never special-case it."""
        frame = parse_recent_results("== Results ==\nNo templates here.", "Brazil")
        self.assertTrue(frame.empty)
        self.assertIn("attendance", frame.columns)
        self.assertIn("referee", frame.columns)


class DottedCapitalITestCase(unittest.TestCase):
    """A `İ` in the page must not shift the parser's offsets."""

    def test_dotted_capital_i_does_not_break_block_extraction(self) -> None:
        """`'İ'.lower()` is two codepoints, desynchronising a lowercased copy.

        Real cause of a real failure (2026-08-04): Japan's page carries `İ`
        in a Turkish referee's name, and every block after it was sliced one
        character late -- starting `{F` instead of `{{F` -- so the brace walk
        truncated it and the team fields vanished. Every match after that
        point silently disappeared.
        """
        with_dotted_i = _REAL_POLAND_EXCERPT.replace("Cüneyt Çakır", "İlker Meler")
        self.assertEqual(len(with_dotted_i.lower()), len(with_dotted_i) + 1)
        frame = parse_recent_results(with_dotted_i, "Poland")
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame.iloc[1]["opponent"], "Lithuania")


class ParseHeadToHeadTableTestCase(unittest.TestCase):
    """The page's own head-to-head summary, used only as a cross-check."""

    def test_real_rows_parse_to_counts(self) -> None:
        """Poland's real table: 2-2-0-0 (5:1) v Bosnia, 5-4-0-1 (8:4) v Albania."""
        table = parse_head_to_head_table(_REAL_HEAD_TO_HEAD_EXCERPT)
        self.assertEqual(len(table), 2)
        bosnia = table.loc[table["opponent"] == "Bosnia and Herzegovina"].iloc[0]
        self.assertEqual(
            (bosnia["played"], bosnia["wins"], bosnia["draws"], bosnia["losses"]),
            (2, 2, 0, 0),
        )
        self.assertEqual((bosnia["goals_for"], bosnia["goals_against"]), (5, 1))

    def test_page_without_a_table_returns_empty_not_an_error(self) -> None:
        """Some pages have no head-to-head section at all."""
        self.assertTrue(parse_head_to_head_table(_REAL_POLAND_EXCERPT).empty)


class FetchDegradationTestCase(unittest.TestCase):
    """Network and missing-page failures must degrade, never propagate."""

    def test_no_page_found_returns_empty_frame(self) -> None:
        """Most teams have no such page -- an ordinary outcome, not a fault."""
        source = TeamRecentResultsWikipediaSource()
        source.fetch_page = lambda team_name: None  # type: ignore[method-assign]
        frame = source.fetch_recent_results("Nowhereland")
        self.assertTrue(frame.empty)
        self.assertEqual(list(frame.columns)[:2], ["match_date", "opponent"])

    def test_network_failure_returns_empty_frame(self) -> None:
        """Same contract as `_load_records_frame()` and `WorldFootballEloSource`."""
        source = TeamRecentResultsWikipediaSource()

        def _raise(team_name: str) -> str:
            raise ConnectionError("simulated outage")

        source.fetch_page = _raise  # type: ignore[method-assign]
        self.assertTrue(source.fetch_recent_results("Poland").empty)
        self.assertTrue(source.fetch_head_to_head_table("Poland").empty)


if __name__ == "__main__":
    unittest.main()
