"""Tests for the UEFA Nations League Wikipedia importer.

The fixture wikitext below is a real excerpt (2026-07-30) from
`2026–27 UEFA Nations League B`'s Group 4 (Poland's real group) -- not
synthetic, trimmed to 3 of the real 12 matches to keep the fixture readable.
"""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from football_ai.config.fifa_country_codes import resolve_fifa_code
from football_ai.scrapers.sources.uefa_nations_league import (
    UefaNationsLeagueSource,
    _extract_group_section,
    parse_football_box_matches,
)


# Real excerpt, 2026-07-30, from `2026–27 UEFA Nations League B` (Group 4 == Poland's real group).
_REAL_GROUP_4_EXCERPT = """===Group 4===
{{2026–27 UEFA Nations League B tables|Group 4|show_matches=yes}}

{{Football box
|date={{Start date|2026|9|25|df=y}}
|time=20:45
|team1={{fb-rt|POL}}
|score=
|report=https://www.uefa.com/uefanationsleague/match/2048013/
|team2={{fb|BIH}}
|goals1=
|goals2=
|stadium=[[Stadion Narodowy]], [[Warsaw]]
|attendance=
|referee=
}}
{{Football box
|date={{Start date|2026|9|25|df=y}}
|time=20:45
|team1={{fb-rt|SWE}}
|score=
|report=https://www.uefa.com/uefanationsleague/match/2048015/
|team2={{fb|ROU}}
|goals1=
|goals2=
|stadium=[[Nationalarenan]], [[Solna]]
|attendance=
|referee=
}}
----
{{Football box
| date = {{Start date|2026|9|28|df=y}}
| time = 20:45 {{small|(21:45 [[UTC+03:00|UTC+3]])}}
| team1 = {{fb-rt|ROU}}
| score =
| report = https://www.uefa.com/uefanationsleague/match/2047984/
| team2 = {{fb|BIH}}
| goals1 =
| goals2 =
| stadium = [[Arena Națională]], [[Bucharest]]
| attendance =
| referee =
}}

===Overall ranking===
Following the league phase, the 16 League B teams..."""

# A played-match variant (hand-constructed to exercise the goals1/goals2-populated
# path, which the real page can't demonstrate yet -- no 2026/27 match has been
# played as of when this was written) with a mid-value {{refn}} to confirm the
# balanced-brace block extractor doesn't truncate on a nested template.
_PLAYED_MATCH_EXCERPT = """{{Football box
|date={{Start date|2026|9|25|df=y}}
|time=20:45
|team1={{fb-rt|POL}}
|score=2–1
|report=https://www.uefa.com/uefanationsleague/match/2048013/
|team2={{fb|BIH}}
|goals1=2
|goals2=1
|stadium=[[Stadion Narodowy]], [[Warsaw]]{{refn|group=note|Some note.}}
|attendance=45000
|referee=
}}"""


class ParseFootballBoxMatchesTestCase(unittest.TestCase):
    """Validate the wikitext template parser against real data."""

    def test_parses_all_matches_in_the_real_excerpt(self) -> None:
        """3 real matches in the fixture -- all must be found, none dropped, none duplicated."""
        matches = parse_football_box_matches(_REAL_GROUP_4_EXCERPT)
        self.assertEqual(len(matches), 3)

    def test_first_real_match_fields_are_correct(self) -> None:
        """Poland v Bosnia and Herzegovina, 2026-09-25, unplayed -- cross-checked by hand against the real page."""
        matches = parse_football_box_matches(_REAL_GROUP_4_EXCERPT)
        first = matches[0]
        self.assertEqual(first["team1_code"], "POL")
        self.assertEqual(first["team2_code"], "BIH")
        self.assertEqual(first["match_date"].isoformat(), "2026-09-25")
        self.assertIsNone(first["goals1"])
        self.assertIsNone(first["goals2"])
        self.assertIn("Stadion Narodowy", first["stadium"])
        self.assertNotIn("[[", first["stadium"])  # wikilink markup stripped

    def test_handles_the_spaced_out_field_formatting_variant(self) -> None:
        """The real page mixes `|key=value` and `| key = value` styles -- both must parse identically."""
        matches = parse_football_box_matches(_REAL_GROUP_4_EXCERPT)
        third = matches[2]  # the "| date = ..." spaced-out variant
        self.assertEqual(third["team1_code"], "ROU")
        self.assertEqual(third["team2_code"], "BIH")
        self.assertEqual(third["match_date"].isoformat(), "2026-09-28")

    def test_nested_template_in_a_field_value_does_not_truncate_the_block(self) -> None:
        """A {{refn}} inside |stadium= must not be mistaken for the outer block's closing braces."""
        matches = parse_football_box_matches(_PLAYED_MATCH_EXCERPT)
        self.assertEqual(len(matches), 1)
        match = matches[0]
        self.assertEqual(match["goals1"], 2)
        self.assertEqual(match["goals2"], 1)
        self.assertIn("Stadion Narodowy", match["stadium"])

    def test_empty_wikitext_returns_no_matches(self) -> None:
        """No Football box templates present -- an empty list, not an error."""
        self.assertEqual(parse_football_box_matches("Just some prose, no matches here."), [])


class ExtractGroupSectionTestCase(unittest.TestCase):
    """Validate slicing one `===Group N===` section out of a page's full wikitext."""

    def test_extracts_only_the_requested_group(self) -> None:
        """Group 4's section must stop before '===Overall ranking===' picks up unrelated content."""
        section = _extract_group_section(_REAL_GROUP_4_EXCERPT, group_number=4)
        self.assertIsNotNone(section)
        self.assertIn("POL", section)
        self.assertNotIn("Overall ranking", section)

    def test_unknown_group_number_returns_none(self) -> None:
        """A group that doesn't exist in the page shouldn't raise -- callers check for None."""
        self.assertIsNone(_extract_group_section(_REAL_GROUP_4_EXCERPT, group_number=99))


class FifaCodeResolutionTestCase(unittest.TestCase):
    """Validate the FIFA-code-to-canonical-name mapping used by this module."""

    def test_real_group_4_codes_resolve_correctly(self) -> None:
        """POL/BIH/SWE/ROU -- the four real teams in Poland's actual group."""
        self.assertEqual(resolve_fifa_code("POL"), "Poland")
        self.assertEqual(resolve_fifa_code("BIH"), "Bosnia and Herzegovina")
        self.assertEqual(resolve_fifa_code("SWE"), "Sweden")
        self.assertEqual(resolve_fifa_code("ROU"), "Romania")

    def test_unmapped_code_raises_rather_than_guessing(self) -> None:
        """An unrecognized code must fail loudly, not silently become a garbage team name."""
        with self.assertRaises(KeyError):
            resolve_fifa_code("XXX")


class UefaNationsLeagueSourceTestCase(unittest.TestCase):
    """Validate the orchestration methods with the network mocked out."""

    def setUp(self) -> None:
        """Build a fresh source for each test."""
        self.source = UefaNationsLeagueSource()

    def test_fetch_group_matches_builds_the_correct_project_schema_frame(self) -> None:
        """End-to-end (mocked fetch): real excerpt in, standard wide-match columns out."""
        full_page = f"prose before\n\n{_REAL_GROUP_4_EXCERPT}"
        with mock.patch.object(self.source, "fetch_text_with_headers", return_value=full_page):
            frame = self.source.fetch_group_matches("B", 4)
        self.assertEqual(len(frame), 3)
        self.assertListEqual(
            list(frame.columns),
            ["match_date", "home_team_name", "away_team_name", "home_score", "away_score", "stadium"],
        )
        self.assertIn("Poland", frame["home_team_name"].to_list())
        self.assertTrue(pd.isna(frame.iloc[0]["home_score"]))

    def test_find_team_group_locates_poland_in_division_b_group_4(self) -> None:
        """Mocked to only return real data for League B -- confirms the search actually inspects each division."""

        def fetch_side_effect(url: str, headers: dict, params: dict) -> str:
            if params.get("title") == "2026–27 UEFA Nations League B":
                return _REAL_GROUP_4_EXCERPT
            return "===Group 1===\n{{Football box\n|date={{Start date|2026|9|1|df=y}}\n|team1={{fb-rt|FRA}}\n|team2={{fb|GER}}\n|goals1=\n|goals2=\n}}"

        with mock.patch.object(self.source, "fetch_text_with_headers", side_effect=fetch_side_effect):
            division, group_number = self.source.find_team_group("Poland")
        self.assertEqual(division, "B")
        self.assertEqual(group_number, 4)

    def test_find_team_group_raises_for_a_team_not_in_any_mocked_group(self) -> None:
        """A team absent from every division's data should fail loudly, not return a wrong guess."""
        with mock.patch.object(self.source, "fetch_text_with_headers", return_value="no matches here"):
            with self.assertRaises(ValueError):
                self.source.find_team_group("Poland")


if __name__ == "__main__":
    unittest.main()
