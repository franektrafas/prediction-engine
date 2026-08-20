"""Tests for the historical tournament squads importer (footballsquads cache)."""

from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from football_ai.scrapers.sources.footballsquads import (
    FootballSquadsSource,
    _continuity_ratio,
    _parse_birth_date,
    _parse_height_meters,
    _parse_squad_text,
    _position_bucket,
    _squad_summary,
    _tournament_start_date,
)

# Real excerpts (trimmed to 3 players) fetched 2026-08-20 from
# raw.githubusercontent.com/footballcsv/cache.footballsquads/master/national/...
_POLAND_2022_EXCERPT = """=  Poland - FIFA World Cup 2022

Number, Name, Nat, Pos, Height, Weight, Date of Birth, Birth Place, Current Club
1, Wojciech Szczęsny, POL, G, 1.95, 90, 18-04-90, Warsaw, Juventus
2, Matty Cash, POL, D, 1.85, 64, 07-08-97, Slough, Aston Villa
9, Robert Lewandowski, POL, F, 1.81, 79, 21-08-88, Warsaw, Barcelona
"""

_BRAZIL_1950_EXCERPT = """=  Brazil - World Cup 1950

Number, Name, Pos, Height, Weight, Date of Birth, Birth Place, Current Club
, Barbosa, G, 5'09", 11 11, 27-03-21, Campinas, Vasco da Gama
, Castilho, G, , , 27-11-27, Rio de Janeiro, Fluminense
, Bigode, D, 5'06", 11 05, 04-04-22, , Flamengo
"""


class ParseHeightMetersTestCase(unittest.TestCase):
    """Height parsing across both real eras' formats."""

    def test_metric_format(self) -> None:
        """A decimal-metres cell ("1.95") should parse as-is."""
        self.assertEqual(_parse_height_meters("1.95"), 1.95)

    def test_imperial_format(self) -> None:
        """A feet/inches cell (used pre-~1994) should convert to metres."""
        self.assertAlmostEqual(_parse_height_meters("5'09\""), 1.75, places=2)

    def test_blank_returns_none(self) -> None:
        """A blank cell must stay unknown, not become a fabricated 0."""
        self.assertIsNone(_parse_height_meters(""))

    def test_garbage_returns_none_not_a_guess(self) -> None:
        """Unparseable text must return None, never a guessed number."""
        self.assertIsNone(_parse_height_meters("tall"))


class ParseBirthDateTestCase(unittest.TestCase):
    """The century-disambiguation logic is the whole point of this parser."""

    def test_old_player_resolves_to_1900s(self) -> None:
        """Real case: Barbosa, born "27-03-21", at the 1950 World Cup -> 1921."""
        result = _parse_birth_date("27-03-21", tournament_year=1950)
        self.assertEqual(result, pd.Timestamp(year=1921, month=3, day=27, tz="UTC"))

    def test_young_player_resolves_to_2000s(self) -> None:
        """A player born "03" at a 2024 tournament must resolve to 2003, not 1903."""
        result = _parse_birth_date("15-06-03", tournament_year=2024)
        self.assertEqual(result, pd.Timestamp(year=2003, month=6, day=15, tz="UTC"))

    def test_blank_returns_none(self) -> None:
        """An empty cell must stay unknown."""
        self.assertIsNone(_parse_birth_date("", tournament_year=2022))

    def test_malformed_value_returns_none(self) -> None:
        """Text that isn't a DD-MM-YY date must return None."""
        self.assertIsNone(_parse_birth_date("not-a-date", tournament_year=2022))

    def test_implausible_age_in_every_century_returns_none(self) -> None:
        """"YY"=60 at a 1950 tournament: neither 1960 nor 2060 is a plausible age."""
        self.assertIsNone(_parse_birth_date("01-01-60", tournament_year=1950))


class ParseSquadTextTestCase(unittest.TestCase):
    """Golden tests against real (trimmed) source text."""

    def test_modern_squad_with_nat_column(self) -> None:
        """A 2022-era file has a "Nat" column and metric heights."""
        frame = _parse_squad_text(_POLAND_2022_EXCERPT, "worldcup", 2022, "poland")
        self.assertEqual(len(frame), 3)
        self.assertEqual(frame["team_name"].iloc[0], "Poland")
        self.assertIn("Nat", frame.columns)
        self.assertListEqual(
            frame["player_name"].to_list(),
            ["Wojciech Szczęsny", "Matty Cash", "Robert Lewandowski"],
        )
        self.assertListEqual(frame["position"].to_list(), ["G", "D", "F"])
        self.assertEqual(frame["height_m"].iloc[0], 1.95)
        expected_birth_date = pd.Timestamp(year=1990, month=4, day=18, tz="UTC")
        self.assertEqual(frame["birth_date"].iloc[0], expected_birth_date)

    def test_1950s_squad_without_nat_column(self) -> None:
        """The real 1950 files have no "Nat" column -- parse by header, not position."""
        frame = _parse_squad_text(_BRAZIL_1950_EXCERPT, "worldcup", 1950, "brazil")
        self.assertEqual(len(frame), 3)
        self.assertNotIn("Nat", frame.columns)
        self.assertEqual(frame["team_name"].iloc[0], "Brazil")
        self.assertListEqual(
            frame["player_name"].to_list(), ["Barbosa", "Castilho", "Bigode"]
        )
        # Castilho's height/weight are blank in the real source -- must stay NaN, not 0.
        self.assertTrue(pd.isna(frame["height_m"].iloc[1]))
        self.assertFalse(pd.isna(frame["height_m"].iloc[0]))

    def test_unparseable_text_returns_empty_frame_not_an_exception(self) -> None:
        """Malformed input degrades to an empty frame, not a crash."""
        frame = _parse_squad_text(
            "not a squad file at all", "worldcup", 2022, "nowhere"
        )
        self.assertTrue(frame.empty)


class PositionBucketTestCase(unittest.TestCase):
    """Position-code bucketing."""

    def test_known_codes(self) -> None:
        """The four standard single-letter codes map onto their buckets."""
        self.assertEqual(_position_bucket("G"), "goalkeepers")
        self.assertEqual(_position_bucket("D"), "defenders")
        self.assertEqual(_position_bucket("M"), "midfielders")
        self.assertEqual(_position_bucket("F"), "forwards")

    def test_unknown_code_returns_none(self) -> None:
        """An unrecognized code is left uncounted rather than guessed."""
        self.assertIsNone(_position_bucket("XYZ"))


class SquadSummaryTestCase(unittest.TestCase):
    """The aggregation step, including honest missing-data handling."""

    def test_poland_2022_summary_matches_manual_sanity_check(self) -> None:
        """Manually verified ~27yo average age for this real squad (squads plan doc)."""
        frame = _parse_squad_text(_POLAND_2022_EXCERPT, "worldcup", 2022, "poland")
        tournament_start = pd.Timestamp(year=2022, month=11, day=20, tz="UTC")
        summary = _squad_summary(frame, "worldcup", 2022, "poland", tournament_start)
        self.assertEqual(summary["squad_size"], 3)
        self.assertEqual(summary["age_known_count"], 3)
        self.assertGreater(summary["average_age_years"], 25)
        self.assertLess(summary["average_age_years"], 35)
        self.assertEqual(summary["goalkeepers"], 1)
        self.assertEqual(summary["defenders"], 1)
        self.assertEqual(summary["forwards"], 1)

    def test_missing_data_stays_none_not_zero(self) -> None:
        """A sparse 1950s squad must report partial coverage honestly."""
        frame = _parse_squad_text(_BRAZIL_1950_EXCERPT, "worldcup", 1950, "brazil")
        tournament_start = pd.Timestamp(year=1950, month=6, day=24, tz="UTC")
        summary = _squad_summary(frame, "worldcup", 1950, "brazil", tournament_start)
        self.assertEqual(summary["squad_size"], 3)
        self.assertEqual(summary["height_known_count"], 2)  # Castilho's height is blank
        self.assertIsNotNone(summary["average_height_m"])

    def test_no_tournament_start_date_leaves_age_fields_none(self) -> None:
        """Without a resolvable tournament start date, age must stay unknown."""
        frame = _parse_squad_text(_POLAND_2022_EXCERPT, "worldcup", 2022, "poland")
        summary = _squad_summary(frame, "worldcup", 2022, "poland", None)
        self.assertIsNone(summary["average_age_years"])
        self.assertEqual(summary["age_known_count"], 0)


class ContinuityRatioTestCase(unittest.TestCase):
    """Best-effort squad continuity between consecutive tournaments."""

    def test_full_overlap(self) -> None:
        """An identical squad set has 100% continuity."""
        names = {"robert lewandowski", "wojciech szczesny"}
        self.assertEqual(_continuity_ratio(names, names), 1.0)

    def test_partial_overlap(self) -> None:
        """Continuity is the fraction of the CURRENT squad found in the previous one."""
        current = {"a", "b", "c", "d"}
        previous = {"a", "b", "x", "y"}
        self.assertEqual(_continuity_ratio(current, previous), 0.5)

    def test_no_previous_squad_returns_none_not_zero(self) -> None:
        """A team's first tournament in this source has nothing to compare against."""
        self.assertIsNone(_continuity_ratio({"a", "b"}, set()))

    def test_empty_current_squad_returns_zero(self) -> None:
        """An empty current squad has zero continuity, not undefined."""
        self.assertEqual(_continuity_ratio(set(), {"a", "b"}), 0.0)


class TournamentStartDateTestCase(unittest.TestCase):
    """Tournament start dates derived from the already-owned records_frame."""

    def test_looks_up_the_earliest_match_of_that_tournament_year(self) -> None:
        """The earliest FINALS match of that year wins; qualifiers are excluded."""
        records = pd.DataFrame(
            {
                "tournament": [
                    "FIFA World Cup",
                    "FIFA World Cup",
                    "FIFA World Cup qualification",
                ],
                "match_date": pd.to_datetime(
                    ["2022-11-21", "2022-11-20", "2022-01-01"], utc=True
                ),
            }
        )
        result = _tournament_start_date(records, "worldcup", 2022)
        self.assertEqual(result, pd.Timestamp(year=2022, month=11, day=20, tz="UTC"))

    def test_postponed_tournament_resolves_via_the_real_played_year(self) -> None:
        """Euro "2020" was really played in 2021 (COVID) -- branding != match year."""
        records = pd.DataFrame(
            {
                "tournament": ["UEFA Euro"],
                "match_date": pd.to_datetime(["2021-06-11"], utc=True),
            }
        )
        result = _tournament_start_date(records, "eurocham", 2020)
        self.assertEqual(result, pd.Timestamp(year=2021, month=6, day=11, tz="UTC"))

    def test_missing_tournament_returns_none(self) -> None:
        """No matching rows means an unresolvable start date, not a guess."""
        records = pd.DataFrame(
            {"tournament": [], "match_date": pd.to_datetime([], utc=True)}
        )
        self.assertIsNone(_tournament_start_date(records, "worldcup", 1899))


class FootballSquadsSourceTestCase(unittest.TestCase):
    """Source-level behavior, fully offline."""

    def test_list_squad_files_filters_to_known_competitions_and_txt_files(self) -> None:
        """Only national/{known competition}/*/*.txt entries should survive."""
        source = FootballSquadsSource()
        fake_tree = {
            "tree": [
                {"path": "national/worldcup/wc2022/poland.txt"},
                {"path": "national/worldcup/wc2022/README.md"},
                {"path": "national/eurocham/euro2024/germany.txt"},
                {"path": "clubs/england/2022-23/arsenal.txt"},
                {"path": "national/unknown_competition/x2020/team.txt"},
            ]
        }
        with mock.patch.object(source, "fetch_json", return_value=fake_tree):
            refs = source.list_squad_files()
        paths = {ref.path for ref in refs}
        self.assertIn("national/worldcup/wc2022/poland.txt", paths)
        self.assertIn("national/eurocham/euro2024/germany.txt", paths)
        self.assertEqual(len(refs), 2)


if __name__ == "__main__":
    unittest.main()
