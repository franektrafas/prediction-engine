"""Tests for the Transfermarkt national-team squad scraper."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest import mock

from football_ai.scrapers.sources.transfermarkt import (
    TransfermarktSource,
    _parse_age,
    _parse_market_value,
)


# A trimmed, structurally-accurate copy of the real squad table markup
# (verified against Transfermarkt's live Poland squad page during planning):
# each player's photo/name/position is nested in an `inline-table`, the club
# name lives only in an `<a title="...">` attribute (not visible text), and
# market value is the last cell's link text.
_SQUAD_TABLE_HTML = """
<table class="items">
<tbody>
<tr class="odd">
<td class="zentriert rueckennummer bg_Torwart" title="Goalkeeper"><div class="rn_nummer">1</div></td>
<td class="">
<table class="inline-table">
<tr>
<td rowspan="2"><img alt="Kamil Grabara" title="Kamil Grabara"/></td>
<td class="hauptlink"><a href="/kamil-grabara/profil/spieler/346551">Kamil Grabara</a></td>
</tr>
<tr><td>Goalkeeper</td></tr>
</table>
</td>
<td class="zentriert">08/01/1999 (27)</td>
<td class="zentriert">
<a href="/vfl-wolfsburg/startseite/verein/82" title="VfL Wolfsburg">
<img alt="VfL Wolfsburg" title="VfL Wolfsburg"/>
</a>
</td>
<td class="rechts hauptlink"><a href="/kamil-grabara/marktwertverlauf/spieler/346551">&euro;12.00m</a></td>
</tr>
<tr class="even">
<td class="zentriert rueckennummer bg_Abwehr" title="Centre-Back"><div class="rn_nummer">4</div></td>
<td class="">
<table class="inline-table">
<tr>
<td rowspan="2"><img alt="Jan Bednarek" title="Jan Bednarek"/></td>
<td class="hauptlink"><a href="/jan-bednarek/profil/spieler/337614">Jan Bednarek</a></td>
</tr>
<tr><td>Centre-Back</td></tr>
</table>
</td>
<td class="zentriert">12/04/1996 (30)</td>
<td class="zentriert">
<a href="/southampton-fc/startseite/verein/180" title="Southampton FC">
<img alt="Southampton FC" title="Southampton FC"/>
</a>
</td>
<td class="rechts hauptlink"><a href="/jan-bednarek/marktwertverlauf/spieler/337614">&euro;500k</a></td>
</tr>
</tbody>
</table>
"""

_SEARCH_RESULTS_HTML = """
<div class="box">
<a href="/polen/startseite/verein/3442">Poland</a>
<a href="/polen-u21/startseite/verein/9324">Poland U21</a>
<a href="/rockspur/startseite/verein/101973">Rockspur FC</a>
</div>
"""


class MarketValueAndAgeParsingTestCase(unittest.TestCase):
    """Validate the small text parsers in isolation."""

    def test_parses_millions(self) -> None:
        """A "€12.00m" value should parse to 12,000,000 euros."""
        self.assertAlmostEqual(_parse_market_value("€12.00m"), 12_000_000.0)

    def test_parses_thousands(self) -> None:
        """A "€500k" value should parse to 500,000 euros."""
        self.assertAlmostEqual(_parse_market_value("€500k"), 500_000.0)

    def test_parses_missing_value(self) -> None:
        """A "-" (no market value) should parse to None."""
        self.assertIsNone(_parse_market_value("-"))

    def test_parses_empty_and_none(self) -> None:
        """Empty string and None should both parse to None."""
        self.assertIsNone(_parse_market_value(""))
        self.assertIsNone(_parse_market_value(None))

    def test_parses_age_suffix(self) -> None:
        """A "(27)" suffix should parse to 27.0."""
        self.assertEqual(_parse_age("08/01/1999 (27)"), 27.0)

    def test_parses_age_missing(self) -> None:
        """Text without a parenthetical age should parse to None."""
        self.assertIsNone(_parse_age("08/01/1999"))
        self.assertIsNone(_parse_age(None))


class TransfermarktSquadParsingTestCase(unittest.TestCase):
    """Validate the squad-table HTML parser against realistic markup."""

    def setUp(self) -> None:
        """Create a fresh source instance for each test."""
        self.source = TransfermarktSource()

    def test_parse_squad_table_extracts_players(self) -> None:
        """Player name/id, position, age, club (title attr), and market value should all parse correctly."""
        frame = self.source._parse_squad_table(_SQUAD_TABLE_HTML, team_id=3442, team_name="Poland")
        self.assertEqual(len(frame), 2)

        grabara = frame.loc[frame["player_name"] == "Kamil Grabara"].iloc[0]
        self.assertEqual(grabara["player_external_id"], 346551)
        self.assertEqual(grabara["position"], "Goalkeeper")
        self.assertEqual(grabara["age"], 27.0)
        self.assertEqual(grabara["club_name"], "VfL Wolfsburg")
        self.assertAlmostEqual(grabara["market_value_eur"], 12_000_000.0)
        self.assertEqual(grabara["team_name"], "Poland")
        self.assertEqual(grabara["team_external_id"], 3442)

        bednarek = frame.loc[frame["player_name"] == "Jan Bednarek"].iloc[0]
        self.assertEqual(bednarek["position"], "Centre-Back")
        self.assertEqual(bednarek["club_name"], "Southampton FC")
        self.assertAlmostEqual(bednarek["market_value_eur"], 500_000.0)

    def test_parse_squad_table_handles_missing_table(self) -> None:
        """HTML without a squad table should return an empty frame, not raise."""
        frame = self.source._parse_squad_table("<html><body>No squad here</body></html>", team_id=1, team_name="Nowhere")
        self.assertTrue(frame.empty)


class TransfermarktSearchTestCase(unittest.TestCase):
    """Validate search-result parsing and age-group filtering."""

    def setUp(self) -> None:
        """Create a fresh source instance for each test."""
        self.source = TransfermarktSource()

    def test_search_teams_flags_age_group_variants(self) -> None:
        """A senior team and its U21 variant should both parse, with the variant flagged."""
        with mock.patch.object(self.source, "fetch_text_with_headers", return_value=_SEARCH_RESULTS_HTML):
            results = self.source.search_teams("Poland")

        by_slug = {result["slug"]: result for result in results}
        self.assertIn("polen", by_slug)
        self.assertFalse(by_slug["polen"]["is_age_group_variant"])
        self.assertEqual(by_slug["polen"]["team_id"], 3442)
        self.assertIn("polen-u21", by_slug)
        self.assertTrue(by_slug["polen-u21"]["is_age_group_variant"])


class TransfermarktRefreshPlayerReferenceTestCase(unittest.TestCase):
    """Validate the batch refresh's resilience, mirroring FootballApiSource's tests."""

    def setUp(self) -> None:
        """Create a fresh source instance for each test."""
        self.source = TransfermarktSource()

    def test_skips_a_failing_team(self) -> None:
        """One team's fetch failing should not stop the others from being stored."""
        good_squad = self.source._parse_squad_table(_SQUAD_TABLE_HTML, team_id=3442, team_name="Poland")

        def fetch_side_effect(team_id, team_slug, team_name):
            if team_name == "France":
                raise RuntimeError("blocked")
            return good_squad

        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "player_reference.csv"
            with mock.patch.object(self.source, "fetch_national_team_squad", side_effect=fetch_side_effect):
                result = self.source.refresh_player_reference(
                    team_ids=[(3442, "polen", "Poland"), (3377, "frankreich", "France")],
                    player_reference_path=str(reference_path),
                    request_delay_seconds=0,
                )
        self.assertEqual(result.added_players, 2)

    def test_handles_every_team_failing(self) -> None:
        """A refresh where every team fails should degrade gracefully, not raise."""
        with tempfile.TemporaryDirectory() as temp_dir:
            reference_path = Path(temp_dir) / "player_reference.csv"
            with mock.patch.object(
                self.source, "fetch_national_team_squad", side_effect=RuntimeError("blocked")
            ):
                result = self.source.refresh_player_reference(
                    team_ids=[(3442, "polen", "Poland")],
                    player_reference_path=str(reference_path),
                    request_delay_seconds=0,
                )
        self.assertEqual(result.added_players, 0)
        self.assertEqual(len(result.frame), 0)


if __name__ == "__main__":
    unittest.main()
