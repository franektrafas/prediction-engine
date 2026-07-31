"""Tests for football_ai/narrative/team_names_pl.py."""

from __future__ import annotations

import unittest

from football_ai.narrative.team_names_pl import accusative_pl, dative_pl, gender_pl, nominative_pl


class TeamNamesPlTestCase(unittest.TestCase):
    """Spot-check a few known-tricky forms, plus the fallback for an untranslated team."""

    def test_poland_feminine_forms(self) -> None:
        self.assertEqual(nominative_pl("Poland"), "Polska")
        self.assertEqual(dative_pl("Poland"), "Polsce")
        self.assertEqual(accusative_pl("Poland"), "Polskę")
        self.assertEqual(gender_pl("Poland"), "f")

    def test_germany_is_grammatically_plural(self) -> None:
        """"Niemcy" is a pluralia tantum -- nominative and accusative happen to be identical."""
        self.assertEqual(nominative_pl("Germany"), "Niemcy")
        self.assertEqual(accusative_pl("Germany"), "Niemcy")
        self.assertEqual(dative_pl("Germany"), "Niemcom")
        self.assertEqual(gender_pl("Germany"), "pl")

    def test_israel_is_masculine(self) -> None:
        self.assertEqual(nominative_pl("Israel"), "Izrael")
        self.assertEqual(dative_pl("Israel"), "Izraelowi")
        self.assertEqual(gender_pl("Israel"), "m")

    def test_san_marino_is_indeclinable(self) -> None:
        """A foreign name ending in -o that Polish doesn't decline -- same form in every case."""
        self.assertEqual(nominative_pl("San Marino"), "San Marino")
        self.assertEqual(dative_pl("San Marino"), "San Marino")
        self.assertEqual(accusative_pl("San Marino"), "San Marino")

    def test_untranslated_team_falls_back_to_the_raw_english_name(self) -> None:
        """A non-UEFA opponent (this table's real, documented scope gap) must degrade, not crash."""
        self.assertEqual(nominative_pl("Brazil"), "Brazil")
        self.assertEqual(dative_pl("Brazil"), "Brazil")
        self.assertEqual(gender_pl("Brazil"), "n")


if __name__ == "__main__":
    unittest.main()
