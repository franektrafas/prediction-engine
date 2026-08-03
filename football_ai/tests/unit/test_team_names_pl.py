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

    def test_world_cup_participants_outside_uefa_are_translated(self) -> None:
        """Non-UEFA World Cup sides got real Polish forms on 2026-08-03.

        Before that the table was UEFA-only, so the 2026 World Cup final
        rendered as "Argentina zanotowało" -- raw English name plus neuter
        agreement -- on the app's most prominent screen.
        """
        self.assertEqual(nominative_pl("Argentina"), "Argentyna")
        self.assertEqual(accusative_pl("Argentina"), "Argentynę")
        self.assertEqual(gender_pl("Argentina"), "f")
        # Pluralia tantum, like Germany: "Stany Zjednoczone wygrały".
        self.assertEqual(nominative_pl("United States"), "Stany Zjednoczone")
        self.assertEqual(dative_pl("United States"), "Stanom Zjednoczonym")
        self.assertEqual(gender_pl("United States"), "pl")
        # Neuter, and indeclinable in the accusative: "Maroko wygrało".
        self.assertEqual(nominative_pl("Morocco"), "Maroko")
        self.assertEqual(dative_pl("Morocco"), "Maroku")
        self.assertEqual(gender_pl("Morocco"), "n")

    def test_untranslated_team_falls_back_to_the_raw_english_name(self) -> None:
        """A team outside the table (~190 of them remain) must degrade, not crash.

        Uses a genuinely absent side rather than a major nation: this
        assertion used to be written against "Brazil", which quietly stopped
        testing the fallback the moment the World Cup sides were added.
        """
        self.assertEqual(nominative_pl("Yoruba Nation"), "Yoruba Nation")
        self.assertEqual(dative_pl("Yoruba Nation"), "Yoruba Nation")
        self.assertEqual(gender_pl("Yoruba Nation"), "n")


if __name__ == "__main__":
    unittest.main()
