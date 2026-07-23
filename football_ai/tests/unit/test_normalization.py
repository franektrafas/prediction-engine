"""Tests for shared text/team-name normalization."""

from __future__ import annotations

import unittest

from football_ai.data.normalization import canonicalize_team_name, normalize_text


class NormalizeTextTestCase(unittest.TestCase):
    """Validate diacritic-stripping and whitespace/case normalization."""

    def test_strips_diacritics(self) -> None:
        """An accented and an unaccented spelling of the same text must normalize identically."""
        self.assertEqual(normalize_text("Côte d'Ivoire"), normalize_text("Cote d'ivoire"))

    def test_collapses_whitespace_and_lowercases(self) -> None:
        """Extra whitespace and casing shouldn't affect the normalized form."""
        self.assertEqual(normalize_text("  Real   Madrid  "), "real madrid")

    def test_none_returns_empty_string(self) -> None:
        """A missing value should normalize to an empty string, not raise."""
        self.assertEqual(normalize_text(None), "")


class CanonicalizeTeamNameTestCase(unittest.TestCase):
    """Validate alias resolution survives a diacritic-only spelling difference."""

    def test_accented_source_spelling_resolves_to_the_same_alias(self) -> None:
        """StatsBomb's accented 'Côte d'Ivoire' must resolve like the plain-ASCII alias key."""
        self.assertEqual(canonicalize_team_name("Côte d'Ivoire"), "Ivory Coast")
        self.assertEqual(canonicalize_team_name("Cote d'Ivoire"), "Ivory Coast")


if __name__ == "__main__":
    unittest.main()
