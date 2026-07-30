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

    def test_west_germany_and_east_germany_resolve_to_distinct_entities(self) -> None:
        """PLAN_SILNIKA.md's own completion criterion for Krok 1.

        A 1974 West Germany match and a 1974 East Germany ("German DR")
        match must land in the correct, *distinct* entity -- West Germany
        aliases forward to today's "Germany" (a continuous federation,
        martj42/international_results' own results.csv already folds it in
        this way), but East Germany does not: it has no clean modern
        successor, so it must stay its own canonical name, never merged
        into "Germany".
        """
        self.assertEqual(canonicalize_team_name("West Germany"), "Germany")
        self.assertEqual(canonicalize_team_name("German DR"), "German DR")
        self.assertNotEqual(canonicalize_team_name("German DR"), canonicalize_team_name("Germany"))

    def test_yugoslavia_and_soviet_union_era_aliases(self) -> None:
        """Soviet Union resolves forward (continuous federation); Yugoslavia does not (no clean successor)."""
        self.assertEqual(canonicalize_team_name("Soviet Union"), "Russia")
        self.assertEqual(canonicalize_team_name("Yugoslavia"), "Yugoslavia")
        self.assertEqual(canonicalize_team_name("FR Yugoslavia"), "Serbia")


if __name__ == "__main__":
    unittest.main()
