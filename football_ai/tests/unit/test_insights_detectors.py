"""Tests for football_ai/insights/detectors.py -- ranking, not dumping everything."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from football_ai.insights.detectors import facts_for_match, facts_for_team


_FULL_HISTORY_PATH = Path("football_ai/data/processed/international_results_full_history.parquet")


@unittest.skipUnless(_FULL_HISTORY_PATH.exists(), "requires the regenerated full-history parquet")
class DetectorsTestCase(unittest.TestCase):
    """Validate aggregation and ranking against real data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real records_frame once for the whole test class."""
        cls.records_frame = pd.read_parquet(_FULL_HISTORY_PATH)

    def test_facts_for_team_returns_multiple_fact_types(self) -> None:
        """A real, well-established team should produce facts from more than one submodule."""
        facts = facts_for_team(self.records_frame, "Poland")
        fact_types = {fact.fact_type for fact in facts}
        self.assertGreater(len(facts), 1)
        self.assertGreater(len(fact_types), 1)

    def test_facts_for_match_is_sorted_by_weight_descending(self) -> None:
        """The whole point of a detector layer -- most notable first, not insertion order."""
        facts = facts_for_match(self.records_frame, "Poland", "Germany", top_n=None)
        weights = [fact.weight for fact in facts]
        self.assertEqual(weights, sorted(weights, reverse=True))

    def test_facts_for_match_respects_top_n(self) -> None:
        """Must not just return everything -- PLAN_SILNIKA.md's explicit completion criterion."""
        facts = facts_for_match(self.records_frame, "Poland", "Germany", top_n=3)
        self.assertLessEqual(len(facts), 3)

    def test_poland_san_marino_surfaces_the_real_biggest_win_fact(self) -> None:
        """Poland's real 10-0 vs San Marino (weight 1.0) must be in the ranked output somewhere."""
        facts = facts_for_match(self.records_frame, "Poland", "San Marino", top_n=None)
        biggest_win_facts = [fact for fact in facts if fact.fact_type == "biggest_win"]
        self.assertTrue(any(fact.subject_team == "Poland" for fact in biggest_win_facts))

    def test_facts_for_match_respects_as_of_date_cutoff(self) -> None:
        """A preview generated before Poland's 2023 win over Germany must not know about it yet."""
        before = facts_for_match(
            self.records_frame, "Poland", "Germany", as_of_date=pd.Timestamp("2020-01-01", tz="UTC"), top_n=None
        )
        h2h_facts_before = [fact for fact in before if fact.data.get("win_year") == 2023]
        self.assertEqual(h2h_facts_before, [])


if __name__ == "__main__":
    unittest.main()
