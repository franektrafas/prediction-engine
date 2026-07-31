"""Tests for football_ai/narrative/renderer.py, including real-data integration."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.insights import h2h, milestones
from football_ai.insights.detectors import facts_for_match
from football_ai.insights.schemas import Fact
from football_ai.narrative.renderer import UnknownFactTypeError, render_fact, render_facts


class RenderFactErrorHandlingTestCase(unittest.TestCase):
    """An insights/narrative drift (new fact_type, no matching template) must fail loudly."""

    def test_unknown_fact_type_raises(self) -> None:
        fact = Fact(
            fact_type="not_a_real_fact_type",
            subject_team="Poland",
            summary="irrelevant",
            data={},
            weight=0.5,
            provenance=(1,),
        )
        with self.assertRaises(UnknownFactTypeError):
            render_fact(fact)


class RealDataRendererTestCase(unittest.TestCase):
    """PLAN_SILNIKA.md's own Krok 4 completion criterion, checked against real data.

    "Dla meczu reprezentacji generuje 8-10 poprawnych językowo zdań, każde
    odtwarzalne z provenance" -- every fact facts_for_match() surfaces for a
    real match must render into a real Polish sentence (a template gap here
    would raise UnknownFactTypeError, not silently drop a fact).
    """

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real, full-history records frame once for every test in this class."""
        cls.records_frame = pd.read_parquet(
            "football_ai/data/processed/international_results_full_history.parquet"
        )

    def test_poland_vs_germany_renders_8_to_10_sentences(self) -> None:
        facts = facts_for_match(self.records_frame, "Poland", "Germany", as_of_date=None, top_n=10)
        sentences = render_facts(facts)
        self.assertGreaterEqual(len(sentences), 8)
        self.assertLessEqual(len(sentences), 10)
        for sentence in sentences:
            self.assertIsInstance(sentence, str)
            self.assertGreater(len(sentence), 0)
            self.assertTrue(sentence.endswith("."))

    def test_every_fact_type_this_package_can_produce_has_a_working_template(self) -> None:
        """Directly exercise every insights/ function at least once against real data, not just whatever facts_for_match happens to rank highly."""
        h2h_never_won_fact = None
        for opponent in ("Senegal", "Cameroon", "Nigeria"):
            candidate = h2h.first_win_since(self.records_frame, "Poland", opponent)
            if candidate is not None and candidate.fact_type == "h2h_never_won":
                h2h_never_won_fact = candidate
                break
        self.assertIsNotNone(h2h_never_won_fact, "expected at least one real h2h_never_won fact for Poland")
        sentence = render_fact(h2h_never_won_fact)
        self.assertIn("bez ani jednego zwycięstwa", sentence)

        milestone_fact = milestones.match_count_milestone(
            self.records_frame, "Poland", pd.Timestamp("2018-06-13", tz="UTC")
        )
        self.assertIsNotNone(milestone_fact)
        self.assertEqual(
            render_fact(milestone_fact),
            "Polska rozegrała swój 800. mecz w historii reprezentacji, przeciwko Litwie.",
        )


if __name__ == "__main__":
    unittest.main()
