"""Tests for football_ai/insights/verification.py -- the reproducibility guarantee itself."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from football_ai.insights import h2h, milestones, records, streaks
from football_ai.insights.schemas import Fact
from football_ai.insights.verification import UnverifiableFactTypeError, verify, verify_all


_FULL_HISTORY_PATH = Path("football_ai/data/processed/international_results_full_history.parquet")


@unittest.skipUnless(_FULL_HISTORY_PATH.exists(), "requires the regenerated full-history parquet")
class VerifyRealFactsTestCase(unittest.TestCase):
    """A fact produced by this package's own detectors must verify against the same data."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load the real records_frame once for the whole test class."""
        cls.records_frame = pd.read_parquet(_FULL_HISTORY_PATH)

    def test_a_real_streak_fact_verifies(self) -> None:
        """A genuinely-computed streak fact must reproduce cleanly from its own provenance."""
        fact = streaks.win_streak(self.records_frame, "Spain")
        if fact is None:
            self.skipTest("Spain has no active win streak in the current data -- nothing to verify")
        self.assertTrue(verify(fact, self.records_frame))

    def test_a_real_record_fact_verifies(self) -> None:
        """Poland's real biggest-win record must reproduce."""
        fact = records.biggest_win(self.records_frame, "Poland")
        self.assertIsNotNone(fact)
        self.assertTrue(verify(fact, self.records_frame))

    def test_a_real_h2h_fact_verifies(self) -> None:
        """The Poland-Germany 'first win since' fact must reproduce."""
        fact = h2h.first_win_since(self.records_frame, "Poland", "Germany")
        self.assertIsNotNone(fact)
        self.assertTrue(verify(fact, self.records_frame))

    def test_a_real_milestone_fact_verifies(self) -> None:
        """Poland's real 800th-match milestone must reproduce."""
        fact = milestones.match_count_milestone(
            self.records_frame, "Poland", as_of_date=pd.Timestamp("2018-06-13", tz="UTC")
        )
        self.assertIsNotNone(fact)
        self.assertTrue(verify(fact, self.records_frame))

    def test_verify_all_keeps_only_verifiable_facts(self) -> None:
        """A mix of a real fact and a tampered one -- only the real one should survive."""
        real_fact = records.biggest_win(self.records_frame, "Poland")
        tampered_fact = Fact(
            fact_type=real_fact.fact_type,
            subject_team=real_fact.subject_team,
            summary=real_fact.summary,
            data=real_fact.data,
            weight=real_fact.weight,
            provenance=(999_999_999,),  # a match_id that doesn't exist
        )
        survivors = verify_all([real_fact, tampered_fact], self.records_frame)
        self.assertEqual(survivors, [real_fact])


class VerifyTamperedFactsTestCase(unittest.TestCase):
    """Validate failure modes without needing the real dataset."""

    def test_provenance_pointing_at_a_nonexistent_match_fails(self) -> None:
        """A fact whose provenance references match_ids absent from records_frame can't verify."""
        records_frame = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_date": pd.Timestamp("2020-01-01", tz="UTC"),
                    "home_team_name": "Testland",
                    "away_team_name": "Opponent",
                    "home_score": 1,
                    "away_score": 0,
                }
            ]
        )
        fake_fact = Fact(
            fact_type="win_streak",
            subject_team="Testland",
            summary="fabricated",
            data={"streak_length": 5, "venue": None},
            weight=0.5,
            provenance=(1, 2, 3, 4, 5),  # only match_id 1 actually exists
        )
        self.assertFalse(verify(fake_fact, records_frame))

    def test_unknown_fact_type_raises(self) -> None:
        """A fact_type no verifier is registered for should fail loudly, not silently pass/fail."""
        records_frame = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_date": pd.Timestamp("2020-01-01", tz="UTC"),
                    "home_team_name": "Testland",
                    "away_team_name": "Opponent",
                    "home_score": 1,
                    "away_score": 0,
                }
            ]
        )
        bogus_fact = Fact(
            fact_type="not_a_real_fact_type",
            subject_team="Testland",
            summary="fabricated",
            data={},
            weight=0.5,
            provenance=(1,),
        )
        with self.assertRaises(UnverifiableFactTypeError):
            verify(bogus_fact, records_frame)


if __name__ == "__main__":
    unittest.main()
