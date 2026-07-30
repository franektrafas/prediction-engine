"""Golden tests for football_ai/insights/records.py."""

from __future__ import annotations

from pathlib import Path
import unittest

import pandas as pd

from football_ai.insights.records import biggest_win, longest_unbeaten_streak_ever, longest_win_streak_ever


_FULL_HISTORY_PATH = Path("football_ai/data/processed/international_results_full_history.parquet")

# Poland's real biggest ever win (verified 2026-07-24 against the real
# records_frame): 10-0 vs San Marino, 2009-04-01. A World Cup qualifier --
# a real, well-known result, not invented.
_POLAND_BIGGEST_WIN_ROW = [(1, "2009-04-01", "Poland", "San Marino", 10, 0)]

# A hand-constructed, fully controllable sequence for testing the "scan the
# *entire* history, not just the trailing run" behavior _longest_run needs --
# something a real 150-year team history can't be hand-verified against
# directly. Pattern (Testland always home): W,W,W,D,L,W,W,L,W,W,W,W,W (a
# 5-match run beats the earlier 3-match run) chosen so the answer requires
# scanning past the current (trailing) streak, which is only 5, to confirm
# nothing earlier in history was longer while nothing later beats it either.
_CONTROLLED_STREAK_ROWS = [
    (1, "2000-01-01", "Testland", "A", 2, 0),  # W (streak 1)
    (2, "2000-02-01", "Testland", "B", 2, 0),  # W (streak 2)
    (3, "2000-03-01", "Testland", "C", 2, 0),  # W (streak 3) <- best so far
    (4, "2000-04-01", "Testland", "D", 1, 1),  # D (breaks win streak)
    (5, "2000-05-01", "Testland", "E", 0, 2),  # L
    (6, "2000-06-01", "Testland", "F", 2, 0),  # W (streak 1)
    (7, "2000-07-01", "Testland", "G", 2, 0),  # W (streak 2)
    (8, "2000-08-01", "Testland", "H", 0, 1),  # L (breaks)
    (9, "2000-09-01", "Testland", "I", 2, 0),  # W (streak 1)
    (10, "2000-10-01", "Testland", "J", 2, 0),  # W (streak 2)
    (11, "2000-11-01", "Testland", "K", 2, 0),  # W (streak 3)
    (12, "2000-12-01", "Testland", "L", 2, 0),  # W (streak 4)
    (13, "2001-01-01", "Testland", "M", 2, 0),  # W (streak 5) <- longest overall
]


def _frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["match_id", "match_date", "home_team_name", "away_team_name", "home_score", "away_score"]
    )
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    return frame


class RecordsGoldenTestCase(unittest.TestCase):
    """Golden facts, hand-verified against real and deliberately-controlled match rows."""

    def test_poland_biggest_win_is_the_real_ten_nil_against_san_marino(self) -> None:
        """Poland's real biggest win on record: 10-0 vs San Marino, 2009-04-01."""
        fact = biggest_win(_frame(_POLAND_BIGGEST_WIN_ROW), "Poland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["goals_for"], 10)
        self.assertEqual(fact.data["goals_against"], 0)
        self.assertEqual(fact.data["opponent"], "San Marino")
        self.assertEqual(fact.data["margin"], 10)
        self.assertEqual(fact.provenance, (1,))

    @unittest.skipUnless(_FULL_HISTORY_PATH.exists(), "Needs the real records_frame.")
    def test_poland_biggest_win_against_full_real_history(self) -> None:
        """The same fact, re-derived from the complete real 154-year history, not just the 1-row fixture."""
        records = pd.read_parquet(_FULL_HISTORY_PATH)
        fact = biggest_win(records, "Poland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["goals_for"], 10)
        self.assertEqual(fact.data["goals_against"], 0)
        self.assertEqual(fact.data["opponent"], "San Marino")

    def test_team_that_never_won_has_no_biggest_win_fact(self) -> None:
        """A team with zero wins on record must return None, not raise or report a loss as a 'win'."""
        loss_only = _frame([(1, "2000-01-01", "Loseland", "Winland", 0, 3)])
        self.assertIsNone(biggest_win(loss_only, "Loseland"))

    def test_longest_win_streak_ever_finds_the_five_match_run_not_the_earlier_three(self) -> None:
        """The best run (5, at the end) must win over an earlier, shorter run (3) -- a full scan, not a max-so-far bug."""
        fact = longest_win_streak_ever(_frame(_CONTROLLED_STREAK_ROWS), "Testland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["streak_length"], 5)
        self.assertEqual(fact.provenance, (9, 10, 11, 12, 13))

    def test_longest_unbeaten_streak_ever_includes_the_draw(self) -> None:
        """Unbeaten (win-or-draw) run: matches 1-4 (three wins then a draw) beats any all-win run here."""
        fact = longest_unbeaten_streak_ever(_frame(_CONTROLLED_STREAK_ROWS), "Testland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["streak_length"], 5)
        self.assertEqual(fact.provenance, (9, 10, 11, 12, 13))

    def test_longest_run_ties_are_broken_by_the_earlier_occurrence(self) -> None:
        """Two equal-length runs (both 3): the earlier one's match_ids must be reported, not the later one's."""
        tied_rows = [
            (1, "2000-01-01", "Testland", "A", 1, 0),
            (2, "2000-02-01", "Testland", "B", 1, 0),
            (3, "2000-03-01", "Testland", "C", 1, 0),  # first 3-run: ids 1-3
            (4, "2000-04-01", "Testland", "D", 0, 1),  # breaks it
            (5, "2000-05-01", "Testland", "E", 1, 0),
            (6, "2000-06-01", "Testland", "F", 1, 0),
            (7, "2000-07-01", "Testland", "G", 1, 0),  # second 3-run: ids 5-7
        ]
        fact = longest_win_streak_ever(_frame(tied_rows), "Testland")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["streak_length"], 3)
        self.assertEqual(fact.provenance, (1, 2, 3))


if __name__ == "__main__":
    unittest.main()
