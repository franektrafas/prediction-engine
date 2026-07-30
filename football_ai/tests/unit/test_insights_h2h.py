"""Golden tests for football_ai/insights/h2h.py.

Poland-Germany is PLAN_SILNIKA.md's own worked example ("Polska nie
wygrała z Niemcami do 2014") -- verified for real (2026-07-30) against the
full records_frame: Poland's real head-to-head record with Germany, oldest
first, is a 1933 loss then nothing but losses/draws until a 2014-10-11 2-0
win (their first ever) and a second win on 2023-06-16. That real sequence
is embedded below as a small fixture so the suite runs without the
gitignored records_frame file.
"""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.insights.h2h import first_win_since, h2h_matches


# Poland-Germany, real matches (verbatim from the real records_frame,
# 2026-07-30), abbreviated to the meetings that matter for these tests:
# the first-ever ("since forever") ties/losses, the historic first win in
# 2014, and the second win in 2023.
_POLAND_GERMANY_REAL_ROWS = [
    (1, "1933-12-03", "Germany", "Poland", 1, 0),
    (2, "1936-09-13", "Poland", "Germany", 1, 1),
    (3, "1971-11-17", "Germany", "Poland", 0, 0),
    (4, "2011-09-06", "Poland", "Germany", 2, 2),
    (5, "2014-10-11", "Poland", "Germany", 2, 0),  # Poland's first-ever win
    (6, "2023-06-16", "Poland", "Germany", 1, 0),  # Poland's second win
]


def _frame(rows: list[tuple]) -> pd.DataFrame:
    frame = pd.DataFrame(
        rows, columns=["match_id", "match_date", "home_team_name", "away_team_name", "home_score", "away_score"]
    )
    frame["match_date"] = pd.to_datetime(frame["match_date"], utc=True)
    return frame


class H2hGoldenTestCase(unittest.TestCase):
    """Golden facts, hand-verified against the real Poland-Germany head-to-head record."""

    def test_h2h_matches_normalizes_both_sides_regardless_of_host(self) -> None:
        """Poland's perspective must be consistent whether Poland or Germany hosted."""
        matches = h2h_matches(_frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Germany")
        self.assertEqual(len(matches), 6)
        # Row 1: Germany 1-0 Poland (Germany home) -> from Poland's side, 0-1, a loss.
        row1 = matches.iloc[0]
        self.assertEqual(row1["team_a_goals"], 0)
        self.assertEqual(row1["team_b_goals"], 1)
        self.assertEqual(row1["team_a_result"], "loss")
        # Row 5: Poland 2-0 Germany (Poland home) -> from Poland's side, 2-0, a win.
        row5 = matches.iloc[4]
        self.assertEqual(row5["team_a_result"], "win")

    def test_poland_first_ever_win_against_germany_as_of_right_after_it_happened(self) -> None:
        """Queried right after the 2014 win (before the 2023 rematch): PLAN_SILNIKA.md's exact example.

        As of e.g. 2020, Poland's head-to-head record shows exactly one win
        (2014) with nothing before it -- "first ever", not "first since Y".
        """
        fact = first_win_since(
            _frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Germany", as_of_date=pd.Timestamp("2020-01-01", tz="UTC")
        )
        self.assertIsNotNone(fact)
        self.assertEqual(fact.fact_type, "h2h_first_win_since")
        self.assertEqual(fact.data["win_year"], 2014)
        self.assertIsNone(fact.data["gap_years"])
        self.assertIn("2014", fact.summary)
        self.assertIn("only win", fact.summary)

    def test_poland_second_win_against_germany_reports_the_real_nine_year_gap(self) -> None:
        """Queried after both real wins: the 2023 win was Poland's first since the 2014 one -- a 9-year gap."""
        fact = first_win_since(_frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Germany")
        self.assertIsNotNone(fact)
        self.assertEqual(fact.data["win_year"], 2023)
        self.assertEqual(fact.data["gap_years"], 9)
        self.assertIn("2023", fact.summary)
        self.assertIn("2014", fact.summary)

    def test_before_the_2014_win_poland_had_never_beaten_germany(self) -> None:
        """Queried before 2014: Poland's real record was winless against Germany since their first 1933 meeting."""
        fact = first_win_since(
            _frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Germany", as_of_date=pd.Timestamp("2012-01-01", tz="UTC")
        )
        self.assertIsNotNone(fact)
        self.assertEqual(fact.fact_type, "h2h_never_won")
        self.assertIn("never beaten", fact.summary)
        self.assertEqual(fact.data["meeting_count"], 4)

    def test_h2h_matches_with_no_shared_history_is_empty(self) -> None:
        """Two teams with zero meetings in the fixture must return an empty frame, not raise."""
        matches = h2h_matches(_frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Atlantis")
        self.assertTrue(matches.empty)

    def test_first_win_since_with_no_shared_history_returns_none(self) -> None:
        """No head-to-head history at all -- genuinely nothing to report, not a 'never won' fact."""
        self.assertIsNone(first_win_since(_frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Atlantis"))

    def test_gap_fact_provenance_spans_both_bracketing_wins(self) -> None:
        """Provenance for the 9-year-gap fact must cover both wins and everything in between, reproducibly."""
        fact = first_win_since(_frame(_POLAND_GERMANY_REAL_ROWS), "Poland", "Germany")
        self.assertEqual(fact.provenance, (5, 6))


if __name__ == "__main__":
    unittest.main()
