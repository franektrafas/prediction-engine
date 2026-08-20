"""Tests for the Data Quality Report (coverage breakdowns + bundled issues)."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.data_quality import report


def _match_row(**overrides: object) -> dict[str, object]:
    row = {
        "match_id": 1,
        "match_date": pd.Timestamp("1985-06-01", tz="UTC"),
        "home_team_name": "Poland",
        "away_team_name": "Germany",
        "tournament": "Friendly",
        "home_score": 2,
        "away_score": 0,
        "home_shots": None,
        "away_shots": None,
        "home_corners": None,
        "away_corners": None,
    }
    row.update(overrides)
    return row


class CoverageByPeriodTestCase(unittest.TestCase):
    """Matches the plan's own example: "1980-1989: shots 12% ..." style output."""

    def test_buckets_by_decade_and_computes_pair_coverage(self) -> None:
        """Buckets by decade and computes pair coverage."""
        frame = pd.DataFrame(
            [
                _match_row(match_id=1, match_date=pd.Timestamp("1985-01-01", tz="UTC")),
                _match_row(
                    match_id=2,
                    match_date=pd.Timestamp("1986-01-01", tz="UTC"),
                    home_shots=10,
                    away_shots=8,
                ),
                _match_row(match_id=3, match_date=pd.Timestamp("2015-01-01", tz="UTC")),
            ]
        )
        coverage = report.coverage_by_period(frame)
        self.assertIn("1980-1989", coverage.index)
        self.assertIn("2010-2019", coverage.index)
        # 1 of 2 matches in the 1980s has both home/away shots known -> 50%.
        self.assertAlmostEqual(coverage.loc["1980-1989", "shots"], 50.0)
        self.assertAlmostEqual(coverage.loc["2010-2019", "shots"], 0.0)

    def test_one_sided_data_does_not_count_as_covered(self) -> None:
        """home_shots known but away_shots missing must not count as "covered"."""
        frame = pd.DataFrame(
            [_match_row(match_id=1, home_shots=10, away_shots=None)]
        )
        coverage = report.coverage_by_period(frame)
        self.assertAlmostEqual(coverage.loc["1980-1989", "shots"], 0.0)

    def test_missing_stat_pair_is_absent_not_zero(self) -> None:
        """A frame with no xg columns at all should omit the "xg" column entirely."""
        frame = pd.DataFrame([_match_row()])
        coverage = report.coverage_by_period(frame)
        self.assertNotIn("xg", coverage.columns)


class CoverageByTournamentTestCase(unittest.TestCase):
    """Coverage By Tournament checks."""

    def test_groups_by_tournament_name(self) -> None:
        """Groups by tournament name."""
        frame = pd.DataFrame(
            [
                _match_row(
                    match_id=1, tournament="FIFA World Cup", home_shots=10, away_shots=8
                ),
                _match_row(match_id=2, tournament="Friendly"),
            ]
        )
        coverage = report.coverage_by_tournament(frame)
        self.assertAlmostEqual(coverage.loc["FIFA World Cup", "shots"], 100.0)
        self.assertAlmostEqual(coverage.loc["Friendly", "shots"], 0.0)

    def test_missing_tournament_column_returns_empty_frame(self) -> None:
        """Missing tournament column returns empty frame."""
        frame = pd.DataFrame([{"match_id": 1}])
        coverage = report.coverage_by_tournament(frame)
        self.assertTrue(coverage.empty)


class FormatCoverageReportTestCase(unittest.TestCase):
    """Format Coverage Report checks."""

    def test_renders_the_plans_example_text_format(self) -> None:
        """Renders the plans example text format."""
        coverage = pd.DataFrame(
            {"shots": [12.0], "corners": [18.0], "cards": [72.0]},
            index=["1980-1989"],
        )
        text = report.format_coverage_report(coverage)
        self.assertEqual(text, "1980-1989: shots 12%  corners 18%  cards 72%")


class GenerateReportTestCase(unittest.TestCase):
    """The bundled report: coverage + every integrity check in one call."""

    def test_bundles_coverage_and_issues(self) -> None:
        """Bundles coverage and issues."""
        frame = pd.DataFrame(
            [
                _match_row(match_id=1),
                _match_row(match_id=1, home_team_name="France"),  # duplicate match_id
            ]
        )
        result = report.generate_report(frame)
        self.assertEqual(result.total_matches, 2)
        self.assertGreaterEqual(result.error_count(), 1)
        self.assertIn("duplicate_match_ids", result.issue_counts_by_check)

    def test_clean_data_has_zero_errors(self) -> None:
        """Clean data has zero errors."""
        frame = pd.DataFrame([_match_row(match_id=1), _match_row(match_id=2)])
        result = report.generate_report(frame)
        self.assertEqual(result.error_count(), 0)

    def test_includes_provenance_issues_when_given(self) -> None:
        """Includes provenance issues when given."""
        frame = pd.DataFrame([_match_row(match_id=1)])
        provenance = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "statistic_name": "final_score",
                    "home_value": 2.0,
                    "away_value": 0.0,
                    "source": None,
                    "license_status": "VERIFIED",
                }
            ]
        )
        result = report.generate_report(frame, provenance_frame=provenance)
        self.assertIn("missing_source", result.issue_counts_by_check)


if __name__ == "__main__":
    unittest.main()
