"""Tests for the Data Quality Layer's match_sources provenance checks."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.data_quality import provenance_checks


def _provenance_row(**overrides: object) -> dict[str, object]:
    """A single valid provenance row (martj42's final score for a real match)."""
    row = {
        "match_id": 1,
        "statistic_name": "final_score",
        "home_value": 1.0,
        "away_value": 1.0,
        "source": "martj42",
        "source_license": "CC0",
        "license_status": "VERIFIED",
        "confidence": "HIGH",
    }
    row.update(overrides)
    return row


class FindMissingSourceTestCase(unittest.TestCase):
    """Find Missing Source checks."""

    def test_populated_source_passes(self) -> None:
        """Populated source passes."""
        frame = pd.DataFrame([_provenance_row()])
        self.assertEqual(provenance_checks.find_missing_source(frame), [])

    def test_null_source_is_an_error(self) -> None:
        """Null source is an error."""
        frame = pd.DataFrame([_provenance_row(source=None)])
        issues = provenance_checks.find_missing_source(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")

    def test_blank_source_is_an_error(self) -> None:
        """Blank source is an error."""
        frame = pd.DataFrame([_provenance_row(source="")])
        issues = provenance_checks.find_missing_source(frame)
        self.assertEqual(len(issues), 1)


class FindMissingLicenseStatusTestCase(unittest.TestCase):
    """Find Missing License Status checks."""

    def test_populated_status_passes(self) -> None:
        """Populated status passes."""
        frame = pd.DataFrame([_provenance_row()])
        self.assertEqual(provenance_checks.find_missing_license_status(frame), [])

    def test_null_status_is_an_error(self) -> None:
        """A missing license_status is worse than "UNKNOWN" -- never classified."""
        frame = pd.DataFrame([_provenance_row(license_status=None)])
        issues = provenance_checks.find_missing_license_status(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")

    def test_the_real_unknown_value_is_not_treated_as_missing(self) -> None:
        """The real unknown value is not treated as missing."""
        frame = pd.DataFrame([_provenance_row(license_status="UNKNOWN")])
        self.assertEqual(provenance_checks.find_missing_license_status(frame), [])


class FindUnverifiedLicensesTestCase(unittest.TestCase):
    """Find Unverified Licenses checks."""

    def test_verified_row_passes(self) -> None:
        """Verified row passes."""
        frame = pd.DataFrame([_provenance_row(license_status="VERIFIED")])
        self.assertEqual(provenance_checks.find_unverified_licenses(frame), [])

    def test_unknown_status_is_a_warning_not_dropped(self) -> None:
        """Per the project's rule: never drop, always surface before ML use."""
        frame = pd.DataFrame([_provenance_row(license_status="UNKNOWN")])
        issues = provenance_checks.find_unverified_licenses(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")

    def test_research_only_status_is_a_warning(self) -> None:
        """Research only status is a warning."""
        frame = pd.DataFrame([_provenance_row(license_status="RESEARCH_ONLY")])
        issues = provenance_checks.find_unverified_licenses(frame)
        self.assertEqual(len(issues), 1)


class FindConflictingSourcesTestCase(unittest.TestCase):
    """The conflict-preserving heart of the Data Provenance Layer."""

    def test_single_source_is_never_a_conflict(self) -> None:
        """Single source is never a conflict."""
        frame = pd.DataFrame([_provenance_row()])
        self.assertEqual(provenance_checks.find_conflicting_sources(frame), [])

    def test_two_sources_agreeing_is_not_a_conflict(self) -> None:
        """Two sources agreeing is not a conflict."""
        frame = pd.DataFrame(
            [
                _provenance_row(source="martj42", home_value=1.0, away_value=1.0),
                _provenance_row(source="espn", home_value=1.0, away_value=1.0),
            ]
        )
        self.assertEqual(provenance_checks.find_conflicting_sources(frame), [])

    def test_two_sources_disagreeing_is_flagged(self) -> None:
        """Two sources disagreeing is flagged."""
        frame = pd.DataFrame(
            [
                _provenance_row(source="martj42", home_value=1.0, away_value=1.0),
                _provenance_row(
                    source="unverified_scraper", home_value=2.0, away_value=1.0
                ),
            ]
        )
        issues = provenance_checks.find_conflicting_sources(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "warning")
        self.assertIn("martj42", issues[0].details["sources"])
        self.assertIn("unverified_scraper", issues[0].details["sources"])

    def test_different_statistics_on_the_same_match_are_independent(self) -> None:
        """A conflict on "shots" must not spuriously involve "final_score" rows."""
        frame = pd.DataFrame(
            [
                _provenance_row(statistic_name="final_score", source="martj42"),
                _provenance_row(
                    statistic_name="shots",
                    source="martj42",
                    home_value=10.0,
                    away_value=8.0,
                ),
                _provenance_row(
                    statistic_name="shots",
                    source="espn",
                    home_value=99.0,
                    away_value=8.0,
                ),
            ]
        )
        issues = provenance_checks.find_conflicting_sources(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].subject, (1, "shots"))


class RunAllChecksTestCase(unittest.TestCase):
    """Run All Checks checks."""

    def test_clean_verified_data_produces_no_issues(self) -> None:
        """Clean verified data produces no issues."""
        frame = pd.DataFrame([_provenance_row()])
        self.assertEqual(provenance_checks.run_all_checks(frame), [])

    def test_combines_findings_from_multiple_checks(self) -> None:
        """Combines findings from multiple checks."""
        frame = pd.DataFrame(
            [
                _provenance_row(source=None),
                _provenance_row(license_status="UNKNOWN"),
            ]
        )
        issues = provenance_checks.run_all_checks(frame)
        check_names = {issue.check_name for issue in issues}
        self.assertIn("missing_source", check_names)
        self.assertIn("unverified_license", check_names)


if __name__ == "__main__":
    unittest.main()
