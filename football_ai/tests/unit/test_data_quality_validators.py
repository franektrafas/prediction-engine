"""Tests for the Data Quality Layer's records_frame/goalscorers_frame checks."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.data_quality import validators


def _match_row(**overrides: object) -> dict[str, object]:
    """A single valid, real match row (Poland 2:0 Germany, 2023-06-16), overridable."""
    row = {
        "match_id": 1,
        "match_date": pd.Timestamp("2023-06-16", tz="UTC"),
        "home_team_name": "Poland",
        "away_team_name": "Germany",
        "home_score": 2,
        "away_score": 0,
        "home_shots": 10,
        "away_shots": 8,
        "home_shots_on_target": 4,
        "away_shots_on_target": 3,
        "home_possession_pct": 45.0,
        "away_possession_pct": 55.0,
    }
    row.update(overrides)
    return row


class FindDuplicateMatchesTestCase(unittest.TestCase):
    """Real finding: martj42 has two rows for Tahiti v New Caledonia, 1974-02-17."""

    def test_clean_frame_has_no_duplicates(self) -> None:
        """Clean frame has no duplicates."""
        frame = pd.DataFrame(
            [_match_row(match_id=1), _match_row(match_id=2, away_team_name="France")]
        )
        self.assertEqual(validators.find_duplicate_matches(frame), [])

    def test_same_fixture_twice_is_flagged(self) -> None:
        """Same fixture twice is flagged."""
        frame = pd.DataFrame([_match_row(match_id=1), _match_row(match_id=2)])
        issues = validators.find_duplicate_matches(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].check_name, "duplicate_matches")
        self.assertEqual(issues[0].details["count"], 2)


class FindDuplicateMatchIdsTestCase(unittest.TestCase):
    """match_id must be a stable, unique key."""

    def test_unique_ids_pass(self) -> None:
        """Unique ids pass."""
        frame = pd.DataFrame([_match_row(match_id=1), _match_row(match_id=2)])
        self.assertEqual(validators.find_duplicate_match_ids(frame), [])

    def test_repeated_id_is_flagged_as_error(self) -> None:
        """Repeated id is flagged as error."""
        frame = pd.DataFrame(
            [_match_row(match_id=1), _match_row(match_id=1, away_team_name="France")]
        )
        issues = validators.find_duplicate_match_ids(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")


class FindDuplicateEventsTestCase(unittest.TestCase):
    """Real finding: martj42 has byte-identical goalscorer rows (unknown minute)."""

    def test_two_different_scorers_same_minute_is_not_a_duplicate(self) -> None:
        """Two different scorers same minute is not a duplicate."""
        frame = pd.DataFrame(
            [
                {"match_id": 1, "scorer": "A", "minute": 10.0, "team_name": "Poland"},
                {"match_id": 1, "scorer": "B", "minute": 10.0, "team_name": "Poland"},
            ]
        )
        self.assertEqual(validators.find_duplicate_events(frame), [])

    def test_identical_row_appearing_twice_is_flagged(self) -> None:
        """Identical row appearing twice is flagged."""
        frame = pd.DataFrame(
            [
                {"match_id": 1, "scorer": "A", "minute": 10.0, "team_name": "Poland"},
                {"match_id": 1, "scorer": "A", "minute": 10.0, "team_name": "Poland"},
            ]
        )
        issues = validators.find_duplicate_events(frame)
        self.assertEqual(len(issues), 2)  # both rows in the duplicate pair are reported


class FindSelfMatchesTestCase(unittest.TestCase):
    """A team can never play itself."""

    def test_normal_match_passes(self) -> None:
        """Normal match passes."""
        frame = pd.DataFrame([_match_row()])
        self.assertEqual(validators.find_self_matches(frame), [])

    def test_team_playing_itself_is_an_error(self) -> None:
        """Team playing itself is an error."""
        frame = pd.DataFrame([_match_row(away_team_name="Poland")])
        issues = validators.find_self_matches(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")


class FindImpossibleScoresTestCase(unittest.TestCase):
    """Real fact: a legitimate 31-0 exists in the data -- no upper bound is enforced."""

    def test_a_real_blowout_score_is_not_flagged(self) -> None:
        """A real blowout score is not flagged."""
        frame = pd.DataFrame([_match_row(home_score=31, away_score=0)])
        self.assertEqual(validators.find_impossible_scores(frame), [])

    def test_negative_score_is_an_error(self) -> None:
        """Negative score is an error."""
        frame = pd.DataFrame([_match_row(home_score=-1)])
        issues = validators.find_impossible_scores(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")


class FindNegativeStatisticsTestCase(unittest.TestCase):
    """No team statistic can be negative."""

    def test_clean_statistics_pass(self) -> None:
        """Clean statistics pass."""
        frame = pd.DataFrame([_match_row()])
        self.assertEqual(validators.find_negative_statistics(frame), [])

    def test_negative_corners_is_an_error(self) -> None:
        """Negative corners is an error."""
        frame = pd.DataFrame([_match_row(home_corners=-2)])
        issues = validators.find_negative_statistics(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].details["column"], "home_corners")


class FindInvalidPossessionTestCase(unittest.TestCase):
    """Possession must be in [0, 100] and both sides must sum to ~100."""

    def test_real_rounding_up_to_101_is_tolerated(self) -> None:
        """Real observed max across 841 real rows (2026-08-20) is 101.0."""
        frame = pd.DataFrame(
            [_match_row(home_possession_pct=51.0, away_possession_pct=50.0)]
        )
        self.assertEqual(validators.find_invalid_possession(frame), [])

    def test_possession_over_100_is_an_error(self) -> None:
        """Possession over 100 is an error."""
        frame = pd.DataFrame([_match_row(home_possession_pct=110.0)])
        issues = validators.find_invalid_possession(frame)
        check_names = {issue.check_name for issue in issues}
        self.assertIn("possession_out_of_range", check_names)

    def test_possession_not_summing_to_100_is_an_error(self) -> None:
        """Possession not summing to 100 is an error."""
        frame = pd.DataFrame(
            [_match_row(home_possession_pct=60.0, away_possession_pct=60.0)]
        )
        issues = validators.find_invalid_possession(frame)
        check_names = {issue.check_name for issue in issues}
        self.assertIn("possession_does_not_sum_to_100", check_names)


class FindShotsLessThanShotsOnTargetTestCase(unittest.TestCase):
    """Shots on target can never exceed total shots -- 0 violations in 841 real rows."""

    def test_clean_data_passes(self) -> None:
        """Clean data passes."""
        frame = pd.DataFrame([_match_row()])
        self.assertEqual(validators.find_shots_less_than_shots_on_target(frame), [])

    def test_more_shots_on_target_than_shots_is_an_error(self) -> None:
        """More shots on target than shots is an error."""
        frame = pd.DataFrame(
            [_match_row(home_shots=5, home_shots_on_target=8)]
        )
        issues = validators.find_shots_less_than_shots_on_target(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")


class FindDateInconsistenciesTestCase(unittest.TestCase):
    """A match cannot be dated after the reference "as_of" moment."""

    def test_past_match_passes(self) -> None:
        """Past match passes."""
        frame = pd.DataFrame([_match_row()])
        as_of = pd.Timestamp("2026-08-20", tz="UTC")
        self.assertEqual(validators.find_date_inconsistencies(frame, as_of), [])

    def test_future_match_is_an_error(self) -> None:
        """Future match is an error."""
        frame = pd.DataFrame(
            [_match_row(match_date=pd.Timestamp("2099-01-01", tz="UTC"))]
        )
        as_of = pd.Timestamp("2026-08-20", tz="UTC")
        issues = validators.find_date_inconsistencies(frame, as_of)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")


class FindImpossibleEventTimestampsTestCase(unittest.TestCase):
    """Real max observed goal-event minute (47,914 real rows, 2026-08-20) is 122."""

    def test_real_extra_time_minute_is_not_flagged(self) -> None:
        """Real extra time minute is not flagged."""
        frame = pd.DataFrame([{"match_id": 1, "minute": 122.0}])
        self.assertEqual(validators.find_impossible_event_timestamps(frame), [])

    def test_missing_minute_is_not_flagged(self) -> None:
        """A NaN minute is a coverage gap, not a violation."""
        frame = pd.DataFrame([{"match_id": 1, "minute": None}])
        self.assertEqual(validators.find_impossible_event_timestamps(frame), [])

    def test_absurd_minute_is_an_error(self) -> None:
        """Absurd minute is an error."""
        frame = pd.DataFrame([{"match_id": 1, "minute": 500.0}])
        issues = validators.find_impossible_event_timestamps(frame)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].severity, "error")

    def test_negative_minute_is_an_error(self) -> None:
        """Negative minute is an error."""
        frame = pd.DataFrame([{"match_id": 1, "minute": -5.0}])
        issues = validators.find_impossible_event_timestamps(frame)
        self.assertEqual(len(issues), 1)


class RunAllChecksTestCase(unittest.TestCase):
    """The aggregator combines every check and tolerates missing optional inputs."""

    def test_clean_data_produces_no_issues(self) -> None:
        """Clean data produces no issues."""
        frame = pd.DataFrame([_match_row()])
        as_of = pd.Timestamp("2026-08-20", tz="UTC")
        self.assertEqual(validators.run_all_checks(frame, as_of=as_of), [])

    def test_works_without_goalscorers_or_as_of(self) -> None:
        """Works without goalscorers or as of."""
        frame = pd.DataFrame([_match_row()])
        issues = validators.run_all_checks(frame)
        self.assertEqual(issues, [])

    def test_combines_findings_from_multiple_checks(self) -> None:
        """Combines findings from multiple checks."""
        frame = pd.DataFrame(
            [_match_row(home_score=-1, home_corners=-3, away_team_name="Poland")]
        )
        issues = validators.run_all_checks(frame)
        check_names = {issue.check_name for issue in issues}
        self.assertIn("impossible_score", check_names)
        self.assertIn("negative_statistic", check_names)
        self.assertIn("self_match", check_names)


if __name__ == "__main__":
    unittest.main()
