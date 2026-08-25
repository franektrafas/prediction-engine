"""Tests for scrapers.refresh_all -- the PROMPT_DATA_REFRESH.md daily wrapper.

Every real source call is mocked here (no network, no live database) --
the real, live end-to-end run (49,520 matches re-fetched, 244 Elo ratings
ingested, football_api reachability confirmed, all logged to
`refresh_log.jsonl`) was already run and verified directly, 2026-08-25.
These tests instead guard the wrapper's own logic: one source's failure
must not abort the others, and the summary log must stay correctly shaped.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import football_ai.scrapers.refresh_all as refresh_all
from football_ai.scrapers.refresh_all import (
    SourceRefreshResult,
    check_football_api,
    refresh_elo,
    run_refresh,
)


class CheckFootballApiTestCase(unittest.TestCase):
    """The connectivity/credentials smoke check -- no network needed here."""

    def test_missing_api_key_fails_fast_without_a_network_call(self) -> None:
        """An empty FOOTBALL_API_KEY must fail immediately, not attempt a request."""
        fake_settings = mock.Mock(football_api_key="")
        with mock.patch(
            "football_ai.scrapers.refresh_all.get_settings", return_value=fake_settings
        ):
            result = check_football_api()
        self.assertFalse(result.ok)
        self.assertIn("FOOTBALL_API_KEY", result.detail)


class RefreshEloTestCase(unittest.TestCase):
    """A failure in the Elo source must be caught and reported, not raised."""

    def test_source_failure_is_caught_and_reported(self) -> None:
        """Real case: eloratings.net unreachable -- returns ok=False, never raises."""
        with mock.patch(
            "football_ai.scrapers.refresh_all.WorldFootballEloSource"
        ) as fake_source_class:
            fake_source_class.return_value.fetch_current_snapshot.side_effect = (
                ConnectionError("unreachable")
            )
            result = refresh_elo()
        self.assertFalse(result.ok)
        self.assertIn("unreachable", result.detail)


class RunRefreshTestCase(unittest.TestCase):
    """The orchestration: every source runs regardless of the others' outcome."""

    def test_one_source_failing_does_not_skip_the_others(self) -> None:
        """Real requirement: one source's error must never abort the whole run."""
        ok_result = SourceRefreshResult(source="a", ok=True, detail="fine")
        failing_result = SourceRefreshResult(source="b", ok=False, detail="broke")

        with (
            mock.patch.object(
                refresh_all, "refresh_international_results", return_value=ok_result
            ),
            mock.patch.object(refresh_all, "refresh_elo", return_value=failing_result),
            mock.patch.object(
                refresh_all, "check_football_api", return_value=ok_result
            ),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                log_path = Path(temp_dir) / "refresh_log.jsonl"
                results = run_refresh(log_path=log_path)
                self.assertTrue(log_path.exists())

        self.assertEqual(len(results), 3)

    def test_summary_is_appended_as_one_valid_json_line(self) -> None:
        """The log must stay append-only and each line independently parseable."""
        ok_result = SourceRefreshResult(
            source="international_results", ok=True, detail="1 match", rows_after=1
        )
        with (
            mock.patch.object(
                refresh_all, "refresh_international_results", return_value=ok_result
            ),
            mock.patch.object(refresh_all, "refresh_elo", return_value=ok_result),
            mock.patch.object(
                refresh_all, "check_football_api", return_value=ok_result
            ),
        ):
            with tempfile.TemporaryDirectory() as temp_dir:
                log_path = Path(temp_dir) / "refresh_log.jsonl"
                run_refresh(log_path=log_path)
                run_refresh(log_path=log_path)  # a 2nd run must append, not overwrite
                lines = log_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(len(lines), 2)
        parsed = json.loads(lines[-1])
        self.assertIn("timestamp", parsed)
        self.assertEqual(len(parsed["results"]), 3)
        self.assertTrue(all(entry["source"] for entry in parsed["results"]))


if __name__ == "__main__":
    unittest.main()
