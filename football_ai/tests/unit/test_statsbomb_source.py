"""Tests for the StatsBomb Open Data wide-matches builder."""

from __future__ import annotations

import unittest
from unittest import mock

from football_ai.scrapers.sources.statsbomb import StatsBombOpenDataSource


def _match_payload(
    match_id: int,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    match_date: str = "2022-12-18",
) -> dict:
    """Build a StatsBomb-shaped match payload for `normalize_matches`."""
    return {
        "match_id": match_id,
        "competition": {"competition_id": 43, "competition_name": "FIFA World Cup"},
        "season": {"season_name": "2022"},
        "match_date": match_date,
        "stadium": {"name": "Lusail Stadium"},
        "referee": {"name": "Szymon Marciniak"},
        "home_team": {"home_team_name": home_team},
        "away_team": {"away_team_name": away_team},
        "home_score": home_score,
        "away_score": away_score,
        "metadata": {"neutral_ground": True},
    }


def _shot_event(team: str, xg: float, outcome: str) -> dict:
    """Build a StatsBomb-shaped Shot event."""
    return {
        "team": {"name": team},
        "type": {"name": "Shot"},
        "shot": {"statsbomb_xg": xg, "outcome": {"name": outcome}},
    }


def _corner_event(team: str) -> dict:
    """Build a StatsBomb-shaped corner pass event."""
    return {"team": {"name": team}, "type": {"name": "Pass"}, "pass": {"type": {"name": "Corner"}}}


def _card_event(team: str, card_name: str) -> dict:
    """Build a StatsBomb-shaped foul-committed card event."""
    return {
        "team": {"name": team},
        "type": {"name": "Foul Committed"},
        "foul_committed": {"card": {"name": card_name}},
    }


def _pressure_event(team: str) -> dict:
    """Build a StatsBomb-shaped Pressure event."""
    return {"team": {"name": team}, "type": {"name": "Pressure"}}


class StatsBombWideMatchesTestCase(unittest.TestCase):
    """Validate the pivot from per-team event stats into wide match rows."""

    def setUp(self) -> None:
        """Create a fresh source instance, with disk-writing side effects stubbed out."""
        self.source = StatsBombOpenDataSource()
        # `normalize_matches()` unconditionally calls `persist_frame()`, which
        # writes a real .parquet file (needs pyarrow, not installed locally)
        # under the real repo's data/raw/statsbomb/ directory -- stub it so
        # these tests stay fast, offline, and side-effect-free.
        self.persist_frame_patch = mock.patch.object(self.source, "persist_frame", return_value=None)
        self.persist_frame_patch.start()
        self.addCleanup(self.persist_frame_patch.stop)

    def test_build_wide_matches_frame_pivots_home_away_stats(self) -> None:
        """Per-team event stats should land as home_*/away_* columns on the match row."""
        match_payload = _match_payload(1, "Argentina", "France", 3, 3)
        events = [
            _shot_event("Argentina", 0.3, "Goal"),
            _shot_event("Argentina", 0.1, "Off T"),
            _corner_event("Argentina"),
            _card_event("Argentina", "Yellow Card"),
            _shot_event("France", 0.2, "Saved"),
            _pressure_event("France"),
        ]

        with (
            mock.patch.object(self.source, "fetch_matches", return_value=[match_payload]),
            mock.patch.object(self.source, "fetch_events", return_value=events),
        ):
            frame = self.source.build_wide_matches_frame(43, 106)

        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["home_team_name"], "Argentina")
        self.assertEqual(row["away_team_name"], "France")
        self.assertAlmostEqual(row["home_xg"], 0.4)
        self.assertEqual(row["home_shots"], 2)
        self.assertEqual(row["home_shots_on_target"], 1)
        self.assertEqual(row["home_corners"], 1)
        self.assertEqual(row["home_yellow_cards"], 1)
        self.assertAlmostEqual(row["away_xg"], 0.2)
        self.assertEqual(row["away_shots"], 1)
        self.assertEqual(row["away_shots_on_target"], 1)
        self.assertEqual(row["away_pressing_intensity"], 1)

    def test_build_wide_matches_frame_handles_match_with_no_events(self) -> None:
        """A match with an empty event stream should still produce an identity row."""
        match_payload = _match_payload(2, "Poland", "Portugal", 1, 1)
        with (
            mock.patch.object(self.source, "fetch_matches", return_value=[match_payload]),
            mock.patch.object(self.source, "fetch_events", return_value=[]),
        ):
            frame = self.source.build_wide_matches_frame(43, 106)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["home_team_name"], "Poland")

    def test_build_all_international_matches_skips_a_failing_competition(self) -> None:
        """One competition failing to fetch shouldn't abort the rest of the batch."""
        good_payload = [_match_payload(3, "Spain", "Italy", 2, 0)]
        good_events = [_shot_event("Spain", 0.5, "Goal")]

        def fetch_matches_side_effect(competition_id: int, season_id: int) -> list[dict]:
            if competition_id == 43 and season_id == 106:
                raise RuntimeError("boom")
            return good_payload

        with (
            mock.patch.object(self.source, "fetch_matches", side_effect=fetch_matches_side_effect),
            mock.patch.object(self.source, "fetch_events", return_value=good_events),
        ):
            frame = self.source.build_all_international_matches()

        self.assertFalse(frame.empty)
        self.assertTrue((frame["home_team_name"] == "Spain").any())


if __name__ == "__main__":
    unittest.main()
