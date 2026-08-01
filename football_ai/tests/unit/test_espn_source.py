"""Tests for the ESPN wide-matches builder."""

from __future__ import annotations

import unittest
from unittest import mock

from football_ai.scrapers.sources.espn import EspnSource


def _competitor(home_away: str, team_name: str, score: str) -> dict:
    """Build an ESPN-shaped scoreboard competitor entry."""
    return {"homeAway": home_away, "team": {"displayName": team_name}, "score": score}


def _scoreboard_event(
    event_id: int,
    home_team: str,
    away_team: str,
    home_score: str,
    away_score: str,
    match_date: str = "2022-11-26T13:00Z",
    status: str = "STATUS_FULL_TIME",
) -> dict:
    """Build an ESPN-shaped scoreboard event payload."""
    return {
        "id": event_id,
        "date": match_date,
        "competitions": [
            {
                "status": {"type": {"name": status}},
                "venue": {"fullName": "Lusail Stadium"},
                "competitors": [
                    _competitor("home", home_team, home_score),
                    _competitor("away", away_team, away_score),
                ],
            }
        ],
    }


def _summary(team_stats: dict[str, dict[str, str]], neutral_site: bool = False) -> dict:
    """Build an ESPN-shaped match summary payload with real boxscore stat names."""
    return {
        "header": {"competitions": [{"neutralSite": neutral_site}]},
        "boxscore": {
            "teams": [
                {
                    "team": {"displayName": team_name},
                    "statistics": [{"name": name, "displayValue": value} for name, value in stats.items()],
                }
                for team_name, stats in team_stats.items()
            ]
        },
    }


class EspnWideMatchesTestCase(unittest.TestCase):
    """Validate the pivot from per-team boxscore stats into wide match rows."""

    def setUp(self) -> None:
        """Create a fresh source instance, with disk-writing side effects stubbed out."""
        self.source = EspnSource()
        self.persist_frame_patch = mock.patch.object(self.source, "persist_frame", return_value=None)
        self.persist_frame_patch.start()
        self.addCleanup(self.persist_frame_patch.stop)
        self.persist_raw_json_patch = mock.patch.object(self.source, "persist_raw_json", return_value=None)
        self.persist_raw_json_patch.start()
        self.addCleanup(self.persist_raw_json_patch.stop)

    def test_build_wide_matches_frame_pivots_home_away_stats(self) -> None:
        """Per-team boxscore stats should land as home_*/away_* columns on the match row."""
        events = [_scoreboard_event(633809, "Poland", "Saudi Arabia", "2", "0")]
        summary = _summary(
            {
                "Poland": {"totalShots": "9", "shotsOnTarget": "3", "wonCorners": "4", "yellowCards": "3"},
                "Saudi Arabia": {"totalShots": "16", "shotsOnTarget": "5", "wonCorners": "5", "yellowCards": "2"},
            }
        )
        with (
            mock.patch.object(self.source, "fetch_scoreboard", return_value=events),
            mock.patch.object(self.source, "fetch_summary", return_value=summary),
        ):
            frame = self.source.build_wide_matches_frame("fifa.world", "20221101", "20221201")

        self.assertEqual(len(frame), 1)
        row = frame.iloc[0]
        self.assertEqual(row["home_team_name"], "Poland")
        self.assertEqual(row["away_team_name"], "Saudi Arabia")
        self.assertEqual(row["home_shots"], 9)
        self.assertEqual(row["home_shots_on_target"], 3)
        self.assertEqual(row["home_corners"], 4)
        self.assertEqual(row["home_yellow_cards"], 3)
        self.assertEqual(row["away_shots"], 16)
        self.assertEqual(row["away_corners"], 5)

    def test_build_wide_matches_frame_handles_a_match_with_no_boxscore(self) -> None:
        """A match ESPN has no boxscore stats for should still produce an identity row.

        Real, expected case: non-UEFA World Cup qualifiers and FIFA
        friendlies come back with an empty per-team statistics list.
        """
        events = [_scoreboard_event(1, "Argentina", "Venezuela", "3", "0")]
        summary = _summary({"Argentina": {}, "Venezuela": {}})
        with (
            mock.patch.object(self.source, "fetch_scoreboard", return_value=events),
            mock.patch.object(self.source, "fetch_summary", return_value=summary),
        ):
            frame = self.source.build_wide_matches_frame("fifa.worldq.conmebol", "20250901", "20250910")
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["home_team_name"], "Argentina")
        self.assertNotIn("home_shots", frame.columns)

    def test_normalize_matches_skips_unplayed_fixtures(self) -> None:
        """A scheduled-but-not-yet-played fixture has no real score and must be dropped."""
        events = [
            _scoreboard_event(1, "Poland", "Sweden", "", "", status="STATUS_SCHEDULED"),
            _scoreboard_event(2, "Spain", "Italy", "2", "1"),
        ]
        frame = self.source.normalize_matches(events)
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["home_team_name"], "Spain")

    def test_build_all_international_matches_skips_a_failing_competition(self) -> None:
        """One competition failing to fetch shouldn't abort the rest of the batch."""
        good_events = [_scoreboard_event(3, "Spain", "Italy", "2", "0")]
        good_summary = _summary({"Spain": {"totalShots": "10"}, "Italy": {"totalShots": "8"}})

        def fetch_scoreboard_side_effect(league_slug: str, start_date: str, end_date: str) -> list[dict]:
            if league_slug == "fifa.world":
                raise RuntimeError("boom")
            return good_events

        with (
            mock.patch.object(self.source, "fetch_scoreboard", side_effect=fetch_scoreboard_side_effect),
            mock.patch.object(self.source, "fetch_summary", return_value=good_summary),
        ):
            frame = self.source.build_all_international_matches()

        self.assertFalse(frame.empty)
        self.assertTrue((frame["home_team_name"] == "Spain").any())


if __name__ == "__main__":
    unittest.main()
