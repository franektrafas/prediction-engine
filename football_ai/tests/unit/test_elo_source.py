"""Tests for football_ai/scrapers/sources/elo.py."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.scrapers.sources.elo import WorldFootballEloSource


class LatestRatingsByTeamTestCase(unittest.TestCase):
    """Validate the snapshot -> {team: rating} reshape used to feed live fixture predictions."""

    def test_builds_a_simple_team_to_rating_lookup(self) -> None:
        snapshot = pd.DataFrame(
            [
                {"rank": 1, "team_name": "Spain", "rating": 2259.0},
                {"rank": 43, "team_name": "Poland", "rating": 1710.0},
            ]
        )
        ratings = WorldFootballEloSource.latest_ratings_by_team(snapshot)
        self.assertEqual(ratings, {"Spain": 2259.0, "Poland": 1710.0})

    def test_empty_snapshot_returns_an_empty_dict(self) -> None:
        empty = pd.DataFrame(columns=["rank", "team_name", "rating"])
        self.assertEqual(WorldFootballEloSource.latest_ratings_by_team(empty), {})


if __name__ == "__main__":
    unittest.main()
