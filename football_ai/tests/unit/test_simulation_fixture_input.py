"""Tests for football_ai/simulation/fixture_input.py.

The unplayed-fixture rows below are real -- captured 2026-07-30 from
`UefaNationsLeagueSource.fetch_group_matches("B", 4)`, Poland's actual
group. No 2026/27 Nations League match has been played yet, so the
played-match row is hand-constructed (flagged as such), matching the same
honest-labeling convention used in test_uefa_nations_league_source.py.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from football_ai.models.registry import ModelRegistry
from football_ai.predictions.schemas import MatchContext, PredictionResponse, ProbabilityPayload
from football_ai.predictions.service import PredictionService
from football_ai.simulation.fixture_input import build_simulation_input


def _empty_registry_prediction_service() -> PredictionService:
    """A real PredictionService with no registered artifacts -- deterministic heuristic fallback.

    `predict()` short-circuits to `_heuristic_prediction` the moment
    `registry.get_latest("match_outcome")` is None, before touching
    `historical_matches` at all -- so this is a real code path, not a mock,
    and is fully deterministic for a given payload (no elo -> elo_diff=0
    -> home_goals=1.35, away_goals=1.10 every time).
    """
    tmp_dir = tempfile.mkdtemp()
    registry = ModelRegistry(registry_path=Path(tmp_dir) / "registry.json")
    return PredictionService(registry=registry, historical_matches=None)


def _minimal_prediction_response(home_goals: float, away_goals: float) -> PredictionResponse:
    """Build a minimally-valid PredictionResponse carrying only the goals this module reads."""
    return PredictionResponse(
        model_version="test",
        outcome_probabilities=ProbabilityPayload(home_win=0.34, draw=0.33, away_win=0.33),
        exact_score={"home": 0, "away": 0},
        goals={"home": home_goals, "away": away_goals},
        btts=0.5,
        over_under={"over_1_5": 0.5, "over_2_5": 0.5, "over_3_5": 0.5},
        corners={"home": 5.0, "away": 5.0},
        cards={"home": 2.0, "away": 2.0},
        shots={"home": 10.0, "away": 10.0},
        shots_on_target={"home": 4.0, "away": 4.0},
        xg={"home": home_goals, "away": away_goals},
        possession={"home": 50.0, "away": 50.0},
        match_context=MatchContext(),
    )


# Real, 2026-07-30, from UefaNationsLeagueSource.fetch_group_matches("B", 4).
_REAL_UNPLAYED_FIXTURES = pd.DataFrame(
    [
        {
            "match_date": pd.Timestamp("2026-09-25"),
            "home_team_name": "Poland",
            "away_team_name": "Bosnia and Herzegovina",
            "home_score": pd.NA,
            "away_score": pd.NA,
            "stadium": "Stadion Narodowy, Warsaw",
        },
        {
            "match_date": pd.Timestamp("2026-09-25"),
            "home_team_name": "Sweden",
            "away_team_name": "Romania",
            "home_score": pd.NA,
            "away_score": pd.NA,
            "stadium": "Nationalarenan, Solna",
        },
    ]
)

# Hand-constructed (no real 2026/27 match has been played yet) -- exercises the played-match path.
_ONE_PLAYED_MATCH = pd.DataFrame(
    [
        {
            "match_date": pd.Timestamp("2026-09-25"),
            "home_team_name": "Poland",
            "away_team_name": "Bosnia and Herzegovina",
            "home_score": 2,
            "away_score": 1,
            "stadium": "Stadion Narodowy, Warsaw",
        },
    ]
)


class BuildSimulationInputTestCase(unittest.TestCase):
    """Validate the played/remaining split and the real-prediction bridge."""

    def setUp(self) -> None:
        """A real PredictionService, no trained artifacts -- deterministic heuristic path."""
        self.prediction_service = _empty_registry_prediction_service()

    def test_unplayed_rows_become_remaining_matches_with_real_predictions(self) -> None:
        """Both real, unplayed rows must end up as MatchExpectation with the heuristic's real goal output."""
        result = build_simulation_input(_REAL_UNPLAYED_FIXTURES, self.prediction_service)
        self.assertEqual(len(result.played_matches), 0)
        self.assertEqual(len(result.remaining_matches), 2)
        first = result.remaining_matches[0]
        self.assertEqual(first.home_team, "Poland")
        self.assertEqual(first.away_team, "Bosnia and Herzegovina")
        # No elo passed -> elo_diff=0 -> _heuristic_prediction's fixed 1.35/1.10 output, deterministically.
        self.assertAlmostEqual(first.expected_home_goals, 1.35)
        self.assertAlmostEqual(first.expected_away_goals, 1.10)

    def test_played_rows_use_the_real_scoreline_not_a_prediction(self) -> None:
        """A played match must carry its actual final score, untouched by the prediction service."""
        result = build_simulation_input(_ONE_PLAYED_MATCH, self.prediction_service)
        self.assertEqual(len(result.remaining_matches), 0)
        self.assertEqual(len(result.played_matches), 1)
        played = result.played_matches[0]
        self.assertEqual(played.home_team, "Poland")
        self.assertEqual(played.away_team, "Bosnia and Herzegovina")
        self.assertEqual(played.home_goals, 2)
        self.assertEqual(played.away_goals, 1)

    def test_a_mixed_frame_splits_correctly(self) -> None:
        """Played and unplayed rows in the same frame must each land in the right bucket."""
        mixed = pd.concat([_ONE_PLAYED_MATCH, _REAL_UNPLAYED_FIXTURES], ignore_index=True)
        result = build_simulation_input(mixed, self.prediction_service)
        self.assertEqual(len(result.played_matches), 1)
        self.assertEqual(len(result.remaining_matches), 2)

    def test_teams_are_the_full_unique_set_from_the_fixtures_frame(self) -> None:
        """All 4 real teams in the group must be present, regardless of played/unplayed status."""
        result = build_simulation_input(_REAL_UNPLAYED_FIXTURES, self.prediction_service)
        self.assertEqual(
            result.teams,
            sorted(["Poland", "Bosnia and Herzegovina", "Sweden", "Romania"]),
        )

    def test_elo_ratings_reach_the_prediction_payload(self) -> None:
        """A real Elo gap must move expected goals away from the flat no-Elo 1.35/1.10 heuristic default.

        `elo_ratings` uses real, current (2026-07-31) ratings for the two
        teams -- Sweden (1731) vs. Romania (1639) -- fed via
        `WorldFootballEloSource.latest_ratings_by_team()`'s real shape.
        """
        elo_ratings = {"Sweden": 1731.0, "Romania": 1639.0}
        result = build_simulation_input(_REAL_UNPLAYED_FIXTURES, self.prediction_service, elo_ratings=elo_ratings)
        sweden_vs_romania = next(m for m in result.remaining_matches if m.home_team == "Sweden")
        self.assertNotAlmostEqual(sweden_vs_romania.expected_home_goals, 1.35, places=2)
        self.assertGreater(sweden_vs_romania.expected_home_goals, sweden_vs_romania.expected_away_goals)

    def test_a_team_missing_from_elo_ratings_falls_back_to_the_default(self) -> None:
        """elo_ratings covering only ONE team in a match must not crash -- the other side stays unset."""
        result = build_simulation_input(
            _REAL_UNPLAYED_FIXTURES, self.prediction_service, elo_ratings={"Poland": 1710.0}
        )
        self.assertEqual(len(result.remaining_matches), 2)

    def test_near_zero_predicted_goals_are_floored_to_a_positive_value(self) -> None:
        """MatchExpectation rejects goals <= 0 -- this module must clamp before constructing it, not crash.

        Patched on the class, not the (slots-based) instance -- PredictionService
        uses `@dataclass(slots=True)`, which rejects arbitrary instance attributes.
        """
        with mock.patch.object(
            PredictionService, "predict", return_value=_minimal_prediction_response(0.0, -0.5)
        ):
            result = build_simulation_input(_REAL_UNPLAYED_FIXTURES.iloc[[0]], self.prediction_service)
        match = result.remaining_matches[0]
        self.assertGreater(match.expected_home_goals, 0.0)
        self.assertGreater(match.expected_away_goals, 0.0)


if __name__ == "__main__":
    unittest.main()
