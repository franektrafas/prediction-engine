"""Tests for streamlit_app._match_stat_rows -- the None-safe stat-chart builder.

Regression coverage for PROMPT_ZAMKNIECIE_WATKOW.md item 3: after
`PredictionResponse.corners`/`cards`/`shots`/`shots_on_target`/`possession`
became optional (2026-08-20, following the ESPN license withdrawal), every
direct `prediction.corners["home"]`-style index in `streamlit_app.py` would
raise `TypeError` the moment a target has no registered model. This only
imports the one pure function under test, not the full Streamlit module's
top-level `st.*` calls.
"""

from __future__ import annotations

import unittest

from football_ai.predictions.schemas import PredictionResponse
from football_ai.streamlit_app import _match_stat_rows

_BASE_KWARGS = {
    "model_version": "v1",
    "outcome_probabilities": {"home_win": 0.4, "draw": 0.3, "away_win": 0.3},
    "exact_score": {"home": 1, "away": 1},
    "goals": {"home": 1.2, "away": 1.1},
    "btts": 0.5,
    "over_under": {"over_2_5": 0.4},
    "xg": {"home": 1.3, "away": 1.0},
}


class MatchStatRowsTestCase(unittest.TestCase):
    """Real bug this guards: indexing into a None stat field used to crash the UI."""

    def test_all_stats_withdrawn_returns_an_empty_list_not_a_crash(self) -> None:
        """The real current production state: all 5 ESPN-derived stats are None."""
        prediction = PredictionResponse(**_BASE_KWARGS)
        self.assertEqual(_match_stat_rows(prediction), [])

    def test_only_populated_stats_are_included(self) -> None:
        """A partially-available future state must not force all-or-nothing."""
        prediction = PredictionResponse(
            **_BASE_KWARGS,
            corners={"home": 5.0, "away": 4.0},
        )
        rows = _match_stat_rows(prediction)
        self.assertEqual(rows, [("Rzuty rożne", 5.0, 4.0)])

    def test_every_stat_populated_preserves_label_order(self) -> None:
        """When every model is registered, all five rows come back in a fixed order."""
        prediction = PredictionResponse(
            **_BASE_KWARGS,
            corners={"home": 5.0, "away": 4.0},
            cards={"home": 2.0, "away": 3.0},
            shots={"home": 12.0, "away": 9.0},
            shots_on_target={"home": 6.0, "away": 4.0},
            possession={"home": 55.0, "away": 45.0},
        )
        rows = _match_stat_rows(prediction)
        labels = [label for label, _, _ in rows]
        expected_labels = [
            "Rzuty rożne",
            "Żółte kartki",
            "Strzały",
            "Strzały celne",
            "Posiadanie (%)",
        ]
        self.assertEqual(labels, expected_labels)


if __name__ == "__main__":
    unittest.main()
