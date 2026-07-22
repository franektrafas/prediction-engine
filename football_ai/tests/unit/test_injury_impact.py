"""Tests for value-weighted injury impact enrichment."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from football_ai.predictions.injury_impact import (
    compute_injured_market_value,
    enrich_payload_with_injury_impact,
)
from football_ai.predictions.schemas import MatchContextPayload


def _injuries_frame() -> pd.DataFrame:
    """Build a small injuries frame for two teams."""
    return pd.DataFrame(
        [
            {"team_name": "Poland", "player_name": "Filip Rozga", "injury_category": "Injuries"},
            {"team_name": "Poland", "player_name": "Unknown Player", "injury_category": "Injuries"},
        ]
    )


def _player_reference_frame() -> pd.DataFrame:
    """Build a small player reference frame with market values."""
    return pd.DataFrame(
        [
            {"team_name": "Poland", "player_name": "Filip Rozga", "market_value_eur": 1_500_000.0},
            {"team_name": "Poland", "player_name": "Robert Lewandowski", "market_value_eur": 8_000_000.0},
            {"team_name": "Croatia", "player_name": "Filip Rozga", "market_value_eur": 999.0},
        ]
    )


def _payload(**overrides: object) -> MatchContextPayload:
    """Build a minimal payload, applying any field overrides."""
    fields = {
        "match_date": datetime(2026, 7, 1, tzinfo=timezone.utc),
        "home_team_name": "Poland",
        "away_team_name": "Croatia",
    }
    fields.update(overrides)
    return MatchContextPayload(**fields)


class ComputeInjuredMarketValueTestCase(unittest.TestCase):
    """Validate the team_name/player_name matched market-value sum."""

    def test_sums_matched_injured_players(self) -> None:
        """Only the matched injured player's value should be summed, not the whole squad."""
        value = compute_injured_market_value("Poland", _injuries_frame(), _player_reference_frame())
        self.assertAlmostEqual(value, 1_500_000.0)

    def test_unmatched_injured_player_contributes_zero(self) -> None:
        """An injured player absent from player_reference shouldn't error or inflate the sum."""
        injuries = pd.DataFrame([{"team_name": "Poland", "player_name": "Nobody Here"}])
        value = compute_injured_market_value("Poland", injuries, _player_reference_frame())
        self.assertEqual(value, 0.0)

    def test_uninjured_team_returns_zero(self) -> None:
        """A team with no rows in the injuries frame should return 0.0, not error."""
        value = compute_injured_market_value("Germany", _injuries_frame(), _player_reference_frame())
        self.assertEqual(value, 0.0)

    def test_does_not_cross_match_a_same_named_player_on_another_team(self) -> None:
        """Croatia's "Filip Rozga" must not be summed for Poland's injury."""
        value = compute_injured_market_value("Poland", _injuries_frame(), _player_reference_frame())
        self.assertNotAlmostEqual(value, 1_500_000.0 + 999.0)

    def test_empty_frames_return_zero(self) -> None:
        """Empty injuries or player_reference frames should be a safe no-op."""
        self.assertEqual(compute_injured_market_value("Poland", pd.DataFrame(), _player_reference_frame()), 0.0)
        self.assertEqual(compute_injured_market_value("Poland", _injuries_frame(), pd.DataFrame()), 0.0)


class EnrichPayloadWithInjuryImpactTestCase(unittest.TestCase):
    """Validate the payload-enrichment wrapper's fill/never-raise contract."""

    def test_fills_unset_fields_for_both_sides(self) -> None:
        """Both home and away injured-market-value fields should be filled when unset."""
        payload = _payload()
        result = enrich_payload_with_injury_impact(
            payload, injuries=_injuries_frame(), player_reference=_player_reference_frame()
        )
        self.assertAlmostEqual(result.home_injured_market_value, 1_500_000.0)
        self.assertEqual(result.away_injured_market_value, 0.0)

    def test_does_not_override_an_explicit_value(self) -> None:
        """A caller-supplied value must survive enrichment untouched."""
        payload = _payload(home_injured_market_value=42.0)
        result = enrich_payload_with_injury_impact(
            payload, injuries=_injuries_frame(), player_reference=_player_reference_frame()
        )
        self.assertEqual(result.home_injured_market_value, 42.0)

    def test_independent_of_fixture_id(self) -> None:
        """Unlike lineup enrichment, this should work with no fixture_id at all."""
        payload = _payload(fixture_id=None)
        result = enrich_payload_with_injury_impact(
            payload, injuries=_injuries_frame(), player_reference=_player_reference_frame()
        )
        self.assertAlmostEqual(result.home_injured_market_value, 1_500_000.0)

    def test_empty_injuries_returns_payload_unchanged(self) -> None:
        """No injuries data at all should be a safe no-op, not force fields to 0.0."""
        payload = _payload()
        result = enrich_payload_with_injury_impact(
            payload, injuries=pd.DataFrame(), player_reference=_player_reference_frame()
        )
        self.assertIsNone(result.home_injured_market_value)
        self.assertIsNone(result.away_injured_market_value)

    def test_loader_failure_degrades_gracefully(self) -> None:
        """A settings/loader failure must not break the caller."""
        payload = _payload()

        class BrokenFrame:
            """Stand-in that raises when treated as a DataFrame."""

            @property
            def empty(self) -> bool:
                raise RuntimeError("no database configured")

        result = enrich_payload_with_injury_impact(payload, injuries=BrokenFrame(), player_reference=None)
        self.assertEqual(result, payload)


if __name__ == "__main__":
    unittest.main()
