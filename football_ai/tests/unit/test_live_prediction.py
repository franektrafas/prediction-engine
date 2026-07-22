"""Tests for wiring live api-football.com lineups into prediction payloads."""

from __future__ import annotations

from datetime import datetime, timezone
import unittest

import pandas as pd

from football_ai.predictions.live_prediction import enrich_payload_with_live_lineups
from football_ai.predictions.schemas import MatchContextPayload


_PORTUGAL_ROSTER = [("Player One", 1), ("Player Two", 2)]
_CROATIA_ROSTER = [("Player Three", 3), ("Player Four", 4)]


def _team_rows(fixture_id: int, team_name: str, roster: list[tuple[str, int]]) -> list[dict[str, object]]:
    """Build lineup rows for one team's roster."""
    return [
        {
            "fixture_id": fixture_id,
            "team_name": team_name,
            "player_name": player_name,
            "player_external_id": external_id,
            "position": "G" if external_id % 2 == 1 else "M",
            "grid": "1:1",
            "started": True,
        }
        for player_name, external_id in roster
    ]


def _lineup_rows(fixture_id: int, reversed_order: bool = False) -> pd.DataFrame:
    """Build a Portugal-vs-Croatia lineup frame for a fixture.

    Each team keeps its own real roster; `reversed_order` only changes which
    team's rows the API lists first -- this is what `build_live_lineup_features`
    uses to decide which team is "home" -- so it exercises the same swap logic
    a real feed reversing home/away would.
    """
    teams = [("Croatia", _CROATIA_ROSTER), ("Portugal", _PORTUGAL_ROSTER)] if reversed_order else [
        ("Portugal", _PORTUGAL_ROSTER),
        ("Croatia", _CROATIA_ROSTER),
    ]
    rows: list[dict[str, object]] = []
    for team_name, roster in teams:
        rows.extend(_team_rows(fixture_id, team_name, roster))
    return pd.DataFrame(rows)


def _player_reference() -> pd.DataFrame:
    """Build a reference table enriching the four fixture players."""
    return pd.DataFrame(
        [
            {"team_name": "Portugal", "player_name": "Player One", "player_external_id": 1, "age": 20.0},
            {"team_name": "Portugal", "player_name": "Player Two", "player_external_id": 2, "age": 30.0},
            {"team_name": "Croatia", "player_name": "Player Three", "player_external_id": 3, "age": 40.0},
            {"team_name": "Croatia", "player_name": "Player Four", "player_external_id": 4, "age": 20.0},
        ]
    )


class FakeLineupSource:
    """Stand-in for `FootballApiSource.fetch_fixture_lineups`."""

    def __init__(self, frame: pd.DataFrame | None = None, error: Exception | None = None) -> None:
        """Configure the fake to return a frame or raise an error."""
        self.frame = frame
        self.error = error
        self.requested_fixture_ids: list[int] = []

    def fetch_fixture_lineups(self, fixture_id: int) -> pd.DataFrame:
        """Return the configured frame or raise the configured error."""
        self.requested_fixture_ids.append(fixture_id)
        if self.error is not None:
            raise self.error
        return self.frame if self.frame is not None else pd.DataFrame()


def _base_payload(**overrides: object) -> MatchContextPayload:
    """Build a minimal payload, applying any field overrides."""
    fields = {
        "match_date": datetime(2026, 7, 1, tzinfo=timezone.utc),
        "home_team_name": "Portugal",
        "away_team_name": "Croatia",
        "fixture_id": 555,
    }
    fields.update(overrides)
    return MatchContextPayload(**fields)


class LivePredictionTestCase(unittest.TestCase):
    """Validate live lineup enrichment of prediction payloads."""

    def test_returns_payload_unchanged_without_fixture_id(self) -> None:
        """No `fixture_id` means no live lookup is attempted."""
        payload = _base_payload(fixture_id=None)
        source = FakeLineupSource(frame=_lineup_rows(555))
        result = enrich_payload_with_live_lineups(payload, football_api_source=source)
        self.assertEqual(result, payload)
        self.assertEqual(source.requested_fixture_ids, [])

    def test_fills_unset_lineup_fields_from_live_lineups(self) -> None:
        """Unset lineup fields should be populated from matched live lineups."""
        payload = _base_payload()
        source = FakeLineupSource(frame=_lineup_rows(555))
        result = enrich_payload_with_live_lineups(
            payload, football_api_source=source, player_reference=_player_reference()
        )
        self.assertEqual(source.requested_fixture_ids, [555])
        self.assertAlmostEqual(result.home_lineup_average_age, 25.0)
        self.assertAlmostEqual(result.away_lineup_average_age, 30.0)

    def test_does_not_override_explicit_payload_values(self) -> None:
        """Fields already supplied by the caller must survive enrichment untouched."""
        payload = _base_payload(home_lineup_average_age=99.0)
        source = FakeLineupSource(frame=_lineup_rows(555))
        result = enrich_payload_with_live_lineups(
            payload, football_api_source=source, player_reference=_player_reference()
        )
        self.assertEqual(result.home_lineup_average_age, 99.0)
        self.assertAlmostEqual(result.away_lineup_average_age, 30.0)

    def test_swaps_home_away_when_lineup_order_is_reversed(self) -> None:
        """Lineups listing the away team first must not flip home/away stats."""
        payload = _base_payload()
        source = FakeLineupSource(frame=_lineup_rows(555, reversed_order=True))
        result = enrich_payload_with_live_lineups(
            payload, football_api_source=source, player_reference=_player_reference()
        )
        self.assertAlmostEqual(result.home_lineup_average_age, 25.0)
        self.assertAlmostEqual(result.away_lineup_average_age, 30.0)

    def test_skips_enrichment_on_team_name_mismatch(self) -> None:
        """Lineup teams that match neither side of the payload are ignored.

        Uses a reference/lineup pair for "Spain"/"France" so the lineups
        still match (matched_rows > 0) and produce real payload updates --
        the point of this case is the team-name cross-check rejecting them,
        not an empty enrichment result.
        """
        payload = _base_payload()
        rows = _team_rows(555, "Spain", _PORTUGAL_ROSTER) + _team_rows(555, "France", _CROATIA_ROSTER)
        mismatched_reference = _player_reference().assign(
            team_name=lambda frame: frame["team_name"].replace(
                {"Portugal": "Spain", "Croatia": "France"}
            )
        )
        source = FakeLineupSource(frame=pd.DataFrame(rows))
        result = enrich_payload_with_live_lineups(
            payload, football_api_source=source, player_reference=mismatched_reference
        )
        self.assertIsNone(result.home_lineup_average_age)
        self.assertIsNone(result.away_lineup_average_age)

    def test_returns_payload_unchanged_on_fetch_error(self) -> None:
        """A live-fetch failure must not break prediction."""
        payload = _base_payload()
        source = FakeLineupSource(error=RuntimeError("boom"))
        result = enrich_payload_with_live_lineups(payload, football_api_source=source)
        self.assertEqual(result, payload)

    def test_returns_payload_unchanged_on_empty_lineups(self) -> None:
        """Empty lineup responses leave the payload untouched."""
        payload = _base_payload()
        source = FakeLineupSource(frame=pd.DataFrame())
        result = enrich_payload_with_live_lineups(payload, football_api_source=source)
        self.assertEqual(result, payload)


if __name__ == "__main__":
    unittest.main()
