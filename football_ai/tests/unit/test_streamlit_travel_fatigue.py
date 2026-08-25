"""Tests for streamlit_app._render_travel_fatigue (PROMPT_ZAMKNIECIE_WATKOW.md item 3).

Uses Streamlit's own `streamlit.testing.v1.AppTest` harness rather than a
plain function call: `st.*` calls silently no-op outside an active script
run context, which would hide a real Streamlit-API misuse (wrong kwarg
name, mismatched `st.columns`/`zip` count, ...) that a bare import-and-call
test cannot catch. Runs against the real, local
`international_results_full_history.parquet` + `geonamescache` data (both
file-based, no network) rather than a mocked fixture, so this also catches
a real regression in the geocoding/team-form plumbing, not just in
`_render_travel_fatigue`'s own logic.
"""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

_REPO_ROOT = str(Path(__file__).resolve().parents[3])


def _travel_fatigue_script(
    repo_root: str, home: str, away: str, match_date, venue
) -> None:
    import sys

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from football_ai.streamlit_app import _render_travel_fatigue

    _render_travel_fatigue(home, away, match_date, venue)


class RenderTravelFatigueTestCase(unittest.TestCase):
    """Real end-to-end rendering, real data, zero mocking."""

    def _run(self, home: str, away: str, match_date, venue: str | None) -> AppTest:
        """Run `_render_travel_fatigue` inside a real Streamlit script context."""
        app_test = AppTest.from_function(
            _travel_fatigue_script, args=(_REPO_ROOT, home, away, match_date, venue)
        )
        app_test.run(timeout=60)
        return app_test

    def test_real_poland_germany_fixture_renders_with_no_exception(self) -> None:
        """Real data: both teams' prior matches resolve to real, plausible distances."""
        app_test = self._run(
            "Poland",
            "Germany",
            pd.Timestamp("2025-09-01").date(),
            "Stadion Narodowy, Warsaw",
        )
        self.assertEqual(len(app_test.exception), 0)
        self.assertIn("Zmęczenie podróżą", [s.value for s in app_test.subheader])
        metrics = {m.label: m.value for m in app_test.metric}
        self.assertIn("Polska", metrics)
        self.assertIn("Niemcy", metrics)
        self.assertTrue(metrics["Polska"].endswith("km"))

    def test_no_venue_renders_nothing(self) -> None:
        """A manual/hypothetical pairing has no venue -- nothing must render."""
        app_test = self._run(
            "Poland", "Germany", pd.Timestamp("2025-09-01").date(), None
        )
        self.assertEqual(len(app_test.exception), 0)
        self.assertNotIn("Zmęczenie podróżą", [s.value for s in app_test.subheader])

    def test_unresolvable_venue_renders_nothing_not_a_crash(self) -> None:
        """A venue geocoding can't resolve must degrade silently, not fabricate."""
        app_test = self._run(
            "Poland",
            "Germany",
            pd.Timestamp("2025-09-01").date(),
            "Made Up Arena, Nowhere",
        )
        self.assertEqual(len(app_test.exception), 0)
        self.assertNotIn("Zmęczenie podróżą", [s.value for s in app_test.subheader])


if __name__ == "__main__":
    unittest.main()
