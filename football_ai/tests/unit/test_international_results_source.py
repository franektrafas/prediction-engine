"""Tests for the full-history international results importer (martj42)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd

from football_ai.data.io import write_table
from football_ai.scrapers.sources.international_results import InternationalResultsSource


_RESULTS_CSV = """date,home_team,away_team,home_score,away_score,tournament,city,country,neutral
1974-06-14,Germany,Chile,1,0,FIFA World Cup,Berlin,Germany,FALSE
1974-06-15,German DR,Australia,2,0,FIFA World Cup,Hamburg,Germany,FALSE
2023-06-17,Poland,Germany,0,1,Friendly,Warsaw,Poland,FALSE
"""


class InternationalResultsSourceTestCase(unittest.TestCase):
    """Validate fetch/normalize/merge behavior, fully offline."""

    def setUp(self) -> None:
        """Build a fresh source for each test."""
        self.source = InternationalResultsSource()

    def test_fetch_results_parses_the_real_csv_shape(self) -> None:
        """fetch_results should parse martj42's exact real column layout, mocked offline."""
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            raw = self.source.fetch_results()
        self.assertEqual(len(raw), 3)
        self.assertListEqual(
            list(raw.columns),
            ["date", "home_team", "away_team", "home_score", "away_score", "tournament", "city", "country", "neutral"],
        )

    def test_normalize_results_maps_onto_the_project_wide_schema(self) -> None:
        """Columns should map 1:1 onto match_date/home_team_name/away_team_name/... with neutral as bool."""
        raw = pd.read_csv(pd.io.common.StringIO(_RESULTS_CSV))
        normalized = self.source.normalize_results(raw)
        self.assertListEqual(
            list(normalized.columns),
            ["match_id", "match_date", "home_team_name", "away_team_name", "home_score", "away_score", "tournament", "city", "country", "neutral_venue"],
        )
        self.assertEqual(len(normalized), 3)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(normalized["match_date"]))
        self.assertEqual(normalized["neutral_venue"].dtype, bool)
        self.assertListEqual(sorted(normalized["match_id"].to_list()), [1, 2, 3])
        self.assertEqual(normalized["match_id"].nunique(), 3)
        # East Germany ("German DR") and West Germany ("Germany") must survive as distinct
        # home_team_name values here -- resolving them is canonicalize_team_name's job at
        # merge/feature-build time, not something normalize_results should collapse early.
        self.assertIn("German DR", normalized["home_team_name"].to_list())
        self.assertIn("Germany", normalized["home_team_name"].to_list())

    def test_build_records_frame_uses_martj42_for_identity_and_supplemental_for_elo(self) -> None:
        """martj42 wins identity/score; the existing Elo file only fills in Elo columns, per plan."""
        elo_source = pd.DataFrame(
            [
                {
                    "match_date": "2023-06-17",
                    "home_team_name": "Poland",
                    "away_team_name": "Germany",
                    "home_score": 99,  # deliberately wrong -- must NOT win over martj42's score
                    "away_score": 99,
                    "home_elo": 1650.0,
                    "away_elo": 2000.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_path = Path(temp_dir) / "elo.parquet"
            write_table(elo_source, elo_path)
            with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
                records_frame = self.source.build_records_frame(elo_source_path=elo_path)

        self.assertEqual(len(records_frame), 3)
        overlapping_row = records_frame.loc[records_frame["home_team_name"] == "Poland"].iloc[0]
        self.assertEqual(overlapping_row["home_score"], 0)  # martj42's real score, not the Elo file's 99
        self.assertEqual(overlapping_row["away_score"], 1)
        self.assertEqual(overlapping_row["home_elo"], 1650.0)  # Elo enrichment did apply
        self.assertEqual(overlapping_row["away_elo"], 2000.0)
        non_overlapping_row = records_frame.loc[records_frame["home_team_name"] == "Germany"].iloc[0]
        self.assertTrue(pd.isna(non_overlapping_row.get("home_elo")))

    def test_build_records_frame_without_elo_source_returns_results_only(self) -> None:
        """A missing/unset Elo source should degrade gracefully, not raise."""
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            records_frame = self.source.build_records_frame(elo_source_path=None)
        self.assertEqual(len(records_frame), 3)
        self.assertNotIn("home_elo", records_frame.columns)

    def test_build_records_frame_with_missing_elo_file_path_degrades_gracefully(self) -> None:
        """A non-existent Elo path (e.g. before first real training run) shouldn't raise."""
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            records_frame = self.source.build_records_frame(elo_source_path="/nonexistent/path.parquet")
        self.assertEqual(len(records_frame), 3)

    def test_build_training_frame_filters_by_cutoff_year(self) -> None:
        """Only the 2023 row should survive a 1990 cutoff; the two 1974 rows must not."""
        raw = pd.read_csv(pd.io.common.StringIO(_RESULTS_CSV))
        records_frame = self.source.normalize_results(raw)
        training_frame = self.source.build_training_frame(records_frame, cutoff_year=1990)
        self.assertEqual(len(training_frame), 1)
        self.assertEqual(training_frame.iloc[0]["home_team_name"], "Poland")


if __name__ == "__main__":
    unittest.main()
