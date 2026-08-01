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

_GOALSCORERS_CSV = """date,home_team,away_team,team,scorer,minute,own_goal,penalty
1974-06-14,Germany,Chile,Germany,Gerd Muller,20.0,False,False
2023-06-17,Poland,Germany,Germany,Kai Havertz,55.0,False,True
1899-01-01,Nowhereland,Somewhereland,Nowhereland,Nobody,10.0,False,False
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

    def test_build_records_frame_merges_espn_stats_alongside_elo(self) -> None:
        """ESPN stats layer in independently of Elo -- martj42 still wins identity/score."""
        elo_source = pd.DataFrame(
            [
                {
                    "match_date": "2023-06-17",
                    "home_team_name": "Poland",
                    "away_team_name": "Germany",
                    "home_elo": 1650.0,
                    "away_elo": 2000.0,
                }
            ]
        )
        espn_source = pd.DataFrame(
            [
                {
                    "match_date": "2023-06-17",
                    "home_team_name": "Poland",
                    "away_team_name": "Germany",
                    "home_score": 99,  # deliberately wrong -- must NOT win over martj42's score
                    "away_score": 99,
                    "home_shots": 12.0,
                    "away_shots": 9.0,
                    "home_corners": 5.0,
                }
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            elo_path = Path(temp_dir) / "elo.parquet"
            espn_path = Path(temp_dir) / "espn.parquet"
            write_table(elo_source, elo_path)
            write_table(espn_source, espn_path)
            with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
                records_frame = self.source.build_records_frame(
                    elo_source_path=elo_path, espn_source_path=espn_path
                )

        self.assertEqual(len(records_frame), 3)
        overlapping_row = records_frame.loc[records_frame["home_team_name"] == "Poland"].iloc[0]
        self.assertEqual(overlapping_row["home_score"], 0)  # martj42's real score, not ESPN's 99
        self.assertEqual(overlapping_row["home_elo"], 1650.0)  # Elo enrichment still applied
        self.assertEqual(overlapping_row["home_shots"], 12.0)  # ESPN enrichment applied too
        self.assertEqual(overlapping_row["home_corners"], 5.0)

    def test_build_records_frame_with_missing_espn_file_path_degrades_gracefully(self) -> None:
        """A non-existent ESPN path shouldn't raise -- same contract as a missing Elo path."""
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            records_frame = self.source.build_records_frame(espn_source_path="/nonexistent/path.parquet")
        self.assertEqual(len(records_frame), 3)

    def test_build_records_frame_without_elo_source_returns_results_only(self) -> None:
        """A missing/unset Elo source should degrade gracefully, not raise."""
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            records_frame = self.source.build_records_frame(elo_source_path=None)
        self.assertEqual(len(records_frame), 3)
        self.assertNotIn("home_elo", records_frame.columns)

    def test_match_date_is_utc_aware_regardless_of_whether_elo_enrichment_ran(self) -> None:
        """Regression test: match_date used to be tz-naive without Elo, tz-aware with it.

        `insights/`'s `as_of_date` filtering compares against `tz="UTC"`
        timestamps and raises `TypeError` on a naive frame -- both
        build_records_frame() code paths (with and without a real Elo
        source) must return the same, UTC-aware dtype.
        """
        with mock.patch.object(self.source, "fetch_text", return_value=_RESULTS_CSV):
            without_elo = self.source.build_records_frame(elo_source_path=None)
            with_missing_elo_path = self.source.build_records_frame(elo_source_path="/nonexistent/path.parquet")
        self.assertEqual(str(without_elo["match_date"].dt.tz), "UTC")
        self.assertEqual(str(with_missing_elo_path["match_date"].dt.tz), "UTC")

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

    def test_normalize_goalscorers_links_to_the_real_match_id(self) -> None:
        """Two of three goalscorer rows share a (date, home, away) with results.csv and must get that match's match_id."""
        results_raw = pd.read_csv(pd.io.common.StringIO(_RESULTS_CSV))
        records_frame = self.source.normalize_results(results_raw)
        goalscorers_raw = pd.read_csv(pd.io.common.StringIO(_GOALSCORERS_CSV))

        goalscorers_frame = self.source.normalize_goalscorers(goalscorers_raw, records_frame)

        # The third row (Nowhereland vs Somewhereland, 1899) has no matching result -- dropped, not kept with a null id.
        self.assertEqual(len(goalscorers_frame), 2)
        self.assertNotIn("Nobody", goalscorers_frame["scorer"].to_list())

        germany_chile_id = records_frame.loc[
            (records_frame["home_team_name"] == "Germany") & (records_frame["away_team_name"] == "Chile"),
            "match_id",
        ].iloc[0]
        muller_row = goalscorers_frame.loc[goalscorers_frame["scorer"] == "Gerd Muller"].iloc[0]
        self.assertEqual(int(muller_row["match_id"]), int(germany_chile_id))
        self.assertEqual(muller_row["team_name"], "Germany")

        havertz_row = goalscorers_frame.loc[goalscorers_frame["scorer"] == "Kai Havertz"].iloc[0]
        self.assertTrue(bool(havertz_row["penalty"]))
        self.assertFalse(bool(havertz_row["own_goal"]))


if __name__ == "__main__":
    unittest.main()
