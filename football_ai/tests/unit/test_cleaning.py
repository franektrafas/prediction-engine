"""Tests for cleaning utilities."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.data.cleaning import clean_matches


class CleaningPipelineTestCase(unittest.TestCase):
    """Test the data cleaning pipeline."""

    def test_clean_matches_removes_duplicates_and_missing_rows(self) -> None:
        """The cleaning pipeline should normalize rows and remove invalid records."""
        frame = pd.DataFrame(
            [
                {
                    "match_date": "2024-01-01",
                    "home_team_name": "USA",
                    "away_team_name": "England",
                },
                {
                    "match_date": "2024-01-01",
                    "home_team_name": "United States of America",
                    "away_team_name": "England",
                },
                {
                    "match_date": None,
                    "home_team_name": "France",
                    "away_team_name": "Germany",
                },
            ]
        )
        cleaned, report = clean_matches(frame)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(cleaned.iloc[0]["home_team_name"], "United States")
        self.assertEqual(report.removed_duplicates, 1)
        self.assertEqual(report.rows_with_missing_required_values, 1)


if __name__ == "__main__":
    unittest.main()
