"""Tests for chronological splitters."""

from __future__ import annotations

import unittest

import pandas as pd

from football_ai.training.splitters import (
    ChronologicalSplitConfig,
    chronological_train_validation_test_split,
    rolling_time_series_splits,
)


class SplitterTestCase(unittest.TestCase):
    """Validate chronological splitting behavior."""

    def test_train_validation_test_split_is_chronological(self) -> None:
        """Indices should follow chronological ordering."""
        frame = pd.DataFrame(
            {
                "match_date": pd.date_range("2024-01-01", periods=20, freq="D"),
            }
        )
        split = chronological_train_validation_test_split(
            frame,
            ChronologicalSplitConfig(train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2),
        )
        self.assertEqual(len(split.train_indices), 12)
        self.assertEqual(len(split.validation_indices), 4)
        self.assertEqual(len(split.test_indices), 4)
        self.assertLess(max(split.train_indices), min(split.validation_indices))
        self.assertLess(max(split.validation_indices), min(split.test_indices))

    def test_rolling_time_series_splits_returns_requested_folds(self) -> None:
        """Rolling splits should produce multiple chronological folds."""
        frame = pd.DataFrame(
            {
                "match_date": pd.date_range("2024-01-01", periods=24, freq="D"),
            }
        )
        splits = list(rolling_time_series_splits(frame, n_splits=4))
        self.assertEqual(len(splits), 4)
        for train_indices, validation_indices in splits:
            self.assertLess(max(train_indices), min(validation_indices))


if __name__ == "__main__":
    unittest.main()
