"""Chronological dataset splitters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd


@dataclass(slots=True)
class SplitIndices:
    """Index sets for chronological splitting."""

    train_indices: np.ndarray
    validation_indices: np.ndarray
    test_indices: np.ndarray


@dataclass(slots=True)
class ChronologicalSplitConfig:
    """Configuration for train/validation/test splitting."""

    date_column: str = "match_date"
    train_fraction: float = 0.7
    validation_fraction: float = 0.15
    test_fraction: float = 0.15


def chronological_train_validation_test_split(
    frame: pd.DataFrame,
    config: ChronologicalSplitConfig | None = None,
) -> SplitIndices:
    """Split a DataFrame chronologically into train, validation, and test sets."""
    split_config = config or ChronologicalSplitConfig()
    if not np.isclose(
        split_config.train_fraction
        + split_config.validation_fraction
        + split_config.test_fraction,
        1.0,
    ):
        raise ValueError("Train, validation, and test fractions must sum to 1.0.")

    ordered = frame.sort_values(split_config.date_column).reset_index()
    total_rows = len(ordered)
    train_end = int(total_rows * split_config.train_fraction)
    validation_end = train_end + int(total_rows * split_config.validation_fraction)
    return SplitIndices(
        train_indices=ordered.loc[: train_end - 1, "index"].to_numpy(),
        validation_indices=ordered.loc[train_end: validation_end - 1, "index"].to_numpy(),
        test_indices=ordered.loc[validation_end:, "index"].to_numpy(),
    )


def rolling_positional_splits(
    n_rows: int,
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield rolling train/validation splits by row position.

    Use this when the rows are already guaranteed to be in chronological
    order (earliest first) but there is no explicit date column to sort by --
    for example when building a bare ``cv`` iterable for a third-party API
    such as scikit-learn's ``StackingClassifier``/``StackingRegressor``,
    which only accept ``(train_indices, test_indices)`` positional arrays.

    Callers are responsible for the chronological-order guarantee; this
    function does not look at dates at all, it only slices positions.
    """
    fold_size = n_rows // (n_splits + 1)
    if fold_size <= 0:
        raise ValueError("Not enough rows for the requested number of splits.")

    for split_index in range(1, n_splits + 1):
        train_end = fold_size * split_index
        validation_end = min(fold_size * (split_index + 1), n_rows)
        train_indices = np.arange(train_end)
        validation_indices = np.arange(train_end, validation_end)
        if len(validation_indices) == 0:
            continue
        yield train_indices, validation_indices


def rolling_time_series_splits(
    frame: pd.DataFrame,
    date_column: str = "match_date",
    n_splits: int = 5,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield rolling train/validation splits in chronological order."""
    ordered = frame.sort_values(date_column).reset_index()
    total_rows = len(ordered)
    for positional_train, positional_validation in rolling_positional_splits(
        total_rows, n_splits=n_splits
    ):
        train_indices = ordered.loc[positional_train, "index"].to_numpy()
        validation_indices = ordered.loc[positional_validation, "index"].to_numpy()
        yield train_indices, validation_indices
