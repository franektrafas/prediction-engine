"""Dataset preparation helpers for model training."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from football_ai.features.schemas import FeatureBuildResult
from football_ai.training.splitters import (
    ChronologicalSplitConfig,
    SplitIndices,
    chronological_train_validation_test_split,
)


@dataclass(slots=True)
class TrainingDataset:
    """Prepared dataset slices for a single target."""

    features: pd.DataFrame
    target: pd.Series
    split_indices: SplitIndices
    feature_columns: list[str]
    categorical_columns: list[str]


def build_training_dataset(
    feature_result: FeatureBuildResult,
    target_name: str,
    split_config: ChronologicalSplitConfig | None = None,
) -> TrainingDataset:
    """Build a training dataset for a target variable."""
    if target_name not in feature_result.targets:
        raise KeyError(f"Unknown target: {target_name}")
    target = feature_result.targets[target_name]
    valid_rows = target.notna()
    filtered_frame = feature_result.frame.loc[valid_rows].reset_index(drop=True)
    aligned_target = target.loc[valid_rows].reset_index(drop=True)
    features = filtered_frame.loc[:, feature_result.feature_columns].copy()
    split_indices = chronological_train_validation_test_split(
        filtered_frame,
        config=split_config,
    )
    return TrainingDataset(
        features=features,
        target=aligned_target,
        split_indices=split_indices,
        feature_columns=feature_result.feature_columns,
        categorical_columns=feature_result.categorical_columns,
    )
