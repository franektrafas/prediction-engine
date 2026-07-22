"""Stacking ensemble model."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import BaseModelWrapper, _require_dependency
from football_ai.training.splitters import rolling_positional_splits


LOGGER = logging.getLogger(__name__)

# scikit-learn's StackingClassifier/StackingRegressor generate the
# meta-learner's training data via `cross_val_predict(cv=...)`, which raises
# "cross_val_predict only works for partitions" unless the cv's validation
# folds cover every row of the input exactly once. This project's leak-safe
# rolling CV (`rolling_positional_splits`) is forward-only by design: its
# earliest rows can never sit in a validation fold, because there is no prior
# history to train that fold on -- and there is no fold arrangement that fixes
# this without either training on an empty set or leaking future rows
# backward into predicting past ones. So instead of handing rolling folds to
# scikit-learn's stacking classes, this module builds the meta-learner's
# out-of-fold features by hand (`_out_of_fold_meta_features`), dropping the
# leading rows that structurally can't be validated from the meta-learner's
# own training set -- they're still used as training data for later folds,
# just never scored -- and fits/predicts through a plain `final_estimator`
# instead of `StackingClassifier`/`StackingRegressor`.
_DEFAULT_STACKING_CV_SPLITS = 5


def _rolling_folds(n_rows: int, max_splits: int = _DEFAULT_STACKING_CV_SPLITS) -> list[tuple[Any, Any]]:
    """Best-effort rolling CV folds; empty if there isn't enough data for at least two."""
    upper_bound = min(max_splits, max(0, n_rows // 20))
    for n_splits in range(upper_bound, 1, -1):
        try:
            folds = list(rolling_positional_splits(n_rows, n_splits=n_splits))
        except ValueError:
            continue
        if folds:
            return folds
    return []


def _base_model_outputs(estimator: Any, features: pd.DataFrame, task_type: str) -> np.ndarray:
    """Return one column (regression) or per-class probability columns (classification)."""
    if task_type in {"binary", "multiclass"}:
        probabilities = np.asarray(estimator.predict_proba(features))
        if task_type == "binary" and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, [1]]
        return probabilities
    predictions = np.asarray(estimator.predict(features))
    return predictions.reshape(-1, 1)


@dataclass(slots=True)
class FittedStackingEstimator:
    """Picklable, `predict()`/`predict_proba()`-compatible fitted ensemble.

    Holds only what `predictions.service.PredictionService` actually touches
    on a loaded artifact's `estimator` -- fitted base pipelines, the fitted
    meta-learner, and enough metadata to rebuild meta-features for new rows
    the same way training built them.
    """

    task_type: str
    base_estimators: list[Any]
    passthrough_columns: list[str]
    final_estimator: Any
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def classes_(self) -> np.ndarray:
        """Expose the meta-learner's classes for classification tasks."""
        return self.final_estimator.classes_

    def _meta_features(self, features: pd.DataFrame) -> np.ndarray:
        """Rebuild [base-model outputs, numeric passthrough] for new rows."""
        base_outputs = [
            _base_model_outputs(estimator, features, self.task_type)
            for estimator in self.base_estimators
        ]
        stacked = np.concatenate(base_outputs, axis=1)
        passthrough = features.loc[:, self.passthrough_columns].to_numpy(dtype=float)
        return np.concatenate([stacked, passthrough], axis=1)

    def predict(self, features: pd.DataFrame) -> Any:
        """Run ensemble predictions."""
        return self.final_estimator.predict(self._meta_features(features))

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Run ensemble probability predictions."""
        return self.final_estimator.predict_proba(self._meta_features(features))


class StackingEnsembleModel(BaseModelWrapper):
    """Stacking ensemble built on top of independently-fitted base models."""

    def __init__(
        self,
        task_type: str,
        target_name: str,
        base_models: list[BaseModelWrapper],
        version: str = "v1",
    ) -> None:
        """Initialize the ensemble wrapper."""
        super().__init__(
            model_name="stacking_ensemble",
            task_type=task_type,
            target_name=target_name,
            version=version,
        )
        self.base_models = base_models

    def _build_final_estimator(self) -> Any:
        """Create the meta-learner pipeline (median-impute + scale + linear model).

        The meta-features include raw numeric passthrough columns (rolling
        features etc. on their native scale, sometimes with `NaN` from short
        windows), not just base-model probabilities in [0, 1] -- so the
        meta-learner needs the same impute/scale treatment `BaseModelWrapper`
        gives every base model's numeric columns, not a bare linear model.
        """
        sklearn_pipeline = _require_dependency("sklearn.pipeline")
        sklearn_impute = _require_dependency("sklearn.impute")
        sklearn_preprocessing = _require_dependency("sklearn.preprocessing")
        linear_model = _require_dependency("sklearn.linear_model")
        estimator = (
            linear_model.LogisticRegression(max_iter=2000)
            if self.task_type in {"binary", "multiclass"}
            else linear_model.Ridge(alpha=1.0)
        )
        return sklearn_pipeline.Pipeline(
            steps=[
                ("imputer", sklearn_impute.SimpleImputer(strategy="median")),
                ("scaler", sklearn_preprocessing.StandardScaler()),
                ("estimator", estimator),
            ]
        )

    def _out_of_fold_meta_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        passthrough_columns: list[str],
    ) -> tuple[np.ndarray, pd.Series]:
        """Build leak-safe out-of-fold meta-features by refitting base-model clones per fold.

        Returns `(meta_features, meta_target)` covering only the rows that a
        rolling fold actually validated -- the earliest rows (no prior
        history to train a fold on) are excluded from the meta-learner's own
        training set entirely, not zero-filled or leaked into.
        """
        sklearn_base = _require_dependency("sklearn.base")
        folds = _rolling_folds(len(features))
        if not folds:
            LOGGER.warning(
                "Not enough rows (%d) to build leak-safe out-of-fold meta-features for "
                "target '%s'; falling back to in-sample base-model outputs. This is only "
                "expected for tiny fixtures, never for real training data.",
                len(features),
                self.target_name,
            )
            base_outputs = [
                _base_model_outputs(model.estimator, features, self.task_type)
                for model in self.base_models
            ]
            passthrough = features.loc[:, passthrough_columns].to_numpy(dtype=float)
            return np.concatenate([*base_outputs, passthrough], axis=1), target

        oof_rows: list[int] = []
        oof_base_outputs: list[list[np.ndarray]] = [[] for _ in self.base_models]
        for train_indices, validation_indices in folds:
            for model_index, model in enumerate(self.base_models):
                clone = sklearn_base.clone(model.estimator)
                clone.fit(features.iloc[train_indices], target.iloc[train_indices])
                oof_base_outputs[model_index].append(
                    _base_model_outputs(clone, features.iloc[validation_indices], self.task_type)
                )
            oof_rows.extend(validation_indices.tolist())

        order = np.argsort(oof_rows)
        row_order = np.asarray(oof_rows)[order]
        base_outputs = [np.concatenate(columns, axis=0)[order] for columns in oof_base_outputs]
        passthrough = features.iloc[row_order].loc[:, passthrough_columns].to_numpy(dtype=float)
        meta_features = np.concatenate([*base_outputs, passthrough], axis=1)
        meta_target = target.iloc[row_order]
        return meta_features, meta_target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
    ) -> "StackingEnsembleModel":
        """Fit the stacking ensemble on the original feature matrix.

        `features`/`target` must already be sorted chronologically (earliest
        row first) -- true for every caller in this project, since they are
        always sliced from `chronological_train_validation_test_split`.
        """
        if any(model.estimator is None for model in self.base_models):
            raise ValueError("All base models must be fitted before the ensemble.")
        self.feature_columns = list(features.columns)
        # Only numeric columns can passthrough into the linear meta-learner;
        # categorical columns already went through each base model's own
        # ColumnTransformer and don't belong in a raw numeric array.
        passthrough_columns = [
            column for column in features.columns if column not in categorical_columns
        ]

        meta_features, meta_target = self._out_of_fold_meta_features(
            features, target, passthrough_columns
        )
        final_estimator = self._build_final_estimator()
        final_estimator.fit(meta_features, meta_target)

        self.estimator = FittedStackingEstimator(
            task_type=self.task_type,
            base_estimators=[model.estimator for model in self.base_models],
            passthrough_columns=passthrough_columns,
            final_estimator=final_estimator,
        )
        return self

    def predict(self, features: pd.DataFrame) -> Any:
        """Run ensemble predictions."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        return self.estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Run ensemble probability predictions."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.model_name} does not support predict_proba.")
        return self.estimator.predict_proba(features)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return an empty space because the ensemble uses base-model tuning."""
        return {}
