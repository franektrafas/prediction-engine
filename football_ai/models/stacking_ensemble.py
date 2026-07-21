"""Stacking ensemble model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_ai.models.base import BaseModelWrapper, _require_dependency
from football_ai.training.splitters import rolling_positional_splits


# scikit-learn's StackingClassifier/StackingRegressor generate the
# meta-learner's training data by cross-validating each base estimator
# internally (`cv=` defaults to a 5-fold (Stratified)KFold). For a football
# dataset ordered by match date that default is a random split in disguise --
# exactly what this project's ML philosophy forbids -- because it can use
# chronologically later matches to help predict earlier ones inside a fold.
# We replace it with rolling, forward-only folds built from row position.
_DEFAULT_STACKING_CV_SPLITS = 5


def _build_chronological_cv(n_rows: int) -> list[tuple[Any, Any]] | None:
    """Build rolling (train, validation) index folds for scikit-learn's `cv=`.

    Falls back to `None` (scikit-learn's own default) only when there is not
    enough data to build even two rolling folds -- e.g. tiny unit-test
    fixtures. Real training data should always have enough rows for this to
    succeed, so silently falling back should never happen outside of tests.
    """
    max_splits = min(_DEFAULT_STACKING_CV_SPLITS, max(0, n_rows // 20))
    for n_splits in range(max_splits, 1, -1):
        try:
            folds = list(rolling_positional_splits(n_rows, n_splits=n_splits))
        except ValueError:
            continue
        if folds:
            return folds
    return None


class StackingEnsembleModel(BaseModelWrapper):
    """Stacking ensemble built on top of fitted base models."""

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

    def build_estimator(self, cv: list[tuple[Any, Any]] | None = None) -> Any:
        """Create the stacking estimator.

        `cv` should be a chronologically-ordered list of (train, validation)
        index folds (see `_build_chronological_cv`); passing `None` lets
        scikit-learn fall back to its own default (random) folds, which is
        only acceptable for tiny fixtures where a time-based fold cannot be
        built.
        """
        ensemble = _require_dependency("sklearn.ensemble")
        linear_model = _require_dependency("sklearn.linear_model")
        if self.task_type in {"binary", "multiclass"}:
            final_estimator = linear_model.LogisticRegression(max_iter=2000)
            return ensemble.StackingClassifier(
                estimators=[(model.model_name, model.estimator) for model in self.base_models],
                final_estimator=final_estimator,
                stack_method="predict_proba",
                passthrough=True,
                cv=cv,
            )
        final_estimator = linear_model.Ridge(alpha=1.0)
        return ensemble.StackingRegressor(
            estimators=[(model.model_name, model.estimator) for model in self.base_models],
            final_estimator=final_estimator,
            passthrough=True,
            cv=cv,
        )

    def build_pipeline(
        self,
        categorical_columns: list[str],
        numeric_columns: list[str],
    ) -> Any:
        """Return a direct estimator because base models already contain preprocessors."""
        del categorical_columns
        del numeric_columns
        return self.build_estimator()

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
        del categorical_columns
        if any(model.estimator is None for model in self.base_models):
            raise ValueError("All base models must be fitted before the ensemble.")
        self.feature_columns = list(features.columns)
        cv = _build_chronological_cv(len(features))
        self.estimator = self.build_estimator(cv=cv)
        self.estimator.fit(features, target)
        return self

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return an empty space because the ensemble uses base-model tuning."""
        return {}
