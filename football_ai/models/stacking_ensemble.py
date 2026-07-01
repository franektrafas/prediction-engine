"""Stacking ensemble model."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_ai.models.base import BaseModelWrapper, _require_dependency


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

    def build_estimator(self) -> Any:
        """Create the stacking estimator."""
        ensemble = _require_dependency("sklearn.ensemble")
        linear_model = _require_dependency("sklearn.linear_model")
        if self.task_type in {"binary", "multiclass"}:
            final_estimator = linear_model.LogisticRegression(max_iter=2000)
            return ensemble.StackingClassifier(
                estimators=[(model.model_name, model.estimator) for model in self.base_models],
                final_estimator=final_estimator,
                stack_method="predict_proba",
                passthrough=True,
            )
        final_estimator = linear_model.Ridge(alpha=1.0)
        return ensemble.StackingRegressor(
            estimators=[(model.model_name, model.estimator) for model in self.base_models],
            final_estimator=final_estimator,
            passthrough=True,
        )

    def build_pipeline(self, categorical_columns: list[str]) -> Any:
        """Return a direct estimator because base models already contain preprocessors."""
        del categorical_columns
        return self.build_estimator()

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
    ) -> "StackingEnsembleModel":
        """Fit the stacking ensemble on the original feature matrix."""
        del categorical_columns
        if any(model.estimator is None for model in self.base_models):
            raise ValueError("All base models must be fitted before the ensemble.")
        self.feature_columns = list(features.columns)
        self.estimator = self.build_estimator()
        self.estimator.fit(features, target)
        return self

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return an empty space because the ensemble uses base-model tuning."""
        return {}
