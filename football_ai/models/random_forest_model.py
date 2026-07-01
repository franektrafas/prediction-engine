"""Random Forest model wrapper."""

from __future__ import annotations

from typing import Any

from football_ai.models.base import BaseModelWrapper, _require_dependency


class RandomForestModel(BaseModelWrapper):
    """Random Forest classifier or regressor."""

    def build_estimator(self) -> Any:
        """Create a Random Forest estimator."""
        ensemble = _require_dependency("sklearn.ensemble")
        params = {
            "n_estimators": self.hyperparameters.get("n_estimators", 500),
            "max_depth": self.hyperparameters.get("max_depth", 10),
            "min_samples_split": self.hyperparameters.get("min_samples_split", 4),
            "min_samples_leaf": self.hyperparameters.get("min_samples_leaf", 2),
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.task_type in {"binary", "multiclass"}:
            return ensemble.RandomForestClassifier(**params)
        return ensemble.RandomForestRegressor(**params)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return the Optuna search space."""
        return {
            "n_estimators": (200, 1200),
            "max_depth": (4, 20),
            "min_samples_split": (2, 16),
            "min_samples_leaf": (1, 8),
        }
