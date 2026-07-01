"""Logistic Regression model wrapper."""

from __future__ import annotations

from typing import Any

from football_ai.models.base import BaseModelWrapper, _require_dependency


class LogisticRegressionModel(BaseModelWrapper):
    """Multiclass or binary logistic regression."""

    def build_estimator(self) -> Any:
        """Create a logistic regression estimator."""
        if self.task_type == "regression":
            raise ValueError("LogisticRegressionModel cannot be used for regression tasks.")
        linear_model = _require_dependency("sklearn.linear_model")
        params = {
            "C": self.hyperparameters.get("C", 1.0),
            "max_iter": self.hyperparameters.get("max_iter", 2000),
            "solver": self.hyperparameters.get("solver", "lbfgs"),
            "class_weight": self.hyperparameters.get("class_weight", "balanced"),
            "multi_class": "auto",
        }
        return linear_model.LogisticRegression(**params)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return the Optuna search space."""
        return {
            "C": (1e-3, 30.0),
            "max_iter": (500, 4000),
        }
