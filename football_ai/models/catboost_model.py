"""CatBoost model wrapper."""

from __future__ import annotations

from typing import Any

from football_ai.models.base import BaseModelWrapper, _require_dependency


class CatBoostModel(BaseModelWrapper):
    """CatBoost classifier or regressor."""

    def build_estimator(self) -> Any:
        """Create a CatBoost estimator."""
        catboost = _require_dependency("catboost")
        params = {
            "iterations": self.hyperparameters.get("iterations", 800),
            "learning_rate": self.hyperparameters.get("learning_rate", 0.03),
            "depth": self.hyperparameters.get("depth", 8),
            "l2_leaf_reg": self.hyperparameters.get("l2_leaf_reg", 5.0),
            "loss_function": "Logloss",
            "verbose": False,
            "random_seed": 42,
        }
        if self.task_type == "multiclass":
            params["loss_function"] = "MultiClass"
            return catboost.CatBoostClassifier(**params)
        if self.task_type == "binary":
            return catboost.CatBoostClassifier(**params)
        params["loss_function"] = "RMSE"
        return catboost.CatBoostRegressor(**params)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return the Optuna search space."""
        return {
            "iterations": (300, 1500),
            "learning_rate": (0.01, 0.2),
            "depth": (4, 10),
            "l2_leaf_reg": (1.0, 10.0),
        }
