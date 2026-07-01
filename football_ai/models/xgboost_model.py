"""XGBoost model wrapper."""

from __future__ import annotations

from typing import Any

from football_ai.models.base import BaseModelWrapper, _require_dependency


class XGBoostModel(BaseModelWrapper):
    """XGBoost classifier or regressor."""

    def build_estimator(self) -> Any:
        """Create an XGBoost estimator."""
        xgboost = _require_dependency("xgboost")
        params = {
            "n_estimators": self.hyperparameters.get("n_estimators", 800),
            "learning_rate": self.hyperparameters.get("learning_rate", 0.03),
            "max_depth": self.hyperparameters.get("max_depth", 6),
            "subsample": self.hyperparameters.get("subsample", 0.9),
            "colsample_bytree": self.hyperparameters.get("colsample_bytree", 0.9),
            "reg_alpha": self.hyperparameters.get("reg_alpha", 0.1),
            "reg_lambda": self.hyperparameters.get("reg_lambda", 1.0),
            "random_state": 42,
            "n_jobs": -1,
        }
        if self.task_type == "multiclass":
            params["objective"] = "multi:softprob"
            return xgboost.XGBClassifier(**params)
        if self.task_type == "binary":
            params["objective"] = "binary:logistic"
            return xgboost.XGBClassifier(**params)
        params["objective"] = "reg:squarederror"
        return xgboost.XGBRegressor(**params)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return the Optuna search space."""
        return {
            "n_estimators": (300, 1500),
            "learning_rate": (0.01, 0.2),
            "max_depth": (3, 12),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0.0, 3.0),
            "reg_lambda": (0.0, 5.0),
        }
