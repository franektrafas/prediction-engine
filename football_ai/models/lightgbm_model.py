"""LightGBM model wrapper."""

from __future__ import annotations

from typing import Any

from football_ai.models.base import BaseModelWrapper, _require_dependency


class LightGBMModel(BaseModelWrapper):
    """LightGBM classifier or regressor."""

    def build_estimator(self) -> Any:
        """Create a LightGBM estimator."""
        lightgbm = _require_dependency("lightgbm")
        params = {
            "n_estimators": self.hyperparameters.get("n_estimators", 600),
            "learning_rate": self.hyperparameters.get("learning_rate", 0.03),
            "num_leaves": self.hyperparameters.get("num_leaves", 31),
            "max_depth": self.hyperparameters.get("max_depth", -1),
            "subsample": self.hyperparameters.get("subsample", 0.9),
            "colsample_bytree": self.hyperparameters.get("colsample_bytree", 0.9),
            "reg_alpha": self.hyperparameters.get("reg_alpha", 0.1),
            "reg_lambda": self.hyperparameters.get("reg_lambda", 0.5),
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": -1,
        }
        if self.task_type in {"binary", "multiclass"}:
            if self.task_type == "multiclass":
                params["objective"] = "multiclass"
                params["class_weight"] = "balanced"
            else:
                params["objective"] = "binary"
                params["class_weight"] = "balanced"
            return lightgbm.LGBMClassifier(**params)
        params["objective"] = "regression"
        return lightgbm.LGBMRegressor(**params)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return the Optuna search space."""
        return {
            "n_estimators": (300, 1500),
            "learning_rate": (0.01, 0.2),
            "num_leaves": (15, 127),
            "max_depth": (3, 14),
            "subsample": (0.6, 1.0),
            "colsample_bytree": (0.6, 1.0),
            "reg_alpha": (0.0, 3.0),
            "reg_lambda": (0.0, 3.0),
        }
