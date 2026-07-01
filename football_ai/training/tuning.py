"""Optuna-based hyperparameter tuning."""

from __future__ import annotations

from typing import Any

import pandas as pd

from football_ai.evaluation.metrics import evaluate_predictions
from football_ai.models.base import BaseModelWrapper, CLASSIFICATION_TASKS, _require_dependency
from football_ai.training.splitters import rolling_time_series_splits


def _sample_hyperparameters(
    trial: Any,
    model: BaseModelWrapper,
) -> dict[str, Any]:
    """Sample hyperparameters from a model-defined search space."""
    sampled: dict[str, Any] = {}
    for name, bounds in model.hyperparameter_space().items():
        low, high = bounds
        if isinstance(low, int) and isinstance(high, int):
            sampled[name] = trial.suggest_int(name, low, high)
        else:
            sampled[name] = trial.suggest_float(name, float(low), float(high))
    return sampled


def tune_model(
    model: BaseModelWrapper,
    features: pd.DataFrame,
    target: pd.Series,
    categorical_columns: list[str],
    date_column: str = "match_date",
    n_trials: int = 20,
) -> dict[str, Any]:
    """Tune a model with Optuna using rolling time-series validation."""
    optuna = _require_dependency("optuna")
    score_name = "f1" if model.task_type in CLASSIFICATION_TASKS else "rmse"
    direction = "maximize" if model.task_type in CLASSIFICATION_TASKS else "minimize"

    def objective(trial: Any) -> float:
        """Optimization objective."""
        hyperparameters = _sample_hyperparameters(trial, model)
        scores: list[float] = []
        for train_indices, validation_indices in rolling_time_series_splits(
            features.assign(**{date_column: pd.RangeIndex(len(features))}),
            date_column=date_column,
            n_splits=4,
        ):
            candidate = model.__class__(
                model_name=model.model_name,
                task_type=model.task_type,
                target_name=model.target_name,
                hyperparameters=hyperparameters,
                version=model.version,
            )
            candidate.fit(
                features.iloc[train_indices],
                target.iloc[train_indices],
                categorical_columns=categorical_columns,
            )
            predictions = candidate.predict(features.iloc[validation_indices])
            probabilities = None
            if model.task_type in CLASSIFICATION_TASKS:
                try:
                    probabilities = candidate.predict_proba(features.iloc[validation_indices])
                except AttributeError:
                    probabilities = None
            metrics = evaluate_predictions(
                task_type=model.task_type,
                target=target.iloc[validation_indices],
                predictions=predictions,
                probabilities=probabilities,
            )
            scores.append(metrics[score_name])
        return float(sum(scores) / len(scores))

    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials)
    return dict(study.best_params)
