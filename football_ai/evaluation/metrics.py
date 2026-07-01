"""Model evaluation metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import CLASSIFICATION_TASKS, _require_dependency


def evaluate_predictions(
    task_type: str,
    target: pd.Series,
    predictions: Any,
    probabilities: Any | None = None,
) -> dict[str, float]:
    """Compute metrics for a task."""
    sklearn_metrics = _require_dependency("sklearn.metrics")
    metrics: dict[str, float] = {}
    if task_type in CLASSIFICATION_TASKS:
        metrics["accuracy"] = float(sklearn_metrics.accuracy_score(target, predictions))
        metrics["precision"] = float(
            sklearn_metrics.precision_score(target, predictions, average="weighted", zero_division=0)
        )
        metrics["recall"] = float(
            sklearn_metrics.recall_score(target, predictions, average="weighted", zero_division=0)
        )
        metrics["f1"] = float(
            sklearn_metrics.f1_score(target, predictions, average="weighted", zero_division=0)
        )
        if probabilities is not None:
            metrics["log_loss"] = float(sklearn_metrics.log_loss(target, probabilities))
            if task_type == "binary":
                positive_probabilities = probabilities[:, 1]
                metrics["roc_auc"] = float(
                    sklearn_metrics.roc_auc_score(target, positive_probabilities)
                )
                metrics["brier_score"] = float(
                    sklearn_metrics.brier_score_loss(target, positive_probabilities)
                )
    else:
        metrics["mae"] = float(sklearn_metrics.mean_absolute_error(target, predictions))
        metrics["rmse"] = float(
            np.sqrt(sklearn_metrics.mean_squared_error(target, predictions))
        )
        metrics["r2"] = float(sklearn_metrics.r2_score(target, predictions))
    return metrics
