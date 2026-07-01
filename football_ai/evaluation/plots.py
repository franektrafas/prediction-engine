"""Evaluation plotting utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from football_ai.data.io import ensure_parent_directory
from football_ai.models.base import _require_dependency


def plot_feature_importance(
    feature_names: list[str],
    importances: np.ndarray,
    output_path: str | Path,
    top_n: int = 25,
) -> Path:
    """Plot and save feature importances."""
    order = np.argsort(importances)[::-1][:top_n]
    figure, axis = plt.subplots(figsize=(12, 8))
    axis.barh(np.array(feature_names)[order][::-1], importances[order][::-1])
    axis.set_title("Feature Importance")
    axis.set_xlabel("Importance")
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_confusion_matrix(
    target: pd.Series,
    predictions: Any,
    output_path: str | Path,
) -> Path:
    """Plot and save a confusion matrix."""
    sklearn_metrics = _require_dependency("sklearn.metrics")
    matrix = sklearn_metrics.confusion_matrix(target, predictions)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title("Confusion Matrix")
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Actual")
    figure.colorbar(image, ax=axis)
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_prediction_distribution(predictions: Any, output_path: str | Path) -> Path:
    """Plot the prediction distribution."""
    figure, axis = plt.subplots(figsize=(10, 6))
    prediction_series = pd.Series(predictions)
    axis.set_title("Prediction Distribution")
    if pd.api.types.is_numeric_dtype(prediction_series):
        axis.hist(prediction_series.astype(float), bins=30, alpha=0.8)
        axis.set_xlabel("Prediction")
        axis.set_ylabel("Count")
    else:
        counts = prediction_series.astype(str).value_counts().sort_index()
        axis.bar(counts.index, counts.values)
        axis.set_xlabel("Prediction Class")
        axis.set_ylabel("Count")
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_roc_curve(
    target: pd.Series,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot and save a ROC curve for binary classification."""
    sklearn_metrics = _require_dependency("sklearn.metrics")
    fpr, tpr, _ = sklearn_metrics.roc_curve(target, probabilities[:, 1])
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(fpr, tpr, label="ROC")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_title("ROC Curve")
    axis.set_xlabel("False Positive Rate")
    axis.set_ylabel("True Positive Rate")
    axis.legend()
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_precision_recall_curve(
    target: pd.Series,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot and save a precision-recall curve for binary classification."""
    sklearn_metrics = _require_dependency("sklearn.metrics")
    precision, recall, _ = sklearn_metrics.precision_recall_curve(target, probabilities[:, 1])
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(recall, precision, label="PR")
    axis.set_title("Precision Recall Curve")
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.legend()
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output


def plot_calibration_curve(
    target: pd.Series,
    probabilities: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Plot and save a calibration curve for binary classification."""
    calibration = _require_dependency("sklearn.calibration")
    prob_true, prob_pred = calibration.calibration_curve(target, probabilities[:, 1], n_bins=10)
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(prob_pred, prob_true, marker="o")
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey")
    axis.set_title("Calibration Curve")
    axis.set_xlabel("Predicted Probability")
    axis.set_ylabel("Observed Frequency")
    output = Path(output_path)
    ensure_parent_directory(output)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)
    return output
