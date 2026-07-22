"""End-to-end training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import argparse
import logging

import pandas as pd
import numpy as np

from football_ai.config.settings import get_settings
from football_ai.data.cleaning import clean_matches
from football_ai.data.io import read_table
from football_ai.evaluation.metrics import evaluate_predictions
from football_ai.evaluation.plots import (
    plot_calibration_curve,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_precision_recall_curve,
    plot_prediction_distribution,
    plot_roc_curve,
)
from football_ai.evaluation.shap_values import save_shap_summary_plot
from football_ai.features.builder import build_features
from football_ai.models.base import BaseModelWrapper, CLASSIFICATION_TASKS
from football_ai.models.catboost_model import CatBoostModel
from football_ai.models.lightgbm_model import LightGBMModel
from football_ai.models.logistic_regression_model import LogisticRegressionModel
from football_ai.models.random_forest_model import RandomForestModel
from football_ai.models.registry import ModelRegistry, RegisteredModel
from football_ai.models.stacking_ensemble import StackingEnsembleModel
from football_ai.models.xgboost_model import XGBoostModel
from football_ai.training.datasets import TrainingDataset, build_training_dataset
from football_ai.training.tuning import tune_model


LOGGER = logging.getLogger(__name__)

TARGET_TASK_MAPPING = {
    "match_outcome": "multiclass",
    "btts": "binary",
    "over_1_5": "binary",
    "over_2_5": "binary",
    "over_3_5": "binary",
    "home_goals": "regression",
    "away_goals": "regression",
    "home_xg": "regression",
    "away_xg": "regression",
    "home_corners": "regression",
    "away_corners": "regression",
    "home_yellow_cards": "regression",
    "away_yellow_cards": "regression",
    "home_shots": "regression",
    "away_shots": "regression",
    "home_shots_on_target": "regression",
    "away_shots_on_target": "regression",
    "home_possession": "regression",
    "away_possession": "regression",
}


@dataclass(slots=True)
class TrainingResult:
    """Training summary for a single model."""

    model_name: str
    target_name: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    artifact_path: Path


def _build_model_suite(task_type: str, target_name: str) -> list[BaseModelWrapper]:
    """Create the list of candidate models for a target."""
    suite: list[BaseModelWrapper] = []
    suite.extend(
        [
            RandomForestModel(
                model_name="random_forest",
                task_type=task_type,
                target_name=target_name,
            ),
            LightGBMModel(
                model_name="lightgbm",
                task_type=task_type,
                target_name=target_name,
            ),
            CatBoostModel(
                model_name="catboost",
                task_type=task_type,
                target_name=target_name,
            ),
            XGBoostModel(
                model_name="xgboost",
                task_type=task_type,
                target_name=target_name,
            ),
        ]
    )
    return suite


def _metric_for_selection(task_type: str) -> str:
    """Return the primary model-selection metric."""
    return "f1" if task_type in CLASSIFICATION_TASKS else "rmse"


def _select_best_result(results: list[TrainingResult], task_type: str) -> TrainingResult:
    """Select the best model according to validation metrics."""
    metric_name = _metric_for_selection(task_type)
    reverse = task_type in CLASSIFICATION_TASKS
    return sorted(results, key=lambda item: item.validation_metrics[metric_name], reverse=reverse)[0]


def _persist_evaluation_artifacts(
    model: BaseModelWrapper,
    dataset: TrainingDataset,
    predictions: object,
    probabilities: object | None,
) -> None:
    """Persist evaluation plots and SHAP explanations."""
    reports_dir = Path("football_ai/evaluation/reports") / model.target_name / model.model_name
    plot_prediction_distribution(predictions, reports_dir / "prediction_distribution.png")
    if model.task_type in CLASSIFICATION_TASKS:
        plot_confusion_matrix(
            dataset.target.iloc[dataset.split_indices.test_indices],
            predictions,
            reports_dir / "confusion_matrix.png",
        )
        if probabilities is not None and model.task_type == "binary":
            plot_roc_curve(
                dataset.target.iloc[dataset.split_indices.test_indices],
                probabilities,
                reports_dir / "roc_curve.png",
            )
            plot_precision_recall_curve(
                dataset.target.iloc[dataset.split_indices.test_indices],
                probabilities,
                reports_dir / "precision_recall_curve.png",
            )
            plot_calibration_curve(
                dataset.target.iloc[dataset.split_indices.test_indices],
                probabilities,
                reports_dir / "calibration_curve.png",
            )
    estimator_step = getattr(model.estimator, "named_steps", {}).get("estimator")
    if estimator_step is not None and hasattr(estimator_step, "feature_importances_"):
        importances = estimator_step.feature_importances_
        if len(importances) == len(dataset.features.columns):
            plot_feature_importance(
                feature_names=list(dataset.features.columns),
                importances=importances,
                output_path=reports_dir / "feature_importance.png",
            )
    try:
        save_shap_summary_plot(
            estimator=model.estimator,
            features=dataset.features.iloc[dataset.split_indices.test_indices],
            output_path=reports_dir / "shap_summary.png",
        )
    except Exception:
        return


def _normalize_predictions(
    predictions: object,
    task_type: str,
) -> object:
    """Normalize model predictions to a 1D array when needed.

    `CatBoostClassifier.predict()` returns an `(n, 1)` array of the actual
    predicted labels, not a one-hot/probability matrix -- squeeze that back to
    1D instead of treating the lone column as class scores via `argmax`, which
    would silently turn every prediction into class index 0 (this isn't
    hypothetical: it reproduces on every real `train_target()` run because
    CatBoost is always one of the four base models).
    """
    normalized = np.asarray(predictions)
    if task_type not in CLASSIFICATION_TASKS:
        return normalized.ravel() if normalized.ndim > 1 else normalized
    if normalized.ndim == 1:
        return normalized
    if normalized.ndim == 2 and normalized.shape[1] == 1:
        return normalized.ravel()
    raise ValueError(f"Unsupported prediction dimensions: {normalized.shape}")


def train_target(
    dataset: TrainingDataset,
    target_name: str,
    version: str = "v1",
    n_trials: int = 20,
) -> list[TrainingResult]:
    """Train, evaluate, and persist all models for a target."""
    task_type = TARGET_TASK_MAPPING[target_name]
    registry = ModelRegistry()
    results: list[TrainingResult] = []
    train_validation_indices = list(dataset.split_indices.train_indices) + list(
        dataset.split_indices.validation_indices
    )

    candidate_models = _build_model_suite(task_type=task_type, target_name=target_name)
    fitted_models: list[BaseModelWrapper] = []

    for model in candidate_models:
        hyperparameters = tune_model(
            model=model,
            features=dataset.features.iloc[train_validation_indices],
            target=dataset.target.iloc[train_validation_indices],
            categorical_columns=dataset.categorical_columns,
            n_trials=n_trials,
        )
        model.hyperparameters = hyperparameters
        model.version = version
        model.fit(
            dataset.features.iloc[dataset.split_indices.train_indices],
            dataset.target.iloc[dataset.split_indices.train_indices],
            categorical_columns=dataset.categorical_columns,
        )
        validation_predictions = model.predict(
            dataset.features.iloc[dataset.split_indices.validation_indices]
        )
        validation_predictions = _normalize_predictions(validation_predictions, task_type)
        validation_probabilities = None
        if task_type in CLASSIFICATION_TASKS:
            try:
                validation_probabilities = model.predict_proba(
                    dataset.features.iloc[dataset.split_indices.validation_indices]
                )
            except AttributeError:
                validation_probabilities = None
        validation_metrics = evaluate_predictions(
            task_type=task_type,
            target=dataset.target.iloc[dataset.split_indices.validation_indices],
            predictions=validation_predictions,
            probabilities=validation_probabilities,
        )

        model.fit(
            dataset.features.iloc[train_validation_indices],
            dataset.target.iloc[train_validation_indices],
            categorical_columns=dataset.categorical_columns,
        )
        test_predictions = model.predict(dataset.features.iloc[dataset.split_indices.test_indices])
        test_predictions = _normalize_predictions(test_predictions, task_type)
        test_probabilities = None
        if task_type in CLASSIFICATION_TASKS:
            try:
                test_probabilities = model.predict_proba(
                    dataset.features.iloc[dataset.split_indices.test_indices]
                )
            except AttributeError:
                test_probabilities = None
        test_metrics = evaluate_predictions(
            task_type=task_type,
            target=dataset.target.iloc[dataset.split_indices.test_indices],
            predictions=test_predictions,
            probabilities=test_probabilities,
        )
        model.metrics = test_metrics
        artifact_path = model.save(model.artifact_path())
        registry.register(
            RegisteredModel(
                model_name=model.model_name,
                target_name=target_name,
                task_type=task_type,
                version=version,
                artifact_path=str(artifact_path),
                metrics=test_metrics,
            )
        )
        _persist_evaluation_artifacts(model, dataset, test_predictions, test_probabilities)
        fitted_models.append(model)
        results.append(
            TrainingResult(
                model_name=model.model_name,
                target_name=target_name,
                validation_metrics=validation_metrics,
                test_metrics=test_metrics,
                artifact_path=artifact_path,
            )
        )

    stacking_model = StackingEnsembleModel(
        task_type=task_type,
        target_name=target_name,
        base_models=fitted_models[: min(3, len(fitted_models))],
        version=version,
    )
    stacking_model.fit(
        dataset.features.iloc[train_validation_indices],
        dataset.target.iloc[train_validation_indices],
        categorical_columns=dataset.categorical_columns,
    )
    ensemble_validation_predictions = stacking_model.predict(
        dataset.features.iloc[dataset.split_indices.validation_indices]
    )
    ensemble_validation_predictions = _normalize_predictions(
        ensemble_validation_predictions,
        task_type,
    )
    ensemble_validation_probabilities = None
    if task_type in CLASSIFICATION_TASKS:
        try:
            ensemble_validation_probabilities = stacking_model.predict_proba(
                dataset.features.iloc[dataset.split_indices.validation_indices]
            )
        except AttributeError:
            ensemble_validation_probabilities = None
    ensemble_validation_metrics = evaluate_predictions(
        task_type=task_type,
        target=dataset.target.iloc[dataset.split_indices.validation_indices],
        predictions=ensemble_validation_predictions,
        probabilities=ensemble_validation_probabilities,
    )
    ensemble_predictions = stacking_model.predict(
        dataset.features.iloc[dataset.split_indices.test_indices]
    )
    ensemble_predictions = _normalize_predictions(ensemble_predictions, task_type)
    ensemble_probabilities = None
    if task_type in CLASSIFICATION_TASKS:
        try:
            ensemble_probabilities = stacking_model.predict_proba(
                dataset.features.iloc[dataset.split_indices.test_indices]
            )
        except AttributeError:
            ensemble_probabilities = None
    ensemble_metrics = evaluate_predictions(
        task_type=task_type,
        target=dataset.target.iloc[dataset.split_indices.test_indices],
        predictions=ensemble_predictions,
        probabilities=ensemble_probabilities,
    )
    stacking_model.metrics = ensemble_metrics
    ensemble_artifact_path = stacking_model.save(stacking_model.artifact_path())
    registry.register(
        RegisteredModel(
            model_name=stacking_model.model_name,
            target_name=target_name,
            task_type=task_type,
            version=version,
            artifact_path=str(ensemble_artifact_path),
            metrics=ensemble_metrics,
        )
    )
    results.append(
        TrainingResult(
            model_name=stacking_model.model_name,
            target_name=target_name,
            validation_metrics=ensemble_validation_metrics,
            test_metrics=ensemble_metrics,
            artifact_path=ensemble_artifact_path,
        )
    )
    best_result = _select_best_result(results, task_type)
    registry.register(
        RegisteredModel(
            model_name="selected_model",
            target_name=target_name,
            task_type=task_type,
            version=version,
            artifact_path=str(best_result.artifact_path),
            metrics=best_result.validation_metrics,
        )
    )
    return results


def run_training(
    matches_path: str | Path,
    version: str = "v1",
    n_trials: int = 20,
    target_names: list[str] | None = None,
) -> list[TrainingResult]:
    """Load a matches file, build features, and train every requested target.

    A target whose source column isn't present in `matches_path` (a bare
    historical-results file has no `home_xg`/`home_corners`/etc.) is skipped
    with a log message instead of letting `build_training_dataset` raise a
    `KeyError`, since minimal datasets legitimately won't cover every target.
    """
    requested_targets = target_names or list(TARGET_TASK_MAPPING)
    unknown_targets = [name for name in requested_targets if name not in TARGET_TASK_MAPPING]
    if unknown_targets:
        raise KeyError(f"Unknown target(s): {', '.join(sorted(unknown_targets))}")

    matches = read_table(matches_path)
    cleaned_matches, _cleaning_report = clean_matches(matches)
    feature_result = build_features(cleaned_matches)

    results: list[TrainingResult] = []
    for target_name in requested_targets:
        if target_name not in feature_result.targets:
            LOGGER.warning(
                "Skipping target '%s': its source column is not present in %s.",
                target_name,
                matches_path,
            )
            continue
        dataset = build_training_dataset(feature_result, target_name)
        target_results = train_target(dataset, target_name, version=version, n_trials=n_trials)
        results.extend(target_results)
        best_result = _select_best_result(target_results, TARGET_TASK_MAPPING[target_name])
        LOGGER.info(
            "Trained '%s': best model '%s' (test metrics: %s)",
            target_name,
            best_result.model_name,
            best_result.test_metrics,
        )
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and register every prediction target.")
    parser.add_argument(
        "--matches-path",
        default=None,
        help="Path to a wide matches file (parquet/csv/json). Defaults to settings.inference_data_path.",
    )
    parser.add_argument("--version", default="v1", help="Model version tag to register artifacts under.")
    parser.add_argument("--n-trials", type=int, default=20, help="Optuna trials per model per target.")
    parser.add_argument(
        "--targets",
        default=None,
        help="Comma-separated subset of targets to train (default: every target in TARGET_TASK_MAPPING).",
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    resolved_matches_path = (
        Path(arguments.matches_path) if arguments.matches_path else get_settings().inference_data_path
    )
    resolved_targets = arguments.targets.split(",") if arguments.targets else None

    training_results = run_training(
        matches_path=resolved_matches_path,
        version=arguments.version,
        n_trials=arguments.n_trials,
        target_names=resolved_targets,
    )
    print(f"Trained and registered {len(training_results)} model artifacts.")
