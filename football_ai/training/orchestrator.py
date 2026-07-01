"""End-to-end training orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

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
from football_ai.models.base import BaseModelWrapper, CLASSIFICATION_TASKS
from football_ai.models.catboost_model import CatBoostModel
from football_ai.models.lightgbm_model import LightGBMModel
from football_ai.models.logistic_regression_model import LogisticRegressionModel
from football_ai.models.random_forest_model import RandomForestModel
from football_ai.models.registry import ModelRegistry, RegisteredModel
from football_ai.models.stacking_ensemble import StackingEnsembleModel
from football_ai.models.xgboost_model import XGBoostModel
from football_ai.training.datasets import TrainingDataset
from football_ai.training.tuning import tune_model


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


if __name__ == "__main__":
    print(
        "Import build_features(...) and build_training_dataset(...), then call train_target(...)."
    )
