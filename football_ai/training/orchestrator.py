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
from football_ai.models.base import BaseModelWrapper, CLASSIFICATION_TASKS, _require_dependency
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


def _calibrate_on_validation(model: BaseModelWrapper, dataset: TrainingDataset) -> None:
    """Calibrate a fitted classifier's probabilities using held-out validation data.

    `evaluation/plots.py` already had `plot_calibration_curve` for *diagnosing*
    miscalibration, but nothing upstream of it ever actually calibrated
    anything served -- good accuracy with badly-calibrated probabilities is
    invisible to accuracy/F1, but is exactly what matters for any
    probability-consuming use (betting-edge evaluation, "68% chance" framing
    for media). Platt scaling (`method="sigmoid"`), not isotonic -- isotonic
    needs more data per class to avoid overfitting than this project's
    chronological validation splits reliably provide; sigmoid is the safer
    default at this data volume.

    Wraps the already-fitted (train-only) estimator in `FrozenEstimator` so
    it's never refit here -- `CalibratedClassifierCV(cv="prefit")` was
    removed in sklearn >=1.6 in favor of this spelling. FrozenEstimator does
    NOT skip internal cross-validation, though (confirmed by actually
    running this, not by the docs, which read as if it does): it still runs
    `cross_val_predict` over the given data to get well-distributed
    out-of-fold probability estimates for fitting the calibrator, using
    `self.cv`'s fold count (default 5) -- which raises if any class in the
    calibration set has fewer members than that. Folds are capped to the
    validation set's smallest class count; if even 2 folds isn't possible
    (pathologically small validation splits -- small synthetic test
    fixtures, not real data), calibration is skipped entirely rather than
    aborting training, matching this project's established
    graceful-degradation pattern for anything that can't universally
    succeed on every possible input.

    Mutates `model.estimator` in place; `model.predict`/`.predict_proba`
    delegate straight to it (see `models/base.py`), so nothing downstream
    (saving, serving) needs to know calibration happened at all.
    """
    validation_target = dataset.target.iloc[dataset.split_indices.validation_indices]
    min_class_count = int(validation_target.value_counts().min())
    if min_class_count < 2:
        LOGGER.warning(
            "Skipping calibration for '%s': validation split has a class with only %d "
            "example(s), too few for even 2-fold cross-validation.",
            model.target_name,
            min_class_count,
        )
        return
    sklearn_calibration = _require_dependency("sklearn.calibration")
    sklearn_frozen = _require_dependency("sklearn.frozen")
    # Stashed before `estimator` gets overwritten -- see `pre_calibration_estimator`'s
    # docstring in `models/base.py`. Needed by `stacking_ensemble.py`'s
    # out-of-fold meta-feature construction, which clones+refits a base
    # model per fold; cloning the *calibrated* (frozen) estimator instead
    # would silently skip the refit and defeat the whole point of "out-of-fold".
    model.pre_calibration_estimator = model.estimator
    calibrated = sklearn_calibration.CalibratedClassifierCV(
        estimator=sklearn_frozen.FrozenEstimator(model.estimator),
        method="sigmoid",
        cv=min(5, min_class_count),
    )
    calibrated.fit(
        dataset.features.iloc[dataset.split_indices.validation_indices],
        validation_target,
    )
    model.estimator = calibrated


def train_target(
    dataset: TrainingDataset,
    target_name: str,
    version: str = "v1",
    n_trials: int = 20,
    optuna_storage_path: str | Path | None = None,
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
        # Set before tune_model(), not after -- CONFIRMED REAL BUG caught
        # wiring up study persistence: tune_model()'s resumable study name
        # includes model.version, and the default BaseModelWrapper.version
        # is the same for every model/target until this line runs. Setting
        # it after tune_model() (as this used to) meant every real run's
        # Optuna studies collided under one shared default-version key
        # regardless of the actual --version tag passed on the CLI.
        model.version = version
        hyperparameters = tune_model(
            model=model,
            features=dataset.features.iloc[train_validation_indices],
            target=dataset.target.iloc[train_validation_indices],
            categorical_columns=dataset.categorical_columns,
            n_trials=n_trials,
            storage_path=optuna_storage_path,
        )
        model.hyperparameters = hyperparameters
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

        if task_type in CLASSIFICATION_TASKS:
            # Calibrate on validation instead of refitting on train+validation --
            # calibration needs data the base estimator has never seen (see
            # `_calibrate_on_validation`); trades a bit of base-model training
            # data for genuinely-held-out calibration and an honest test score.
            _calibrate_on_validation(model, dataset)
        else:
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
        # Was `fitted_models[: min(3, len(fitted_models))]` -- traced back
        # (found 2026-07-30 auditing the engine) to the original scaffold
        # commit, never touched again since despite the stacking machinery
        # being rewritten multiple times (leak-safe OOF, calibration,
        # passthrough-column removal). Back when the candidate suite
        # conditionally included LogisticRegression for classification
        # tasks, "first 3" silently dropped CatBoost+XGBoost; today's
        # 4-model suite (random_forest/lightgbm/catboost/xgboost, see
        # `_build_model_suite`) has it silently drop only xgboost, purely
        # because it's last in that list -- no evidence (real training
        # today: xgboost's match_outcome test accuracy was competitive
        # with every other base model) that this was ever a deliberate
        # merit-based choice. Use every fitted candidate.
        base_models=fitted_models,
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
            # test_metrics, not validation_metrics -- every other entry in the
            # registry (per-model, including this same model under its own
            # name) reports test-set metrics. Registering "selected_model"
            # with validation metrics instead made the one entry
            # `PredictionService` actually serves report a different,
            # typically inflated number than its own per-model twin --
            # caught comparing a real local run's registry entries side by
            # side (stacking_ensemble: 0.49 test accuracy vs selected_model,
            # same run, same model: 0.74).
            metrics=best_result.test_metrics,
        )
    )
    return results


def run_training(
    matches_path: str | Path,
    version: str = "v1",
    n_trials: int = 20,
    target_names: list[str] | None = None,
    optuna_storage_path: str | Path | None = None,
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
    # clean_matches only requires match_date/home_team_name/away_team_name, so a
    # scheduled-but-unplayed fixture (NaN score) survives it -- unlike inference,
    # which deliberately appends one such row to predict, training must never see
    # one: it would otherwise pick up whatever placeholder outcome label the source
    # file assigned to an unplayed match as if it were a real result.
    score_columns = [c for c in ("home_score", "away_score") if c in cleaned_matches.columns]
    if score_columns:
        cleaned_matches = cleaned_matches.dropna(subset=score_columns)
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
        target_results = train_target(
            dataset,
            target_name,
            version=version,
            n_trials=n_trials,
            optuna_storage_path=optuna_storage_path,
        )
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
    parser.add_argument(
        "--optuna-storage",
        default=None,
        help=(
            "Path to a SQLite file for resumable Optuna studies, keyed on "
            "(target, model, version) -- a killed/interrupted run resumes from "
            "its last completed trial instead of losing all progress. Defaults "
            "to <model_artifact_dir>/optuna_studies.db; pass an empty string to "
            "disable persistence and use an in-memory (non-resumable) study."
        ),
    )
    arguments = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    resolved_matches_path = (
        Path(arguments.matches_path) if arguments.matches_path else get_settings().inference_data_path
    )
    resolved_targets = arguments.targets.split(",") if arguments.targets else None
    resolved_optuna_storage = (
        arguments.optuna_storage
        if arguments.optuna_storage is not None
        else str(get_settings().model_artifact_dir / "optuna_studies.db")
    ) or None

    training_results = run_training(
        matches_path=resolved_matches_path,
        version=arguments.version,
        n_trials=arguments.n_trials,
        target_names=resolved_targets,
        optuna_storage_path=resolved_optuna_storage,
    )
    print(f"Trained and registered {len(training_results)} model artifacts.")
