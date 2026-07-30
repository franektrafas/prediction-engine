"""Stacking ensemble model."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import BaseModelWrapper, _require_dependency
from football_ai.training.splitters import rolling_positional_splits


LOGGER = logging.getLogger(__name__)

# scikit-learn's StackingClassifier/StackingRegressor generate the
# meta-learner's training data via `cross_val_predict(cv=...)`, which raises
# "cross_val_predict only works for partitions" unless the cv's validation
# folds cover every row of the input exactly once. This project's leak-safe
# rolling CV (`rolling_positional_splits`) is forward-only by design: its
# earliest rows can never sit in a validation fold, because there is no prior
# history to train that fold on -- and there is no fold arrangement that fixes
# this without either training on an empty set or leaking future rows
# backward into predicting past ones. So instead of handing rolling folds to
# scikit-learn's stacking classes, this module builds the meta-learner's
# out-of-fold features by hand (`_out_of_fold_meta_features`), dropping the
# leading rows that structurally can't be validated from the meta-learner's
# own training set -- they're still used as training data for later folds,
# just never scored -- and fits/predicts through a plain `final_estimator`
# instead of `StackingClassifier`/`StackingRegressor`.
_DEFAULT_STACKING_CV_SPLITS = 5


def _rolling_folds(n_rows: int, max_splits: int = _DEFAULT_STACKING_CV_SPLITS) -> list[tuple[Any, Any]]:
    """Best-effort rolling CV folds; empty if there isn't enough data for at least two."""
    upper_bound = min(max_splits, max(0, n_rows // 20))
    for n_splits in range(upper_bound, 1, -1):
        try:
            folds = list(rolling_positional_splits(n_rows, n_splits=n_splits))
        except ValueError:
            continue
        if folds:
            return folds
    return []


def _base_model_outputs(estimator: Any, features: pd.DataFrame, task_type: str) -> np.ndarray:
    """Return one column (regression) or per-class probability columns (classification)."""
    if task_type in {"binary", "multiclass"}:
        probabilities = np.asarray(estimator.predict_proba(features))
        if task_type == "binary" and probabilities.ndim == 2 and probabilities.shape[1] == 2:
            return probabilities[:, [1]]
        return probabilities
    predictions = np.asarray(estimator.predict(features))
    return predictions.reshape(-1, 1)


@dataclass(slots=True)
class FittedStackingEstimator:
    """Picklable, `predict()`/`predict_proba()`-compatible fitted ensemble.

    Holds only what `predictions.service.PredictionService` actually touches
    on a loaded artifact's `estimator` -- fitted base pipelines, the fitted
    meta-learner, and enough metadata to rebuild meta-features for new rows
    the same way training built them.
    """

    task_type: str
    base_estimators: list[Any]
    final_estimator: Any
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def classes_(self) -> np.ndarray:
        """Expose the meta-learner's classes for classification tasks."""
        return self.final_estimator.classes_

    def _meta_features(self, features: pd.DataFrame) -> np.ndarray:
        """Rebuild base-model output columns for new rows."""
        base_outputs = [
            _base_model_outputs(estimator, features, self.task_type)
            for estimator in self.base_estimators
        ]
        return np.concatenate(base_outputs, axis=1)

    def predict(self, features: pd.DataFrame) -> Any:
        """Run ensemble predictions."""
        return self.final_estimator.predict(self._meta_features(features))

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Run ensemble probability predictions."""
        return self.final_estimator.predict_proba(self._meta_features(features))


class StackingEnsembleModel(BaseModelWrapper):
    """Stacking ensemble built on top of independently-fitted base models."""

    def __init__(
        self,
        task_type: str,
        target_name: str,
        base_models: list[BaseModelWrapper],
        version: str = "v1",
    ) -> None:
        """Initialize the ensemble wrapper."""
        super().__init__(
            model_name="stacking_ensemble",
            task_type=task_type,
            target_name=target_name,
            version=version,
        )
        self.base_models = base_models

    def _build_final_estimator(self) -> Any:
        """Create the meta-learner pipeline (median-impute + scale + linear model).

        Meta-features are just base-model outputs (probabilities in [0, 1]
        for classification, a single predicted value for regression) --
        impute/scale is defensive (e.g. a base model returning a stray NaN
        probability) rather than load-bearing, now that raw passthrough
        columns aren't part of the meta-feature matrix (see
        `_out_of_fold_meta_features`).
        """
        sklearn_pipeline = _require_dependency("sklearn.pipeline")
        sklearn_impute = _require_dependency("sklearn.impute")
        sklearn_preprocessing = _require_dependency("sklearn.preprocessing")
        linear_model = _require_dependency("sklearn.linear_model")
        estimator = (
            linear_model.LogisticRegression(max_iter=2000)
            if self.task_type in {"binary", "multiclass"}
            else linear_model.Ridge(alpha=1.0)
        )
        return sklearn_pipeline.Pipeline(
            steps=[
                ("imputer", sklearn_impute.SimpleImputer(strategy="median")),
                ("scaler", sklearn_preprocessing.StandardScaler()),
                ("estimator", estimator),
            ]
        )

    def _out_of_fold_meta_features(
        self,
        features: pd.DataFrame,
        target: pd.Series,
    ) -> tuple[np.ndarray, pd.Series]:
        """Build leak-safe out-of-fold meta-features by refitting base-model clones per fold.

        Returns `(meta_features, meta_target)` covering only the rows that a
        rolling fold actually validated -- the earliest rows (no prior
        history to train a fold on) are excluded from the meta-learner's own
        training set entirely, not zero-filled or leaked into.

        Meta-features are base-model outputs only -- no raw passthrough
        columns. Real measurement (2026-07-24, on the enriched dataset)
        showed this outperforming a variant with ~530 raw passthrough
        columns added on top: match_outcome test accuracy 57.8% vs 55.7%
        (and beating every individual base model, as stacking should,
        instead of losing to all of them, as it previously did), same
        pattern on btts. A linear meta-learner drowning in hundreds of raw,
        non-linear-friendly features -- fit on the out-of-fold subset, which
        is smaller than the base models' own training data -- was actively
        hurting the combination it was meant to improve.

        CONFIRMED REAL LEAK, fixed 2026-07-30: for every classification
        target, `model.estimator` at this point is a
        `CalibratedClassifierCV(FrozenEstimator(base))` (see
        `training.orchestrator._calibrate_on_validation`) -- and
        `sklearn.base.clone()` on a `FrozenEstimator` returns the *same*,
        already-fitted-on-the-original-split object rather than a fresh
        unfitted one (that's `FrozenEstimator`'s whole purpose). Cloning
        `model.estimator` directly meant `clone.fit(...)` below only ever
        refit the calibration curve, never the base decision function --
        so any fold whose validation rows overlapped the model's *original*
        train split (most of them, given the split sizes) produced in-sample,
        not out-of-fold, base-model outputs, undermining this whole
        function's reason to exist. Fixed by cloning
        `pre_calibration_estimator` (the plain, still-clonable pipeline
        stashed before calibration wrapped it) when available -- a true
        per-fold refit again. This does mean OOF meta-features are built
        from *uncalibrated* probabilities while serving later uses the
        *calibrated* ones (see `predict`/`FittedStackingEstimator`) -- a
        real, smaller distribution shift the meta-learner's own
        `StandardScaler` absorbs reasonably well, and a strictly better
        trade-off than not being out-of-fold at all.
        """
        sklearn_base = _require_dependency("sklearn.base")
        folds = _rolling_folds(len(features))
        if not folds:
            LOGGER.warning(
                "Not enough rows (%d) to build leak-safe out-of-fold meta-features for "
                "target '%s'; falling back to in-sample base-model outputs. This is only "
                "expected for tiny fixtures, never for real training data.",
                len(features),
                self.target_name,
            )
            base_outputs = [
                _base_model_outputs(model.estimator, features, self.task_type)
                for model in self.base_models
            ]
            return np.concatenate(base_outputs, axis=1), target

        oof_rows: list[int] = []
        oof_base_outputs: list[list[np.ndarray]] = [[] for _ in self.base_models]
        for train_indices, validation_indices in folds:
            for model_index, model in enumerate(self.base_models):
                refittable_estimator = (
                    model.pre_calibration_estimator
                    if model.pre_calibration_estimator is not None
                    else model.estimator
                )
                clone = sklearn_base.clone(refittable_estimator)
                clone.fit(features.iloc[train_indices], target.iloc[train_indices])
                oof_base_outputs[model_index].append(
                    _base_model_outputs(clone, features.iloc[validation_indices], self.task_type)
                )
            oof_rows.extend(validation_indices.tolist())

        order = np.argsort(oof_rows)
        row_order = np.asarray(oof_rows)[order]
        base_outputs = [np.concatenate(columns, axis=0)[order] for columns in oof_base_outputs]
        meta_features = np.concatenate(base_outputs, axis=1)
        meta_target = target.iloc[row_order]
        return meta_features, meta_target

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
    ) -> "StackingEnsembleModel":
        """Fit the stacking ensemble on the original feature matrix.

        `features`/`target` must already be sorted chronologically (earliest
        row first) -- true for every caller in this project, since they are
        always sliced from `chronological_train_validation_test_split`.
        """
        if any(model.estimator is None for model in self.base_models):
            raise ValueError("All base models must be fitted before the ensemble.")
        self.feature_columns = list(features.columns)

        meta_features, meta_target = self._out_of_fold_meta_features(features, target)
        final_estimator = self._build_final_estimator()
        final_estimator.fit(meta_features, meta_target)

        self.estimator = FittedStackingEstimator(
            task_type=self.task_type,
            base_estimators=[model.estimator for model in self.base_models],
            final_estimator=final_estimator,
        )
        return self

    def predict(self, features: pd.DataFrame) -> Any:
        """Run ensemble predictions."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        return self.estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Run ensemble probability predictions."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        if not hasattr(self.estimator, "predict_proba"):
            raise AttributeError(f"{self.model_name} does not support predict_proba.")
        return self.estimator.predict_proba(features)

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return an empty space because the ensemble uses base-model tuning."""
        return {}
