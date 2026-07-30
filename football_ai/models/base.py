"""Base abstractions for trainable models."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from football_ai.config.settings import get_settings


CLASSIFICATION_TASKS = {"binary", "multiclass"}


def _require_dependency(module_name: str) -> Any:
    """Import an optional dependency and raise a helpful error when missing."""
    try:
        return __import__(module_name, fromlist=["*"])
    except ModuleNotFoundError as exc:  # pragma: no cover - depends on environment
        raise ModuleNotFoundError(
            f"Missing optional dependency '{module_name}'. Install project requirements first."
        ) from exc


@dataclass(slots=True)
class TrainedArtifact:
    """Serialized model metadata."""

    model_name: str
    target_name: str
    task_type: str
    version: str
    feature_columns: list[str]
    categorical_columns: list[str]
    hyperparameters: dict[str, Any]
    estimator: Any
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class BaseModelWrapper:
    """Base wrapper for a trainable model pipeline."""

    model_name: str
    task_type: str
    target_name: str
    hyperparameters: dict[str, Any] = field(default_factory=dict)
    version: str = "v1"
    estimator: Any | None = None
    # Set by `training.orchestrator._calibrate_on_validation` right before it
    # overwrites `estimator` with a `CalibratedClassifierCV(FrozenEstimator(...))`
    # wrapper -- kept around so `models.stacking_ensemble`'s out-of-fold
    # meta-feature construction has a genuinely re-clonable/re-fittable
    # estimator to work with. `sklearn.base.clone()` on a `FrozenEstimator`
    # returns the same, still-fitted-on-the-original-split object rather than
    # a fresh unfitted one (that's the whole point of "frozen"), so cloning
    # `estimator` itself post-calibration silently defeats per-fold OOF
    # retraining for every classification target -- see stacking_ensemble.py.
    pre_calibration_estimator: Any | None = None
    feature_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)

    def build_estimator(self) -> Any:
        """Create a configured estimator instance."""
        raise NotImplementedError

    def build_pipeline(
        self,
        categorical_columns: list[str],
        numeric_columns: list[str],
    ) -> Any:
        """Create a preprocessing and estimator pipeline."""
        sklearn_pipeline = _require_dependency("sklearn.pipeline")
        sklearn_compose = _require_dependency("sklearn.compose")
        sklearn_impute = _require_dependency("sklearn.impute")
        sklearn_preprocessing = _require_dependency("sklearn.preprocessing")

        numeric_pipeline = sklearn_pipeline.Pipeline(
            steps=[
                ("imputer", sklearn_impute.SimpleImputer(strategy="median")),
                ("scaler", sklearn_preprocessing.StandardScaler()),
            ]
        )
        categorical_pipeline = sklearn_pipeline.Pipeline(
            steps=[
                ("imputer", sklearn_impute.SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    sklearn_preprocessing.OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        preprocessor = sklearn_compose.ColumnTransformer(
            transformers=[
                ("numeric", numeric_pipeline, numeric_columns),
                ("categorical", categorical_pipeline, categorical_columns),
            ],
            remainder="drop",
        )
        return sklearn_pipeline.Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("estimator", self.build_estimator()),
            ]
        )

    def fit(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        categorical_columns: list[str],
    ) -> "BaseModelWrapper":
        """Fit the model pipeline."""
        self.feature_columns = list(features.columns)
        self.categorical_columns = categorical_columns
        numeric_columns = [
            column for column in features.columns if column not in categorical_columns
        ]
        self.estimator = self.build_pipeline(
            categorical_columns=categorical_columns,
            numeric_columns=numeric_columns,
        )
        self.estimator.fit(features, target)
        return self

    def predict(self, features: pd.DataFrame) -> Any:
        """Run model predictions."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        return self.estimator.predict(features)

    def predict_proba(self, features: pd.DataFrame) -> Any:
        """Run probabilistic predictions when supported."""
        if self.estimator is None:
            raise ValueError("Model has not been fitted.")
        if hasattr(self.estimator, "predict_proba"):
            return self.estimator.predict_proba(features)
        estimator_step = self.estimator.named_steps.get("estimator")
        if hasattr(estimator_step, "predict_proba"):
            transformed = self.estimator.named_steps["preprocessor"].transform(features)
            return estimator_step.predict_proba(transformed)
        raise AttributeError(f"{self.model_name} does not support predict_proba.")

    def save(self, output_path: str | Path) -> Path:
        """Serialize the trained model to disk."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        artifact = TrainedArtifact(
            model_name=self.model_name,
            target_name=self.target_name,
            task_type=self.task_type,
            version=self.version,
            feature_columns=self.feature_columns,
            categorical_columns=self.categorical_columns,
            hyperparameters=self.hyperparameters,
            estimator=self.estimator,
            metrics=self.metrics,
        )
        joblib.dump(artifact, path)
        return path

    @classmethod
    def load(cls, input_path: str | Path) -> TrainedArtifact:
        """Load a serialized model artifact."""
        return joblib.load(Path(input_path))

    def artifact_path(self) -> Path:
        """Build a default artifact path for the model."""
        settings = get_settings()
        file_name = f"{self.target_name}_{self.model_name}_{self.version}.joblib"
        return settings.model_artifact_dir / file_name

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return a model-specific hyperparameter search space."""
        raise NotImplementedError
