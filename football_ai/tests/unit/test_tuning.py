"""Tests for Optuna-based hyperparameter tuning, incl. SQLite-backed resume."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from football_ai.models.base import BaseModelWrapper, _require_dependency
from football_ai.training.tuning import tune_model


class _FastRidgeModel(BaseModelWrapper):
    """A cheap-to-fit regression model, purely so tuning tests run in milliseconds."""

    def build_estimator(self) -> Any:
        """Create a tiny Ridge estimator."""
        linear_model = _require_dependency("sklearn.linear_model")
        return linear_model.Ridge(alpha=self.hyperparameters.get("alpha", 1.0))

    def hyperparameter_space(self) -> dict[str, tuple[Any, Any]]:
        """Return a tiny, fast-to-search space."""
        return {"alpha": (0.1, 2.0)}


def _synthetic_dataset(n_rows: int = 40) -> tuple[pd.DataFrame, pd.Series]:
    """Build a tiny, deterministic regression dataset."""
    rng = np.random.default_rng(0)
    features = pd.DataFrame({"x1": rng.normal(size=n_rows), "x2": rng.normal(size=n_rows)})
    target = pd.Series(features["x1"] * 2.0 + rng.normal(scale=0.1, size=n_rows), name="y")
    return features, target


class TuneModelTestCase(unittest.TestCase):
    """Validate tune_model's default (in-memory) behavior and SQLite-backed resume."""

    def test_tune_model_without_storage_returns_hyperparameters(self) -> None:
        """The default, non-persistent path should still work exactly as before."""
        features, target = _synthetic_dataset()
        model = _FastRidgeModel(model_name="fast_ridge", task_type="regression", target_name="y")
        result = tune_model(model, features, target, categorical_columns=[], n_trials=2)
        self.assertIn("alpha", result)

    def test_a_resumed_study_only_runs_the_remaining_trials(self) -> None:
        """Regression test for the real scenario this exists for: a killed/interrupted run

        must pick up from its last completed trial, not restart from zero and not run
        n_trials *more* on top of what's already there.
        """
        features, target = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "studies.db"

            model = _FastRidgeModel(
                model_name="fast_ridge", task_type="regression", target_name="y", version="v1"
            )
            tune_model(model, features, target, categorical_columns=[], n_trials=3, storage_path=storage_path)

            import optuna

            study = optuna.load_study(study_name="y__fast_ridge__v1", storage=f"sqlite:///{storage_path}")
            self.assertEqual(len(study.trials), 3)

            # "Resume" with the same key and the same n_trials -- since 3 trials
            # already completed, this must run 0 more, not another 3 (6 total).
            tune_model(model, features, target, categorical_columns=[], n_trials=3, storage_path=storage_path)
            study_after = optuna.load_study(study_name="y__fast_ridge__v1", storage=f"sqlite:///{storage_path}")
            self.assertEqual(len(study_after.trials), 3)

            # Raising n_trials on the resumed call must only run the delta (2 more, for 5 total).
            tune_model(model, features, target, categorical_columns=[], n_trials=5, storage_path=storage_path)
            study_final = optuna.load_study(study_name="y__fast_ridge__v1", storage=f"sqlite:///{storage_path}")
            self.assertEqual(len(study_final.trials), 5)

    def test_different_versions_get_independent_studies(self) -> None:
        """Two versions of the same model/target must not share -- or collide on -- a study.

        Regression test for a real bug: model.version used to be set *after* tune_model()
        ran, so every real training run's studies shared one default-version key regardless
        of the actual --version tag, silently mixing unrelated runs' trials together.
        """
        features, target = _synthetic_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir) / "studies.db"
            model_v1 = _FastRidgeModel(
                model_name="fast_ridge", task_type="regression", target_name="y", version="v1"
            )
            model_v2 = _FastRidgeModel(
                model_name="fast_ridge", task_type="regression", target_name="y", version="v2"
            )
            tune_model(model_v1, features, target, categorical_columns=[], n_trials=2, storage_path=storage_path)
            tune_model(model_v2, features, target, categorical_columns=[], n_trials=2, storage_path=storage_path)

            import optuna

            study_v1 = optuna.load_study(study_name="y__fast_ridge__v1", storage=f"sqlite:///{storage_path}")
            study_v2 = optuna.load_study(study_name="y__fast_ridge__v2", storage=f"sqlite:///{storage_path}")
            self.assertEqual(len(study_v1.trials), 2)
            self.assertEqual(len(study_v2.trials), 2)


if __name__ == "__main__":
    unittest.main()
