"""Tests for football_ai/models/stacking_ensemble.py.

No test file covered this module before (confirmed 2026-07-30 auditing the
engine) -- these focus specifically on the real, confirmed out-of-fold leak
found that day: `_out_of_fold_meta_features` used to clone `model.estimator`
directly, but for every classification target that's a
`CalibratedClassifierCV(FrozenEstimator(base))` by the time stacking runs,
and `sklearn.base.clone()` on a `FrozenEstimator` returns the same,
already-fitted object rather than a fresh unfitted one -- so the "refit per
fold" this module exists to guarantee silently never happened for
classification targets. Fixed by cloning `pre_calibration_estimator` (stashed
before calibration wraps it) instead when available.
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from football_ai.models.base import BaseModelWrapper
from football_ai.models.stacking_ensemble import StackingEnsembleModel


class _TaggedClassifier:
    """A minimal sklearn-compatible "classifier" whose output directly reveals its identity.

    `predict_proba` returns a constant column equal to `tag` regardless of
    input -- so a test can tell exactly which underlying object a clone/refit
    traces back to, without needing any real statistical inference.
    """

    def __init__(self, tag: float = 0.0) -> None:
        """Store the identifying tag."""
        self.tag = tag

    def get_params(self, deep: bool = True) -> dict[str, float]:
        """Expose params for sklearn.base.clone()."""
        return {"tag": self.tag}

    def set_params(self, **params: float) -> "_TaggedClassifier":
        """Accept params for sklearn.base.clone()."""
        self.tag = params.get("tag", self.tag)
        return self

    def fit(self, features: pd.DataFrame, target: pd.Series) -> "_TaggedClassifier":
        """No-op fit -- this stub's whole point is to ignore the data and just report its tag."""
        return self

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Return a constant [1-tag, tag] probability pair for every row."""
        column = np.full(len(features), self.tag)
        return np.stack([1 - column, column], axis=1)


def _dummy_dataset(n_rows: int = 100) -> tuple[pd.DataFrame, pd.Series]:
    """A large-enough (for `_rolling_folds`), otherwise-irrelevant feature/target pair.

    Values don't matter -- `_TaggedClassifier` ignores its input entirely --
    only `n_rows` (needs >= 40 for `_rolling_folds` to produce >= 2 splits).
    """
    features = pd.DataFrame({"x": np.arange(n_rows, dtype=float)})
    target = pd.Series(np.arange(n_rows) % 2, name="y")
    return features, target


class OutOfFoldMetaFeaturesTestCase(unittest.TestCase):
    """Regression tests for the confirmed 2026-07-30 OOF-cloning leak."""

    def test_clones_pre_calibration_estimator_not_the_frozen_calibrated_one(self) -> None:
        """The exact bug: a base model with both attributes set must be refit from pre_calibration_estimator."""
        features, target = _dummy_dataset()
        base_model = BaseModelWrapper(
            model_name="stub",
            task_type="binary",
            target_name="match_outcome",
            estimator=_TaggedClassifier(tag=0.1),  # stands in for the frozen, calibrated wrapper
            pre_calibration_estimator=_TaggedClassifier(tag=0.9),  # the genuinely refittable one
        )
        ensemble = StackingEnsembleModel(task_type="binary", target_name="match_outcome", base_models=[base_model])

        meta_features, meta_target = ensemble._out_of_fold_meta_features(features, target)

        self.assertGreater(len(meta_target), 0)
        # Every OOF value must trace back to the 0.9-tagged (pre-calibration) estimator,
        # never the 0.1-tagged frozen one -- cloning the wrong object was the whole bug.
        self.assertTrue(np.allclose(meta_features[:, 0], 0.9))

    def test_falls_back_to_estimator_when_no_pre_calibration_estimator_exists(self) -> None:
        """Regression targets (and calibration-skipped edge cases) never set pre_calibration_estimator."""
        features, target = _dummy_dataset()
        base_model = BaseModelWrapper(
            model_name="stub",
            task_type="binary",
            target_name="match_outcome",
            estimator=_TaggedClassifier(tag=0.4),
            pre_calibration_estimator=None,
        )
        ensemble = StackingEnsembleModel(task_type="binary", target_name="match_outcome", base_models=[base_model])

        meta_features, _ = ensemble._out_of_fold_meta_features(features, target)

        self.assertTrue(np.allclose(meta_features[:, 0], 0.4))

    def test_fit_end_to_end_produces_a_working_predictor(self) -> None:
        """Sanity check: the full fit()/predict() path still works with a real sklearn estimator."""
        sklearn_linear_model = __import__("sklearn.linear_model", fromlist=["LogisticRegression"])
        rng = np.random.default_rng(7)
        n_rows = 120
        x = rng.normal(size=n_rows)
        features = pd.DataFrame({"x": x})
        target = pd.Series((x > 0).astype(int), name="y")

        real_estimator = sklearn_linear_model.LogisticRegression().fit(features, target)
        base_model = BaseModelWrapper(
            model_name="logreg",
            task_type="binary",
            target_name="match_outcome",
            estimator=real_estimator,
            pre_calibration_estimator=real_estimator,
        )
        ensemble = StackingEnsembleModel(task_type="binary", target_name="match_outcome", base_models=[base_model])
        ensemble.fit(features, target, categorical_columns=[])

        predictions = ensemble.predict(features)
        self.assertEqual(len(predictions), n_rows)
        probabilities = ensemble.predict_proba(features)
        self.assertEqual(probabilities.shape, (n_rows, 2))


if __name__ == "__main__":
    unittest.main()
