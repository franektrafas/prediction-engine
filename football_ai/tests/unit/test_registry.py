"""Tests for the filesystem model registry."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from football_ai.models.registry import ModelRegistry, RegisteredModel


class RegistryTestCase(unittest.TestCase):
    """Validate registry persistence."""

    def test_register_and_get_latest(self) -> None:
        """The registry should persist entries and return the newest one."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry.json")
            registry.register(
                RegisteredModel(
                    model_name="lightgbm",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="v1",
                    artifact_path="/tmp/model_v1.joblib",
                    metrics={"f1": 0.5},
                )
            )
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="v2",
                    artifact_path="/tmp/model_v2.joblib",
                    metrics={"f1": 0.6},
                )
            )
            latest = registry.get_latest("match_outcome")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.version, "v2")

    def test_selected_model_wins_even_when_not_registered_last(self) -> None:
        """Regression test: get_latest() used to be purely positional, not selection-aware.

        `training.orchestrator.train_target()` always registers
        "selected_model" last on a successful run, so positional-last
        happened to agree with it by construction -- but if a run crashes
        (or, as here, a later unrelated entry gets registered) after
        "selected_model" but before/instead of a fresh one, positional-last
        would silently serve the wrong artifact. "selected_model" must win
        regardless of registration order.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry.json")
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="chosen_v1",
                    artifact_path="/tmp/chosen.joblib",
                    metrics={"f1": 0.6},
                )
            )
            registry.register(
                RegisteredModel(
                    model_name="stacking_ensemble",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="mid_crash_v2",
                    artifact_path="/tmp/mid_crash.joblib",
                    metrics={"f1": 0.4},
                )
            )
            latest = registry.get_latest("match_outcome")
            self.assertIsNotNone(latest)
            self.assertEqual(latest.model_name, "selected_model")
            self.assertEqual(latest.artifact_path, "/tmp/chosen.joblib")

    def test_falls_back_to_positional_last_when_no_selected_model_exists(self) -> None:
        """A registry that never uses the "selected_model" convention keeps its prior behavior."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry.json")
            registry.register(
                RegisteredModel(
                    model_name="random_forest",
                    target_name="btts",
                    task_type="binary",
                    version="v1",
                    artifact_path="/tmp/rf.joblib",
                    metrics={"f1": 0.5},
                )
            )
            registry.register(
                RegisteredModel(
                    model_name="xgboost",
                    target_name="btts",
                    task_type="binary",
                    version="v2",
                    artifact_path="/tmp/xgb.joblib",
                    metrics={"f1": 0.5},
                )
            )
            latest = registry.get_latest("btts")
            self.assertEqual(latest.model_name, "xgboost")

    def test_withdraw_removes_target_from_active_serving(self) -> None:
        """After withdraw(), get_latest() must return None for that target."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry.json")
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="home_possession",
                    task_type="regression",
                    version="v1",
                    artifact_path="/tmp/possession.joblib",
                    metrics={"r2": 0.66},
                )
            )
            withdrawn = registry.withdraw("home_possession", reason="ESPN license")
            self.assertEqual(len(withdrawn), 1)
            self.assertIsNone(registry.get_latest("home_possession"))

    def test_withdraw_does_not_touch_other_targets(self) -> None:
        """Withdrawing one target must leave every other target's entries intact."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry = ModelRegistry(Path(temp_dir) / "registry.json")
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="match_outcome",
                    task_type="multiclass",
                    version="v1",
                    artifact_path="/tmp/outcome.joblib",
                    metrics={"f1": 0.6},
                )
            )
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="home_possession",
                    task_type="regression",
                    version="v1",
                    artifact_path="/tmp/possession.joblib",
                    metrics={"r2": 0.66},
                )
            )
            registry.withdraw("home_possession", reason="ESPN license")
            self.assertIsNotNone(registry.get_latest("match_outcome"))

    def test_withdraw_archives_entries_with_the_reason(self) -> None:
        """Withdrawn entries must be recoverable, not deleted -- archived with why."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry = ModelRegistry(registry_path)
            registry.register(
                RegisteredModel(
                    model_name="selected_model",
                    target_name="home_possession",
                    task_type="regression",
                    version="v1",
                    artifact_path="/tmp/possession.joblib",
                    metrics={"r2": 0.66},
                )
            )
            registry.withdraw(
                "home_possession", reason="ESPN license, confirmed 2026-08-20"
            )

            archive_path = registry_path.with_name("registry_withdrawn.json")
            self.assertTrue(archive_path.exists())
            archive = json.loads(archive_path.read_text(encoding="utf-8"))
            self.assertIn("home_possession", archive)
            self.assertEqual(len(archive["home_possession"]), 1)
            record = archive["home_possession"][0]
            self.assertEqual(record["reason"], "ESPN license, confirmed 2026-08-20")
            self.assertEqual(
                record["entries"][0]["artifact_path"], "/tmp/possession.joblib"
            )

    def test_withdraw_on_unregistered_target_is_a_safe_no_op(self) -> None:
        """Withdrawing a target with no entries must not raise or create an archive."""
        with tempfile.TemporaryDirectory() as temp_dir:
            registry_path = Path(temp_dir) / "registry.json"
            registry = ModelRegistry(registry_path)
            withdrawn = registry.withdraw("never_trained", reason="n/a")
            self.assertEqual(withdrawn, [])
            self.assertFalse(registry_path.with_name("registry_withdrawn.json").exists())


if __name__ == "__main__":
    unittest.main()
