"""Tests for the filesystem model registry."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
