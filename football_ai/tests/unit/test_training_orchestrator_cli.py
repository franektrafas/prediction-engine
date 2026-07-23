"""Tests for the real, runnable training entrypoint (`run_training`)."""

from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import pandas as pd

from football_ai.models.registry import ModelRegistry
from football_ai.training.orchestrator import run_training


def _matches_frame(rows: int = 60, include_extended_stats: bool = True) -> pd.DataFrame:
    """Build a small, chronologically balanced synthetic matches frame.

    Cycles home/away/draw outcomes every three rows so every chronological
    slice (train/validation/test) contains all three `match_outcome` classes
    -- with too few examples of one class in a slice, `log_loss`/`predict_proba`
    shapes stop lining up between the classifier's `classes_` and the target.
    """
    teams = ["England", "France", "Spain", "Germany", "Italy", "Portugal"]
    start = pd.Timestamp("2023-01-01")
    records: list[dict[str, object]] = []
    for index in range(rows):
        outcome = index % 3
        home_score, away_score = {0: (2, 0), 1: (1, 1), 2: (0, 2)}[outcome]
        record: dict[str, object] = {
            "match_date": start + pd.Timedelta(days=index * 3),
            "home_team_name": teams[index % len(teams)],
            "away_team_name": teams[(index + 1) % len(teams)],
            "home_score": home_score,
            "away_score": away_score,
        }
        if include_extended_stats:
            record.update(
                {
                    "home_xg": 1.5 + 0.1 * outcome,
                    "away_xg": 1.0 + 0.1 * outcome,
                    "home_shots": 12 + outcome,
                    "away_shots": 9 + outcome,
                    "home_shots_on_target": 5,
                    "away_shots_on_target": 3,
                    "home_corners": 5,
                    "away_corners": 4,
                    "home_yellow_cards": 1,
                    "away_yellow_cards": 2,
                    "home_red_cards": 0,
                    "away_red_cards": 0,
                    "home_possession": 55.0,
                    "away_possession": 45.0,
                    "home_elo": 1800.0 + outcome * 10,
                    "away_elo": 1780.0,
                }
            )
        records.append(record)
    return pd.DataFrame(records)


class TrainingOrchestratorCliTestCase(unittest.TestCase):
    """Validate that `run_training` wires the existing pipeline pieces together."""

    def setUp(self) -> None:
        """Redirect model artifacts to a temp directory for the duration of each test."""
        self.artifact_dir = tempfile.mkdtemp()
        self.env_patch = mock.patch.dict("os.environ", {"MODEL_ARTIFACT_DIR": self.artifact_dir})
        self.env_patch.start()
        self.reports_root = Path("football_ai/evaluation/reports")
        self.pre_existing_report_dirs = (
            {path.name for path in self.reports_root.iterdir()} if self.reports_root.exists() else set()
        )

    def tearDown(self) -> None:
        """Restore the environment and remove any report artifacts this test created.

        `_persist_evaluation_artifacts` writes plots to a hardcoded relative
        path (not settings-driven), so a temp `MODEL_ARTIFACT_DIR` alone
        doesn't keep training runs from touching the real repo tree.
        """
        self.env_patch.stop()
        shutil.rmtree(self.artifact_dir, ignore_errors=True)
        if self.reports_root.exists():
            for path in self.reports_root.iterdir():
                if path.name not in self.pre_existing_report_dirs:
                    shutil.rmtree(path, ignore_errors=True)

    def test_run_training_trains_and_registers_requested_targets(self) -> None:
        """A full-column fixture should train classification, binary, and regression targets."""
        matches_path = Path(self.artifact_dir) / "matches.csv"
        _matches_frame().to_csv(matches_path, index=False)

        results = run_training(
            matches_path=matches_path,
            version="test_v1",
            n_trials=2,
            target_names=["match_outcome", "btts", "home_goals"],
        )

        trained_targets = {result.target_name for result in results}
        self.assertEqual(trained_targets, {"match_outcome", "btts", "home_goals"})

        registry = ModelRegistry()
        for target_name in ("match_outcome", "btts", "home_goals"):
            entry = registry.get_latest(target_name)
            self.assertIsNotNone(entry)
            self.assertEqual(entry.model_name, "selected_model")
            self.assertTrue(Path(entry.artifact_path).exists())

            # Regression test: "selected_model" must report the winning
            # model's *test*-set metrics, matching every other registry
            # entry -- it used to report validation-set metrics instead,
            # so the one entry PredictionService actually serves showed a
            # different (typically inflated) number than its own per-model
            # twin in the same run.
            matching_result = next(
                result
                for result in results
                if result.target_name == target_name
                and str(result.artifact_path) == entry.artifact_path
            )
            self.assertEqual(entry.metrics, matching_result.test_metrics)

    def test_run_training_skips_targets_missing_from_reduced_input(self) -> None:
        """Targets whose source column is absent should be skipped, not raise."""
        matches_path = Path(self.artifact_dir) / "matches_reduced.csv"
        _matches_frame(include_extended_stats=False).to_csv(matches_path, index=False)

        results = run_training(
            matches_path=matches_path,
            version="test_v1",
            n_trials=2,
            target_names=["match_outcome", "home_xg"],
        )

        trained_targets = {result.target_name for result in results}
        self.assertEqual(trained_targets, {"match_outcome"})

    def test_run_training_rejects_unknown_target_names(self) -> None:
        """An unknown target name should fail fast instead of failing deep in the pipeline."""
        matches_path = Path(self.artifact_dir) / "matches.csv"
        _matches_frame().to_csv(matches_path, index=False)

        with self.assertRaises(KeyError):
            run_training(matches_path=matches_path, target_names=["not_a_real_target"])


if __name__ == "__main__":
    unittest.main()
