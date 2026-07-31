"""Prediction service for model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from numbers import Real

import logging
import math

import joblib
import numpy as np
import pandas as pd

from football_ai.config.settings import get_settings
from football_ai.data.io import read_table
from football_ai.data.normalization import MATCH_OUTCOME_LABELS
from football_ai.models.base import TrainedArtifact
from football_ai.models.registry import ModelRegistry
from football_ai.predictions.inference_feature_builder import build_inference_feature_frame
from football_ai.predictions.schemas import (
    MatchContext,
    MatchContextPayload,
    PredictionResponse,
    ProbabilityPayload,
)


LOGGER = logging.getLogger(__name__)


def _logistic(value: float) -> float:
    """Apply a logistic transformation."""
    return 1.0 / (1.0 + math.exp(-value))


def _clamp_probability(value: float) -> float:
    """Clamp a probability into the valid range."""
    return max(0.0, min(1.0, float(value)))


def _normalize_probabilities(probabilities: dict[str, float]) -> dict[str, float]:
    """Normalize probabilities so they sum to one."""
    sanitized = {
        key: _clamp_probability(value)
        for key, value in probabilities.items()
    }
    probability_sum = sum(sanitized.values())
    if probability_sum <= 0.0:
        return {"home_win": 1.0 / 3.0, "draw": 1.0 / 3.0, "away_win": 1.0 / 3.0}
    return {
        key: value / probability_sum
        for key, value in sanitized.items()
    }


def _probability_spread(probabilities: dict[str, float]) -> float:
    """Return the spread between the largest and smallest probability."""
    values = list(probabilities.values())
    if not values:
        return 0.0
    return max(values) - min(values)


_MATCH_CONTEXT_FIELDS = (
    "home_injured_market_value",
    "away_injured_market_value",
    "home_lineup_market_value",
    "away_lineup_market_value",
    "home_goalkeeper_form",
    "away_goalkeeper_form",
    "home_forward_form",
    "away_forward_form",
)


def _build_match_context(payload: MatchContextPayload | dict[str, object]) -> MatchContext:
    """Surface live, non-learned context fields from the payload onto the response.

    Never raises -- a missing/absent field just stays None, matching every
    other live-enrichment path's graceful-degradation contract in this
    project.
    """
    if isinstance(payload, dict):
        values = {field: payload.get(field) for field in _MATCH_CONTEXT_FIELDS}
    else:
        values = {field: getattr(payload, field, None) for field in _MATCH_CONTEXT_FIELDS}
    return MatchContext(**values)


def _canonicalize_class_label(label: object) -> object:
    """Map model-specific class labels into project-level outcome labels."""
    if isinstance(label, (int, float, np.integer, np.floating)) and not isinstance(label, bool):
        numeric_mapping = dict(enumerate(MATCH_OUTCOME_LABELS))
        integer_label = int(label)
        if float(label) == float(integer_label):
            return numeric_mapping.get(integer_label, label)
        return label
    if isinstance(label, str):
        normalized = label.strip()
        string_mapping = {
            "A": "away_win",
            "D": "draw",
            "H": "home_win",
            "0": "away_win",
            "1": "draw",
            "2": "home_win",
            "away_win": "away_win",
            "draw": "draw",
            "home_win": "home_win",
        }
        return string_mapping.get(normalized, normalized)
    return label


def _repair_simple_imputer_state(estimator: object) -> None:
    """Backfill sklearn SimpleImputer internals for reloaded legacy artifacts."""
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            _repair_simple_imputer_state(step)
    if hasattr(estimator, "transformers_"):
        for _, transformer, _ in getattr(estimator, "transformers_", []):
            _repair_simple_imputer_state(transformer)
    if hasattr(estimator, "steps"):
        for _, step in getattr(estimator, "steps", []):
            _repair_simple_imputer_state(step)

    if estimator.__class__.__name__ != "SimpleImputer":
        return
    if hasattr(estimator, "_fill_dtype"):
        return
    statistics = getattr(estimator, "statistics_", None)
    if statistics is None:
        return
    try:
        estimator._fill_dtype = np.asarray(statistics).dtype
    except Exception:
        estimator._fill_dtype = np.dtype("float64")


@dataclass(slots=True)
class PredictionService:
    """Prediction service that loads trained artifacts when available."""

    registry: ModelRegistry
    historical_matches: pd.DataFrame | None = None

    @classmethod
    def create(cls) -> "PredictionService":
        """Create a prediction service with the default registry."""
        settings = get_settings()
        historical_matches: pd.DataFrame | None = None
        if settings.inference_data_path.exists():
            historical_matches = read_table(settings.inference_data_path)
        return cls(registry=ModelRegistry(), historical_matches=historical_matches)

    def predict(self, payload: MatchContextPayload) -> PredictionResponse:
        """Predict match outcome and statistics."""
        registry_entry = self.registry.get_latest("match_outcome")
        if registry_entry is None:
            return self._heuristic_prediction(payload)
        match_frame = self._build_feature_frame(payload)
        outcome_probabilities = self._predict_classification(
            artifact_path=registry_entry.artifact_path,
            frame=match_frame,
            labels=["away_win", "draw", "home_win"],
        )
        model_versions = {"match_outcome": registry_entry.version}
        home_goals = self._predict_regression("home_goals", match_frame, default=1.4)
        away_goals = self._predict_regression("away_goals", match_frame, default=1.1)
        home_xg = self._predict_regression("home_xg", match_frame, default=home_goals)
        away_xg = self._predict_regression("away_xg", match_frame, default=away_goals)
        home_corners = self._predict_regression("home_corners", match_frame, default=5.2)
        away_corners = self._predict_regression("away_corners", match_frame, default=4.5)
        home_cards = self._predict_regression("home_yellow_cards", match_frame, default=1.8)
        away_cards = self._predict_regression("away_yellow_cards", match_frame, default=2.0)
        home_shots = self._predict_regression("home_shots", match_frame, default=13.0)
        away_shots = self._predict_regression("away_shots", match_frame, default=10.2)
        home_shots_on_target = self._predict_regression(
            "home_shots_on_target", match_frame, default=4.7
        )
        away_shots_on_target = self._predict_regression(
            "away_shots_on_target", match_frame, default=3.6
        )
        home_possession = self._predict_regression("home_possession", match_frame, default=53.0)
        away_possession = self._predict_regression("away_possession", match_frame, default=47.0)
        total_goals = home_goals + away_goals
        outcome_probabilities = self._resolve_outcome_probabilities(
            payload=payload,
            model_probabilities=outcome_probabilities,
            home_goals=home_goals,
            away_goals=away_goals,
        )
        btts = self._predict_binary("btts", match_frame, default=float(home_goals > 0 and away_goals > 0))
        for target_name in (
            "home_goals",
            "away_goals",
            "home_xg",
            "away_xg",
            "home_corners",
            "away_corners",
            "home_yellow_cards",
            "away_yellow_cards",
            "home_shots",
            "away_shots",
            "home_shots_on_target",
            "away_shots_on_target",
            "home_possession",
            "away_possession",
            "btts",
            "over_1_5",
            "over_2_5",
            "over_3_5",
        ):
            version = self._get_model_version(target_name)
            if version is not None:
                model_versions[target_name] = version
        over_under = {
            "over_1_5": self._predict_binary("over_1_5", match_frame, default=_logistic(total_goals - 1.5)),
            "over_2_5": self._predict_binary("over_2_5", match_frame, default=_logistic(total_goals - 2.5)),
            "over_3_5": self._predict_binary("over_3_5", match_frame, default=_logistic(total_goals - 3.5)),
        }
        return PredictionResponse(
            model_version=registry_entry.version,
            model_versions=model_versions,
            outcome_probabilities=ProbabilityPayload(
                home_win=outcome_probabilities["home_win"],
                draw=outcome_probabilities["draw"],
                away_win=outcome_probabilities["away_win"],
            ),
            exact_score={
                "home": max(0, int(round(home_goals))),
                "away": max(0, int(round(away_goals))),
            },
            goals={"home": home_goals, "away": away_goals},
            btts=btts,
            over_under=over_under,
            corners={"home": home_corners, "away": away_corners},
            cards={"home": home_cards, "away": away_cards},
            shots={"home": home_shots, "away": away_shots},
            shots_on_target={"home": home_shots_on_target, "away": away_shots_on_target},
            xg={"home": home_xg, "away": away_xg},
            possession={"home": home_possession, "away": away_possession},
            match_context=_build_match_context(payload),
        )

    def _build_feature_frame(self, payload: MatchContextPayload) -> pd.DataFrame:
        """Build a single-row feature frame for inference."""
        return build_inference_feature_frame(
            payload=payload,
            historical_matches=self.historical_matches,
        ).frame

    def _load_artifact(self, artifact_path: str | Path) -> TrainedArtifact:
        """Load a serialized model artifact."""
        artifact = joblib.load(Path(artifact_path))
        _repair_simple_imputer_state(artifact.estimator)
        return artifact

    def _align_features(
        self,
        frame: pd.DataFrame,
        artifact_path: str | Path,
    ) -> tuple[pd.DataFrame, TrainedArtifact]:
        """Load an artifact and align inference features to the trained schema.

        Missing values are left as NaN, not pre-filled with 0.0/"" --
        CONFIRMED bug (found 2026-07-30 auditing the engine): every trained
        artifact is a full sklearn Pipeline with its own SimpleImputer
        (median for numeric, most_frequent for categorical -- see
        `models/base.py`'s `build_pipeline()`). Pre-filling with 0.0 before
        `.predict()` means that imputer never sees a real NaN to act on --
        the raw 0.0 flows straight to `StandardScaler`, scaled using the
        training-time mean/std, which for a column like `home_elo_rank`
        (real median ~78) renders as an extreme, out-of-distribution "best
        team ever" signal rather than "no data". Letting NaN flow through
        instead lets the model's own trained imputer handle it as designed
        -- column routing to the numeric vs. categorical branch is done by
        name (`ColumnTransformer` against `feature_columns`/
        `categorical_columns`), not by dtype, so this is safe even for a
        column that's entirely missing and gets created here as float NaN.

        Logs a warning when the live feature frame is missing columns the
        artifact expects -- CONFIRMED gap (found 2026-07-30 auditing the
        engine): `RegisteredModel` carries no feature-schema id, and every
        target's artifact is loaded/aligned independently, so a
        `features/builder.py` change that adds/renames/removes columns
        degrades silently here (the NaN-fill above keeps `.predict()` from
        crashing, but a caller has no way to notice a target is quietly
        running on a stale schema until they go looking). This doesn't fix
        the root cause -- registering a feature-schema hash alongside each
        `RegisteredModel` and rejecting/flagging a mismatch would -- but a
        loud log line is a real, cheap improvement over total silence, and
        surfaces exactly which columns and target were affected.
        """
        loaded_artifact = self._load_artifact(artifact_path)
        aligned = frame.copy()
        missing_columns = [
            column for column in loaded_artifact.feature_columns if column not in aligned.columns
        ]
        if missing_columns:
            LOGGER.warning(
                "Feature schema drift for target '%s' (artifact %s): %d column(s) the "
                "trained artifact expects are absent from the live feature frame and will "
                "be NaN-filled: %s%s",
                loaded_artifact.target_name,
                artifact_path,
                len(missing_columns),
                missing_columns[:10],
                "..." if len(missing_columns) > 10 else "",
            )
        for column in missing_columns:
            aligned[column] = np.nan
        aligned = aligned[loaded_artifact.feature_columns].copy()
        aligned = aligned.replace([np.inf, -np.inf], np.nan)
        return aligned, loaded_artifact

    def _predict_classification(
        self,
        artifact_path: str | Path,
        frame: pd.DataFrame,
        labels: list[str],
    ) -> dict[str, float]:
        """Predict class probabilities using a trained model."""
        aligned, artifact = self._align_features(frame, artifact_path)
        if not hasattr(artifact.estimator, "predict_proba"):
            return self._heuristic_prediction(frame.iloc[0].to_dict()).outcome_probabilities.model_dump()
        probabilities = np.asarray(artifact.estimator.predict_proba(aligned))
        if probabilities.ndim == 2:
            probability_values = probabilities[0]
        else:
            probability_values = probabilities
        estimator_with_classes = artifact.estimator
        if hasattr(artifact.estimator, "named_steps"):
            estimator_with_classes = artifact.estimator.named_steps.get("estimator", artifact.estimator)
        raw_classes = getattr(estimator_with_classes, "classes_", labels)
        classes = [_canonicalize_class_label(label) for label in raw_classes]
        label_mapping = dict(zip(classes, probability_values, strict=False))
        if not {"home_win", "draw", "away_win"}.issubset(set(label_mapping)):
            if len(probability_values) == 3:
                label_mapping = {
                    "away_win": float(probability_values[0]),
                    "draw": float(probability_values[1]),
                    "home_win": float(probability_values[2]),
                }
        return _normalize_probabilities({
            "home_win": float(label_mapping.get("home_win", 0.33)),
            "draw": float(label_mapping.get("draw", 0.34)),
            "away_win": float(label_mapping.get("away_win", 0.33)),
        })

    def _resolve_outcome_probabilities(
        self,
        payload: MatchContextPayload,
        model_probabilities: dict[str, float],
        home_goals: float,
        away_goals: float,
    ) -> dict[str, float]:
        """Resolve final outcome probabilities for API responses.

        When the multiclass model produces an almost-flat distribution, fall back
        to a heuristic estimate informed by Elo and expected goals.
        """
        if _probability_spread(model_probabilities) >= 0.05:
            return model_probabilities
        heuristic = self._heuristic_prediction(payload)
        heuristic_probabilities = heuristic.outcome_probabilities.model_dump()
        goal_margin = home_goals - away_goals
        goal_based = _normalize_probabilities(
            {
                "home_win": _logistic(goal_margin + 0.15),
                "draw": max(0.08, 1.0 - _logistic(goal_margin + 0.15) - _logistic(-goal_margin - 0.15) * 0.9),
                "away_win": _logistic(-goal_margin - 0.15) * 0.9,
            }
        )
        blended = {
            "home_win": 0.65 * heuristic_probabilities["home_win"] + 0.35 * goal_based["home_win"],
            "draw": 0.65 * heuristic_probabilities["draw"] + 0.35 * goal_based["draw"],
            "away_win": 0.65 * heuristic_probabilities["away_win"] + 0.35 * goal_based["away_win"],
        }
        return _normalize_probabilities(blended)

    def _predict_binary(
        self,
        target_name: str,
        frame: pd.DataFrame,
        default: float,
    ) -> float:
        """Predict a binary probability with fallback defaults."""
        registry_entry = self.registry.get_latest(target_name)
        if registry_entry is None:
            return float(default)
        aligned, artifact = self._align_features(frame, registry_entry.artifact_path)
        if hasattr(artifact.estimator, "predict_proba"):
            probability = np.asarray(artifact.estimator.predict_proba(aligned))
            if probability.ndim == 2:
                positive_class_probability = probability[0][-1]
            else:
                positive_class_probability = probability[-1]
            return _clamp_probability(float(positive_class_probability))
        return _clamp_probability(float(default))

    def _predict_regression(
        self,
        target_name: str,
        frame: pd.DataFrame,
        default: float,
    ) -> float:
        """Predict a numeric target with fallback defaults."""
        registry_entry = self.registry.get_latest(target_name)
        if registry_entry is None:
            return float(default)
        aligned, artifact = self._align_features(frame, registry_entry.artifact_path)
        return float(artifact.estimator.predict(aligned)[0])

    def _get_model_version(self, target_name: str) -> str | None:
        """Resolve the latest registered version for a target."""
        registry_entry = self.registry.get_latest(target_name)
        if registry_entry is None:
            return None
        return registry_entry.version

    def _heuristic_prediction(self, payload: MatchContextPayload | dict[str, object]) -> PredictionResponse:
        """Fallback heuristic prediction when trained models are unavailable."""
        if isinstance(payload, dict):
            home_elo = float(payload.get("home_elo", payload.get("home_elo_pre_match", 1500.0)) or 1500.0)
            away_elo = float(payload.get("away_elo", payload.get("away_elo_pre_match", 1500.0)) or 1500.0)
        else:
            home_elo = payload.home_elo or 1500.0
            away_elo = payload.away_elo or 1500.0
        elo_diff = (home_elo - away_elo) / 400.0
        home_win = _logistic(elo_diff + 0.18)
        away_win = _logistic(-elo_diff - 0.18) * 0.78
        draw = max(0.05, 1.0 - home_win - away_win)
        probability_sum = home_win + draw + away_win
        home_win /= probability_sum
        draw /= probability_sum
        away_win /= probability_sum
        home_goals = max(0.2, 1.35 + elo_diff * 0.55)
        away_goals = max(0.15, 1.10 - elo_diff * 0.45)
        total_goals = home_goals + away_goals
        return PredictionResponse(
            model_version="heuristic_baseline_v0",
            model_versions={"match_outcome": "heuristic_baseline_v0"},
            outcome_probabilities=ProbabilityPayload(
                home_win=home_win,
                draw=draw,
                away_win=away_win,
            ),
            exact_score={"home": int(round(home_goals)), "away": int(round(away_goals))},
            goals={"home": home_goals, "away": away_goals},
            btts=_logistic(min(home_goals, away_goals) - 0.15),
            over_under={
                "over_1_5": _logistic(total_goals - 1.5),
                "over_2_5": _logistic(total_goals - 2.5),
                "over_3_5": _logistic(total_goals - 3.5),
            },
            corners={"home": 5.2 + elo_diff, "away": 4.6 - elo_diff},
            cards={"home": 1.8, "away": 1.9},
            shots={"home": 13.5 + elo_diff * 2.5, "away": 10.8 - elo_diff * 2.0},
            shots_on_target={"home": 4.9 + elo_diff, "away": 3.7 - elo_diff * 0.8},
            xg={"home": home_goals + 0.2, "away": away_goals + 0.15},
            possession={"home": 52.5 + elo_diff * 6.0, "away": 47.5 - elo_diff * 6.0},
            match_context=_build_match_context(payload),
        )
