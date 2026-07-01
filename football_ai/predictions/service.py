"""Prediction service for model inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import math

import joblib
import pandas as pd

from football_ai.models.base import TrainedArtifact
from football_ai.models.registry import ModelRegistry
from football_ai.predictions.schemas import MatchContextPayload, PredictionResponse, ProbabilityPayload


def _logistic(value: float) -> float:
    """Apply a logistic transformation."""
    return 1.0 / (1.0 + math.exp(-value))


@dataclass(slots=True)
class PredictionService:
    """Prediction service that loads trained artifacts when available."""

    registry: ModelRegistry

    @classmethod
    def create(cls) -> "PredictionService":
        """Create a prediction service with the default registry."""
        return cls(registry=ModelRegistry())

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
        btts = self._predict_binary("btts", match_frame, default=float(home_goals > 0 and away_goals > 0))
        over_under = {
            "over_1_5": self._predict_binary("over_1_5", match_frame, default=_logistic(total_goals - 1.5)),
            "over_2_5": self._predict_binary("over_2_5", match_frame, default=_logistic(total_goals - 2.5)),
            "over_3_5": self._predict_binary("over_3_5", match_frame, default=_logistic(total_goals - 3.5)),
        }
        return PredictionResponse(
            model_version=registry_entry.version,
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
        )

    def _build_feature_frame(self, payload: MatchContextPayload) -> pd.DataFrame:
        """Build a single-row feature frame for inference."""
        frame = pd.DataFrame(
            [
                {
                    "match_id": 1,
                    "match_date": payload.match_date,
                    "home_team_name": payload.home_team_name,
                    "away_team_name": payload.away_team_name,
                    "tournament": payload.tournament,
                    "stadium_name": payload.stadium_name,
                    "country": payload.country,
                    "neutral_venue": payload.neutral_venue,
                    "home_elo_pre_match": payload.home_elo or 1500.0,
                    "away_elo_pre_match": payload.away_elo or 1500.0,
                    "home_elo": payload.home_elo or 1500.0,
                    "away_elo": payload.away_elo or 1500.0,
                    "home_lineup_average_age": payload.home_lineup_average_age or 27.0,
                    "away_lineup_average_age": payload.away_lineup_average_age or 27.0,
                    "home_lineup_market_value": payload.home_lineup_market_value or 100000000.0,
                    "away_lineup_market_value": payload.away_lineup_market_value or 100000000.0,
                    "home_goalkeeper_form": payload.home_goalkeeper_form or 0.0,
                    "away_goalkeeper_form": payload.away_goalkeeper_form or 0.0,
                    "home_forward_form": payload.home_forward_form or 0.0,
                    "away_forward_form": payload.away_forward_form or 0.0,
                    "home_travel_distance_km": payload.home_travel_distance_km or 0.0,
                    "away_travel_distance_km": payload.away_travel_distance_km or 0.0,
                    "home_xg_mean_5": payload.home_recent_xg or 1.5,
                    "away_xg_mean_5": payload.away_recent_xg or 1.2,
                    "home_xga_mean_5": payload.home_recent_xga or 1.1,
                    "away_xga_mean_5": payload.away_recent_xga or 1.3,
                    "weather_temperature_c": payload.weather_temperature_c or 18.0,
                    "weather_precipitation_mm": payload.weather_precipitation_mm or 0.0,
                }
            ]
        )
        return frame

    def _load_artifact(self, artifact_path: str | Path) -> TrainedArtifact:
        """Load a serialized model artifact."""
        return joblib.load(Path(artifact_path))

    def _align_features(
        self,
        frame: pd.DataFrame,
        artifact_path: str | Path,
    ) -> tuple[pd.DataFrame, TrainedArtifact]:
        """Load an artifact and align inference features to the trained schema."""
        loaded_artifact = self._load_artifact(artifact_path)
        aligned = frame.copy()
        for column in loaded_artifact.feature_columns:
            if column not in aligned.columns:
                aligned[column] = 0.0
        return aligned[loaded_artifact.feature_columns], loaded_artifact

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
        probabilities = artifact.estimator.predict_proba(aligned)[0]
        label_mapping = dict(zip(getattr(artifact.estimator, "classes_", labels), probabilities, strict=False))
        return {
            "home_win": float(label_mapping.get("home_win", label_mapping.get("H", 0.33))),
            "draw": float(label_mapping.get("draw", label_mapping.get("D", 0.34))),
            "away_win": float(label_mapping.get("away_win", label_mapping.get("A", 0.33))),
        }

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
            return float(artifact.estimator.predict_proba(aligned)[0][1])
        return float(default)

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
        )
