from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd


@dataclass
class PredictionResult:
    congestion_level: str
    predicted_travel_time_min: float
    confidence_score: float
    explanation: Dict[str, float]


class TrafficPredictor:
    def __init__(self, model_path: str = "artifacts/best_model.joblib") -> None:
        artifact_file = Path(model_path)
        if not artifact_file.exists():
            raise FileNotFoundError(
                f"Model artifact not found at {model_path}. Run train.py first."
            )
        artifact = joblib.load(artifact_file)
        self.pipeline = artifact["pipeline"]
        self.best_model_name = artifact["best_model_name"]
        self.metrics = artifact["metrics"]

    @staticmethod
    def _congestion_from_time(travel_time_min: float) -> str:
        if travel_time_min < 12:
            return "Low"
        if travel_time_min < 20:
            return "Medium"
        return "High"

    def _confidence(self, predicted_time: float) -> float:
        rmse = self.metrics[self.best_model_name]["cv_rmse"]
        confidence = 1.0 - min(1.0, rmse / max(predicted_time, 1))
        return float(round(confidence, 3))

    @staticmethod
    def build_input(
        timestamp: datetime,
        location: str,
        latitude: float,
        longitude: float,
        vehicle_count: int,
        weather: str,
        road_type: str,
        distance_km: float,
        is_holiday: int = 0,
        is_special_event: int = 0,
        traffic_speed: float = 30.0,
    ) -> pd.DataFrame:
        hour = timestamp.hour
        dayofweek = timestamp.weekday()
        is_weekend = int(dayofweek >= 5)
        is_peak_hour = int(hour in [7, 8, 9, 17, 18, 19])
        rush_hour_flag = int(hour in [8, 9, 18])
        day_type = "weekend" if is_weekend else "weekday"
        weather_lower = weather.lower()
        weather_severity = {"clear": 0, "fog": 1, "rain": 2}.get(weather_lower, 0)

        row = {
            "latitude": latitude,
            "longitude": longitude,
            "traffic_speed": traffic_speed,
            "vehicle_count": vehicle_count,
            "distance_km": distance_km,
            "hour": hour,
            "dayofweek": dayofweek,
            "is_weekend": is_weekend,
            "is_peak_hour": is_peak_hour,
            "rush_hour_flag": rush_hour_flag,
            "weather_severity": weather_severity,
            "zone_id": location,
            "road_type": road_type,
            "weather": weather_lower,
            "day_type": day_type,
            "is_holiday": is_holiday,
            "is_special_event": is_special_event,
        }
        return pd.DataFrame([row])

    def predict(self, model_input: pd.DataFrame) -> PredictionResult:
        prediction = float(self.pipeline.predict(model_input)[0])
        congestion = self._congestion_from_time(prediction)
        confidence = self._confidence(prediction)

        explanation = {
            "vehicle_count": float(model_input["vehicle_count"].iloc[0]),
            "is_peak_hour": float(model_input["is_peak_hour"].iloc[0]),
            "weather_severity": float(model_input["weather_severity"].iloc[0]),
            "distance_km": float(model_input["distance_km"].iloc[0]),
        }

        return PredictionResult(
            congestion_level=congestion,
            predicted_travel_time_min=round(prediction, 2),
            confidence_score=confidence,
            explanation=explanation,
        )
