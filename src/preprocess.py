from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


REQUIRED_COLUMNS = [
    "timestamp",
    "zone_id",
    "latitude",
    "longitude",
    "traffic_speed",
    "vehicle_count",
    "road_type",
    "weather",
    "is_holiday",
    "is_special_event",
    "distance_km",
    "travel_time_min",
]


@dataclass
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    preprocessor: ColumnTransformer
    feature_columns: List[str]


class TrafficPreprocessor:
    def __init__(self) -> None:
        self.numeric_columns = [
            "latitude",
            "longitude",
            "traffic_speed",
            "vehicle_count",
            "distance_km",
            "hour",
            "dayofweek",
            "is_weekend",
            "is_peak_hour",
            "rush_hour_flag",
            "weather_severity",
        ]
        self.categorical_columns = ["zone_id", "road_type", "weather", "day_type"]

    @staticmethod
    def validate_input(df: pd.DataFrame) -> None:
        missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Dataset missing required columns: {missing}")

    @staticmethod
    def remove_outliers_iqr(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        cleaned_df = df.copy()
        mask = pd.Series(True, index=cleaned_df.index)
        for col in columns:
            q1 = cleaned_df[col].quantile(0.25)
            q3 = cleaned_df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            mask &= cleaned_df[col].between(lower_bound, upper_bound)
        return cleaned_df.loc[mask].reset_index(drop=True)

    @staticmethod
    def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["hour"] = df["timestamp"].dt.hour
        df["dayofweek"] = df["timestamp"].dt.dayofweek
        df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
        df["is_peak_hour"] = df["hour"].isin([7, 8, 9, 17, 18, 19]).astype(int)
        df["rush_hour_flag"] = df["hour"].isin([8, 9, 18]).astype(int)
        df["day_type"] = np.where(df["is_weekend"] == 1, "weekend", "weekday")
        return df

    @staticmethod
    def add_weather_features(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        severity_map = {"clear": 0, "fog": 1, "rain": 2}
        df["weather"] = df["weather"].str.lower().fillna("clear")
        df["weather_severity"] = df["weather"].map(severity_map).fillna(0)
        return df

    def build_preprocessor(self) -> ColumnTransformer:
        numeric_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        categorical_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        return ColumnTransformer(
            transformers=[
                ("num", numeric_pipe, self.numeric_columns),
                ("cat", categorical_pipe, self.categorical_columns),
            ]
        )

    def prepare(self, df: pd.DataFrame) -> PreparedData:
        self.validate_input(df)
        transformed = self.add_time_features(df)
        transformed = self.add_weather_features(transformed)
        transformed = self.remove_outliers_iqr(
            transformed, columns=["traffic_speed", "vehicle_count", "travel_time_min"]
        )
        transformed["is_holiday"] = transformed["is_holiday"].fillna(0).astype(int)
        transformed["is_special_event"] = (
            transformed["is_special_event"].fillna(0).astype(int)
        )

        feature_columns = [
            *self.numeric_columns,
            *self.categorical_columns,
            "is_holiday",
            "is_special_event",
        ]
        features = transformed[feature_columns]
        target = transformed["travel_time_min"]
        preprocessor = self.build_preprocessor()

        return PreparedData(
            features=features,
            target=target,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
        )


def split_features_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    if "travel_time_min" not in df.columns:
        raise ValueError("Expected 'travel_time_min' column for target.")
    return df.drop(columns=["travel_time_min"]), df["travel_time_min"]
