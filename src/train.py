from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    from .data_generator import generate_synthetic_traffic_data
    from .preprocess import TrafficPreprocessor
except ImportError:
    from data_generator import generate_synthetic_traffic_data
    from preprocess import TrafficPreprocessor

try:
    from xgboost import XGBRegressor  # type: ignore

    XGBOOST_AVAILABLE = True
except Exception:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow.keras import Sequential  # type: ignore
    from tensorflow.keras.layers import GRU, LSTM, Dense  # type: ignore

    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False


def load_or_generate_data(dataset_path: str) -> pd.DataFrame:
    path = Path(dataset_path)
    if path.exists():
        return pd.read_csv(path)
    return generate_synthetic_traffic_data(output_path=dataset_path)


def build_models(random_state: int = 42) -> Dict[str, object]:
    models: Dict[str, object] = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=250, max_depth=14, random_state=random_state, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=random_state),
    }
    if XGBOOST_AVAILABLE:
        models["xgboost"] = XGBRegressor(
            n_estimators=350,
            learning_rate=0.06,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    return models


def evaluate_model(
    pipeline: Pipeline, x_test: pd.DataFrame, y_test: pd.Series
) -> Dict[str, float]:
    preds = pipeline.predict(x_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    r2 = float(r2_score(y_test, preds))
    return {"rmse": rmse, "mae": mae, "r2": r2}


def train_models(
    dataset_path: str = "data/traffic_data.csv",
    model_output_path: str = "artifacts/best_model.joblib",
    metrics_output_path: str = "artifacts/model_metrics.json",
) -> Tuple[str, Dict[str, dict]]:
    df = load_or_generate_data(dataset_path)
    prep = TrafficPreprocessor()
    prepared = prep.prepare(df)

    x_train, x_test, y_train, y_test = train_test_split(
        prepared.features, prepared.target, test_size=0.2, random_state=42
    )

    results: Dict[str, dict] = {}
    best_model_name = ""
    best_rmse = float("inf")
    best_pipeline = None

    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    mlflow_enabled = MLFLOW_AVAILABLE
    if mlflow_enabled:
        try:
            mlflow.set_tracking_uri("file:./artifacts/mlruns")
            mlflow.set_experiment("traffic_prediction_training")
        except Exception as exc:
            print(f"MLflow setup failed, continuing without tracking: {exc}")
            mlflow_enabled = False

    for model_name, model in build_models().items():
        if mlflow_enabled:
            try:
                with mlflow.start_run(run_name=model_name):
                    pipeline = Pipeline(
                        [("preprocessor", prepared.preprocessor), ("model", model)]
                    )
                    cv_results = cross_validate(
                        pipeline,
                        x_train,
                        y_train,
                        cv=cv,
                        scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
                    )

                    pipeline.fit(x_train, y_train)
                    holdout = evaluate_model(pipeline, x_test, y_test)
                    cv_rmse = float(-np.mean(cv_results["test_neg_root_mean_squared_error"]))
                    cv_mae = float(-np.mean(cv_results["test_neg_mean_absolute_error"]))
                    cv_r2 = float(np.mean(cv_results["test_r2"]))

                    mlflow.log_params(
                        {
                            "model_name": model_name,
                            "train_rows": len(x_train),
                            "test_rows": len(x_test),
                        }
                    )
                    mlflow.log_metrics(
                        {
                            "cv_rmse": cv_rmse,
                            "cv_mae": cv_mae,
                            "cv_r2": cv_r2,
                            "holdout_rmse": holdout["rmse"],
                            "holdout_mae": holdout["mae"],
                            "holdout_r2": holdout["r2"],
                        }
                    )
                    mlflow.set_tag("project", "traffic_prediction_system")
            except Exception as exc:
                print(f"MLflow run failed for {model_name}, using local-only logging: {exc}")
                mlflow_enabled = False

        if not mlflow_enabled:
            pipeline = Pipeline(
                [("preprocessor", prepared.preprocessor), ("model", model)]
            )
            cv_results = cross_validate(
                pipeline,
                x_train,
                y_train,
                cv=cv,
                scoring=("neg_root_mean_squared_error", "neg_mean_absolute_error", "r2"),
            )

            pipeline.fit(x_train, y_train)
            holdout = evaluate_model(pipeline, x_test, y_test)
            cv_rmse = float(-np.mean(cv_results["test_neg_root_mean_squared_error"]))
            cv_mae = float(-np.mean(cv_results["test_neg_mean_absolute_error"]))
            cv_r2 = float(np.mean(cv_results["test_r2"]))

        results[model_name] = {
            "cv_rmse": cv_rmse,
            "cv_mae": cv_mae,
            "cv_r2": cv_r2,
            "holdout_rmse": holdout["rmse"],
            "holdout_mae": holdout["mae"],
            "holdout_r2": holdout["r2"],
        }

        if cv_rmse < best_rmse:
            best_rmse = cv_rmse
            best_model_name = model_name
            best_pipeline = pipeline

    if best_pipeline is None:
        raise RuntimeError("Model training failed to select a best model.")

    artifact = {
        "pipeline": best_pipeline,
        "best_model_name": best_model_name,
        "metrics": results,
        "feature_columns": prepared.feature_columns,
    }

    output_path = Path(model_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, output_path)
    Path(metrics_output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return best_model_name, results


def train_optional_lstm_gru(dataset_path: str = "data/traffic_data.csv") -> None:
    if not TF_AVAILABLE:
        print("TensorFlow not installed. Skipping LSTM/GRU training.")
        return

    df = load_or_generate_data(dataset_path)
    prep = TrafficPreprocessor()
    prepared = prep.prepare(df)
    x = prepared.features[["traffic_speed", "vehicle_count", "distance_km"]].values
    y = prepared.target.values

    # Lightweight sequence shaping for demonstration purposes.
    x_seq = x.reshape((x.shape[0], 1, x.shape[1]))

    lstm_model = Sequential([LSTM(32, input_shape=(1, x.shape[1])), Dense(1)])
    lstm_model.compile(optimizer="adam", loss="mse")
    lstm_model.fit(x_seq, y, epochs=3, batch_size=64, verbose=0)

    gru_model = Sequential([GRU(32, input_shape=(1, x.shape[1])), Dense(1)])
    gru_model.compile(optimizer="adam", loss="mse")
    gru_model.fit(x_seq, y, epochs=3, batch_size=64, verbose=0)

    print("Optional LSTM/GRU models trained (demo configuration).")


if __name__ == "__main__":
    best_name, all_metrics = train_models()
    print(f"Best model selected: {best_name}")
    print(json.dumps(all_metrics[best_name], indent=2))
    if MLFLOW_AVAILABLE:
        print("MLflow runs logged to artifacts/mlruns")
