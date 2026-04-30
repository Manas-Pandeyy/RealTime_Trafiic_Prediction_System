from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .predict import TrafficPredictor
from .osm_router import route_from_coordinates
from .route_optimizer import (
    apply_congestion_to_graph,
    astar_route,
    build_sample_road_graph,
    optimize_routes,
)


app = FastAPI(
    title="Traffic Prediction and Route Optimization API",
    version="1.0.0",
    description="Production-style API for traffic ETA prediction and route planning.",
)


class PredictRequest(BaseModel):
    timestamp: datetime
    location: str = Field(..., examples=["C1"])
    latitude: float = 12.97
    longitude: float = 77.59
    vehicle_count: int = 90
    weather: str = "clear"
    road_type: str = "city"
    distance_km: float = 8.0
    is_holiday: int = 0
    is_special_event: int = 0
    traffic_speed: float = 35.0


class RouteRequest(BaseModel):
    source: str = Field(..., examples=["H1"])
    destination: str = Field(..., examples=["C3"])
    source_congestion: str = Field(default="Medium", pattern="^(Low|Medium|High)$")
    top_k: int = Field(default=3, ge=1, le=5)


class OSMRouteRequest(BaseModel):
    place_name: str = Field(default="Bengaluru, Karnataka, India")
    source_lat: float
    source_lon: float
    destination_lat: float
    destination_lon: float
    congestion_level: str = Field(default="Medium", pattern="^(Low|Medium|High)$")


def get_predictor() -> TrafficPredictor:
    model_path = Path("artifacts/best_model.joblib")
    if not model_path.exists():
        raise HTTPException(
            status_code=500,
            detail="Model artifact not found. Run `python src/train.py` first.",
        )
    return TrafficPredictor(str(model_path))


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict")
def predict_traffic(payload: PredictRequest) -> Dict[str, object]:
    predictor = get_predictor()
    model_input = predictor.build_input(
        timestamp=payload.timestamp,
        location=payload.location,
        latitude=payload.latitude,
        longitude=payload.longitude,
        vehicle_count=payload.vehicle_count,
        weather=payload.weather,
        road_type=payload.road_type,
        distance_km=payload.distance_km,
        is_holiday=payload.is_holiday,
        is_special_event=payload.is_special_event,
        traffic_speed=payload.traffic_speed,
    )
    result = predictor.predict(model_input)
    return {
        "congestion_level": result.congestion_level,
        "predicted_travel_time_min": result.predicted_travel_time_min,
        "confidence_score": result.confidence_score,
        "explanation": result.explanation,
    }


@app.post("/optimize-route")
def optimize_route(payload: RouteRequest) -> Dict[str, object]:
    graph = build_sample_road_graph()
    zone_congestion = {node: "Low" for node in graph.nodes}
    zone_congestion[payload.source] = payload.source_congestion
    weighted = apply_congestion_to_graph(graph, zone_congestion)

    try:
        plan = optimize_routes(weighted, payload.source, payload.destination, payload.top_k)
        a_star_path = astar_route(weighted, payload.source, payload.destination)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    alternatives: List[Dict[str, object]] = [
        {"path": path, "eta_min": eta} for path, eta in plan.alternative_paths
    ]
    return {
        "best_path": plan.best_path,
        "best_eta_min": plan.best_eta_min,
        "a_star_path": a_star_path,
        "alternative_routes": alternatives,
    }


@app.post("/optimize-route-osm")
def optimize_route_osm(payload: OSMRouteRequest) -> Dict[str, object]:
    try:
        result = route_from_coordinates(
            place_name=payload.place_name,
            source_lat=payload.source_lat,
            source_lon=payload.source_lon,
            dest_lat=payload.destination_lat,
            dest_lon=payload.destination_lon,
            congestion_level=payload.congestion_level,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"OSM route failed: {exc}") from exc

    return {
        "distance_km": result.distance_km,
        "eta_min": result.eta_min,
        "node_path": result.node_path,
        "coordinate_path": result.coordinate_path,
    }
