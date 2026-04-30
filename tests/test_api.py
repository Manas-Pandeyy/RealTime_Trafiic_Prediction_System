from datetime import datetime

from fastapi.testclient import TestClient

from src.api import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_optimize_route() -> None:
    payload = {"source": "H1", "destination": "C3", "source_congestion": "High", "top_k": 3}
    response = client.post("/optimize-route", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["best_path"]) >= 2
    assert body["best_eta_min"] > 0
    assert len(body["alternative_routes"]) >= 1


def test_predict() -> None:
    payload = {
        "timestamp": datetime.now().isoformat(),
        "location": "C1",
        "latitude": 12.97,
        "longitude": 77.59,
        "vehicle_count": 95,
        "weather": "rain",
        "road_type": "city",
        "distance_km": 9.0,
        "is_holiday": 0,
        "is_special_event": 1,
        "traffic_speed": 26.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["congestion_level"] in {"Low", "Medium", "High"}
    assert body["predicted_travel_time_min"] > 0
    assert 0 <= body["confidence_score"] <= 1


def test_optimize_route_osm_validation_error() -> None:
    payload = {
        "place_name": "Bengaluru, Karnataka, India",
        "source_lat": 12.97,
        "source_lon": 77.59,
        "destination_lat": 12.93,
        "destination_lon": 77.62,
        "congestion_level": "INVALID",
    }
    response = client.post("/optimize-route-osm", json=payload)
    assert response.status_code == 422
