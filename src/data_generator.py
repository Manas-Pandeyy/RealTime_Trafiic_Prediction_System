from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd


def _road_type_from_zone(zone_id: str) -> str:
    if zone_id.startswith("H"):
        return "highway"
    if zone_id.startswith("C"):
        return "city"
    return "rural"


def generate_synthetic_traffic_data(
    output_path: str = "data/traffic_data.csv",
    n_days: int = 45,
    zones: List[str] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    zones = zones or ["H1", "H2", "C1", "C2", "C3", "R1", "R2"]
    weather_states = ["clear", "rain", "fog"]

    start_time = datetime.now() - timedelta(days=n_days)
    rows = []

    for hour_idx in range(n_days * 24):
        current_ts = start_time + timedelta(hours=hour_idx)
        hour = current_ts.hour
        is_weekend = int(current_ts.weekday() >= 5)
        is_holiday = int(current_ts.day in [1, 15])
        is_special_event = int(current_ts.weekday() in [4, 5] and hour in [18, 19, 20])

        for zone_id in zones:
            road_type = _road_type_from_zone(zone_id)
            lat = 12.85 + rng.uniform(-0.15, 0.15)
            lon = 77.55 + rng.uniform(-0.15, 0.15)

            weather = rng.choice(weather_states, p=[0.65, 0.25, 0.10])
            base_flow = {"highway": 90, "city": 70, "rural": 40}[road_type]

            rush_boost = 45 if hour in [7, 8, 9, 17, 18, 19] else 0
            weekend_drop = -15 if is_weekend else 0
            event_boost = 30 if is_special_event else 0
            holiday_drop = -10 if is_holiday else 0
            weather_boost = {"clear": 0, "fog": 8, "rain": 18}[weather]

            vehicle_count = max(
                10,
                int(
                    base_flow
                    + rush_boost
                    + weather_boost
                    + event_boost
                    + weekend_drop
                    + holiday_drop
                    + rng.normal(0, 8)
                ),
            )

            speed_base = {"highway": 70, "city": 40, "rural": 55}[road_type]
            speed_penalty = vehicle_count * 0.18 + weather_boost * 0.7
            traffic_speed = max(8, speed_base - speed_penalty + rng.normal(0, 4))

            distance_km = max(1.0, rng.normal(7.5, 2.0))
            travel_time_min = max(3.0, (distance_km / max(traffic_speed, 1)) * 60)
            travel_time_min += 0.2 * vehicle_count / 10 + rng.normal(0, 1.5)

            rows.append(
                {
                    "timestamp": current_ts,
                    "zone_id": zone_id,
                    "latitude": lat,
                    "longitude": lon,
                    "traffic_speed": round(float(traffic_speed), 2),
                    "vehicle_count": int(vehicle_count),
                    "road_type": road_type,
                    "weather": weather,
                    "is_holiday": is_holiday,
                    "is_special_event": is_special_event,
                    "distance_km": round(float(distance_km), 2),
                    "travel_time_min": round(float(travel_time_min), 2),
                }
            )

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    return df


if __name__ == "__main__":
    generated = generate_synthetic_traffic_data()
    print(f"Generated dataset with {len(generated)} rows.")
