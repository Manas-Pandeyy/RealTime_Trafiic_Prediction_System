from __future__ import annotations

from datetime import datetime
from typing import Dict, Iterator

import numpy as np
import pandas as pd


def simulate_live_traffic(
    zones: list[str], weather: str, iterations: int = 30, seed: int = 10
) -> Iterator[Dict[str, object]]:
    rng = np.random.default_rng(seed)
    for _ in range(iterations):
        now = datetime.now()
        hour = now.hour
        peak_multiplier = 1.4 if hour in [7, 8, 9, 17, 18, 19] else 0.9
        weather_penalty = {"clear": 0, "fog": 7, "rain": 15}.get(weather.lower(), 0)

        zone = rng.choice(zones)
        vehicle_count = int(max(8, rng.normal(70 * peak_multiplier, 15)))
        traffic_speed = float(max(7, rng.normal(48 - weather_penalty - vehicle_count * 0.12, 4)))

        yield {
            "timestamp": now,
            "zone_id": zone,
            "vehicle_count": vehicle_count,
            "traffic_speed": round(traffic_speed, 2),
            "weather": weather.lower(),
        }


def to_frame(stream_items: list[Dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(stream_items)
