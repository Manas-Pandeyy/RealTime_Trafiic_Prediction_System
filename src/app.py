from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from osm_router import alternative_routes_from_coordinates
from predict import TrafficPredictor
from route_optimizer import apply_congestion_to_graph, build_sample_road_graph, optimize_routes
from simulation import simulate_live_traffic

try:
    from streamlit_autorefresh import st_autorefresh

    AUTOREFRESH_AVAILABLE = True
except Exception:
    AUTOREFRESH_AVAILABLE = False

st.set_page_config(page_title="AI Traffic Prediction System", layout="wide")
st.title("Real-Time Traffic Prediction and Route Optimization")

DATA_PATH = Path("data/traffic_data.csv")
if not DATA_PATH.exists():
    st.error("Dataset missing. Run `python src/train.py` once to generate data and model.")
    st.stop()

df = pd.read_csv(DATA_PATH)
zones = sorted(df["zone_id"].unique().tolist())
zone_geo = (
    df.groupby("zone_id", as_index=False)[["latitude", "longitude"]]
    .mean()
    .set_index("zone_id")
)

if "live_updates" not in st.session_state:
    st.session_state.live_updates = []

try:
    predictor = TrafficPredictor()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

left, right = st.columns(2)
with left:
    source = st.selectbox("Source Zone", zones, index=0)
    destination = st.selectbox("Destination Zone", zones, index=min(1, len(zones) - 1))
    travel_hour = st.slider("Hour of Travel", 0, 23, datetime.now().hour)
with right:
    weather = st.selectbox("Weather", ["clear", "rain", "fog"])
    vehicle_count = st.slider("Vehicle Count", 5, 220, 90)
    distance_km = st.slider("Trip Distance (km)", 1.0, 25.0, 8.0)

if st.button("Predict and Optimize"):
    timestamp = datetime.now().replace(hour=travel_hour, minute=0, second=0, microsecond=0)
    model_input = predictor.build_input(
        timestamp=timestamp,
        location=source,
        latitude=float(df["latitude"].mean()),
        longitude=float(df["longitude"].mean()),
        vehicle_count=vehicle_count,
        weather=weather,
        road_type="city",
        distance_km=float(distance_km),
        is_holiday=0,
        is_special_event=int(travel_hour in [18, 19]),
        traffic_speed=max(8, 60 - vehicle_count * 0.18),
    )
    result = predictor.predict(model_input)

    st.subheader("Prediction Output")
    metric_cols = st.columns(3)
    metric_cols[0].metric("Congestion", result.congestion_level)
    metric_cols[1].metric("Predicted Travel Time (min)", result.predicted_travel_time_min)
    metric_cols[2].metric("Confidence Score", result.confidence_score)

    if result.congestion_level == "High":
        st.warning("Heavy traffic expected. Avoid this route if possible.")

    st.caption("Explainable AI signals used for this prediction.")
    st.json(result.explanation)

    zone_congestion = {zone: "Low" for zone in zones}
    zone_congestion[source] = result.congestion_level
    graph = build_sample_road_graph()
    updated_graph = apply_congestion_to_graph(graph, zone_congestion)
    plan = optimize_routes(updated_graph, source, destination, top_k=3)

    st.subheader("Route Optimization")
    st.write(f"Fastest route: {' -> '.join(plan.best_path)}")
    st.write(f"Estimated ETA: {plan.best_eta_min} min")
    st.write("Alternative routes:")
    for idx, (path, eta) in enumerate(plan.alternative_paths[:3], start=1):
        st.write(f"{idx}. {' -> '.join(path)} | ETA: {eta} min")

st.subheader("Traffic Heatmap")
latest = df.copy()
latest["congestion"] = latest["vehicle_count"] / latest["traffic_speed"].clip(lower=1)
heatmap = px.density_mapbox(
    latest.sample(min(2000, len(latest))),
    lat="latitude",
    lon="longitude",
    z="congestion",
    radius=12,
    center={"lat": float(latest["latitude"].mean()), "lon": float(latest["longitude"].mean())},
    zoom=9,
    mapbox_style="carto-positron",
    title="Congestion Density",
    color_continuous_scale=["blue", "red"],
)
heatmap.update_layout(coloraxis_colorbar_title="Congestion")
st.plotly_chart(heatmap, use_container_width=True)

st.subheader("OSM Real Route Map")
map_col1, map_col2 = st.columns(2)
with map_col1:
    place_name = st.text_input("City / Place", value="Bengaluru, Karnataka, India")
    osm_source = st.selectbox("OSM Source Zone", zones, index=0, key="osm_source")
with map_col2:
    osm_destination = st.selectbox(
        "OSM Destination Zone", zones, index=min(1, len(zones) - 1), key="osm_destination"
    )
    osm_congestion = st.selectbox("OSM Congestion Level", ["Low", "Medium", "High"], index=1)
    osm_top_k = st.slider("OSM Route Options", 1, 3, 3)

if st.button("Show Real OSM Route"):
    try:
        src_lat = float(zone_geo.loc[osm_source, "latitude"])
        src_lon = float(zone_geo.loc[osm_source, "longitude"])
        dst_lat = float(zone_geo.loc[osm_destination, "latitude"])
        dst_lon = float(zone_geo.loc[osm_destination, "longitude"])
        osm_routes = alternative_routes_from_coordinates(
            place_name=place_name,
            source_lat=src_lat,
            source_lon=src_lon,
            dest_lat=dst_lat,
            dest_lon=dst_lon,
            congestion_level=osm_congestion,
            top_k=osm_top_k,
        )
        if not osm_routes:
            st.warning("No drivable OSM route found for selected points.")
        else:
            route_fig = go.Figure()
            route_colors = ["#ff4d4f", "#ffa940", "#40a9ff"]
            for idx, route in enumerate(osm_routes, start=1):
                route_df = pd.DataFrame(route.coordinate_path, columns=["lat", "lon"])
                route_fig.add_trace(
                    go.Scattermapbox(
                        lat=route_df["lat"],
                        lon=route_df["lon"],
                        mode="lines",
                        line={"width": 5 if idx == 1 else 4, "color": route_colors[idx - 1]},
                        name=f"Route {idx}",
                    )
                )

            route_fig.add_trace(
                go.Scattermapbox(
                    lat=[src_lat, dst_lat],
                    lon=[src_lon, dst_lon],
                    mode="markers",
                    marker={"size": 10, "color": ["#2f80ed", "#eb5757"]},
                    name="Source / Destination",
                )
            )
            route_fig.update_layout(
                mapbox_style="carto-positron",
                mapbox_zoom=10,
                mapbox_center={"lat": src_lat, "lon": src_lon},
                margin={"l": 0, "r": 0, "t": 30, "b": 0},
                title="OSM Multi-Route Comparison",
            )
            st.plotly_chart(route_fig, use_container_width=True)

            st.caption("Route comparison (best route is Route 1)")
            metric_cols = st.columns(len(osm_routes))
            for idx, route in enumerate(osm_routes, start=1):
                metric_cols[idx - 1].metric(
                    f"Route {idx}",
                    f"{route.eta_min} min",
                    f"{route.distance_km} km",
                )
    except Exception as exc:
        st.error(f"OSM route fetch failed: {exc}")

st.subheader("Real-Time Simulation")
tick_col1, tick_col2 = st.columns([2, 1])
with tick_col1:
    updates_per_tick = st.slider("Updates per live tick", 5, 60, 20, key="updates_per_tick")
with tick_col2:
    if st.button("Generate Live Tick"):
        tick_seed = int(datetime.now().timestamp()) % 10_000_000
        new_updates = list(
            simulate_live_traffic(
                zones=zones, weather=weather, iterations=updates_per_tick, seed=tick_seed
            )
        )
        st.session_state.live_updates.extend(new_updates)
        # Keep recent window only, so chart reflects current behavior.
        st.session_state.live_updates = st.session_state.live_updates[-1000:]

auto_live = st.toggle("Auto-live updates (every 3 seconds)", value=False)
if auto_live and AUTOREFRESH_AVAILABLE:
    st_autorefresh(interval=3000, key="traffic_autorefresh")
    auto_seed = int(datetime.now().timestamp()) % 10_000_000
    auto_updates = list(
        simulate_live_traffic(
            zones=zones,
            weather=weather,
            iterations=max(5, updates_per_tick // 2),
            seed=auto_seed,
        )
    )
    st.session_state.live_updates.extend(auto_updates)
    st.session_state.live_updates = st.session_state.live_updates[-1000:]
elif auto_live and not AUTOREFRESH_AVAILABLE:
    st.warning("Install `streamlit-autorefresh` for auto-live mode support.")

if st.session_state.live_updates:
    live_df = pd.DataFrame(st.session_state.live_updates)
    st.line_chart(live_df.set_index("timestamp")[["vehicle_count", "traffic_speed"]])
else:
    st.info("Click 'Generate Live Tick' to stream new traffic updates.")

st.subheader("Peak Hour Analytics")
df["hour"] = pd.to_datetime(df["timestamp"]).dt.hour
base_peak_df = df.groupby("hour", as_index=False)["vehicle_count"].mean()
base_peak_df["series"] = "Historical Avg"

if st.session_state.live_updates:
    live_peak_df = pd.DataFrame(st.session_state.live_updates).copy()
    live_peak_df["hour"] = pd.to_datetime(live_peak_df["timestamp"]).dt.hour
    live_peak_df = live_peak_df.groupby("hour", as_index=False)["vehicle_count"].mean()
    live_peak_df["series"] = "Live Window Avg"
    peak_df = pd.concat([base_peak_df, live_peak_df], ignore_index=True)
else:
    peak_df = base_peak_df

peak_chart = px.bar(
    peak_df,
    x="hour",
    y="vehicle_count",
    color="series",
    barmode="group",
    color_discrete_map={"Historical Avg": "#7db7e8", "Live Window Avg": "#ff7f50"},
    title="Average Vehicle Count by Hour (Historical vs Live)",
)
st.plotly_chart(peak_chart, use_container_width=True)
