from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import osmnx as ox


@dataclass
class OSMRouteResult:
    node_path: List[int]
    coordinate_path: List[Tuple[float, float]]
    distance_km: float
    eta_min: float


def _avg_speed_from_congestion(congestion_level: str) -> float:
    speeds = {"Low": 38.0, "Medium": 28.0, "High": 18.0}
    return speeds.get(congestion_level, 28.0)


def get_drive_graph(place_name: str) -> nx.MultiDiGraph:
    return ox.graph_from_place(place_name, network_type="drive")


def nearest_nodes_for_coords(
    graph: nx.MultiDiGraph, source_lat: float, source_lon: float, dest_lat: float, dest_lon: float
) -> Tuple[int, int]:
    source_node = ox.distance.nearest_nodes(graph, source_lon, source_lat)
    dest_node = ox.distance.nearest_nodes(graph, dest_lon, dest_lat)
    return int(source_node), int(dest_node)


def _path_distance_m(graph: nx.MultiDiGraph, path: List[int]) -> float:
    total = 0.0
    for u, v in zip(path[:-1], path[1:]):
        edge_data: Dict = graph.get_edge_data(u, v)
        if not edge_data:
            continue
        first_key = next(iter(edge_data.keys()))
        total += float(edge_data[first_key].get("length", 0.0))
    return total


def route_from_coordinates(
    place_name: str,
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
    congestion_level: str = "Medium",
) -> OSMRouteResult:
    graph = get_drive_graph(place_name)
    source_node, dest_node = nearest_nodes_for_coords(
        graph, source_lat, source_lon, dest_lat, dest_lon
    )
    node_path = nx.shortest_path(graph, source_node, dest_node, weight="length")
    distance_m = _path_distance_m(graph, node_path)
    distance_km = round(distance_m / 1000.0, 3)
    speed = _avg_speed_from_congestion(congestion_level)
    eta_min = round((distance_km / max(speed, 5.0)) * 60, 2)
    coordinates = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in node_path]
    return OSMRouteResult(
        node_path=node_path,
        coordinate_path=coordinates,
        distance_km=distance_km,
        eta_min=eta_min,
    )


def alternative_routes_from_coordinates(
    place_name: str,
    source_lat: float,
    source_lon: float,
    dest_lat: float,
    dest_lon: float,
    congestion_level: str = "Medium",
    top_k: int = 3,
) -> List[OSMRouteResult]:
    graph = get_drive_graph(place_name)
    source_node, dest_node = nearest_nodes_for_coords(
        graph, source_lat, source_lon, dest_lat, dest_lon
    )
    speed = _avg_speed_from_congestion(congestion_level)

    routes: List[OSMRouteResult] = []
    try:
        candidate_paths = nx.shortest_simple_paths(
            graph, source_node, dest_node, weight="length"
        )
        for idx, node_path in enumerate(candidate_paths):
            if idx >= top_k:
                break
            distance_m = _path_distance_m(graph, node_path)
            distance_km = round(distance_m / 1000.0, 3)
            eta_min = round((distance_km / max(speed, 5.0)) * 60, 2)
            coordinates = [(graph.nodes[n]["y"], graph.nodes[n]["x"]) for n in node_path]
            routes.append(
                OSMRouteResult(
                    node_path=list(node_path),
                    coordinate_path=coordinates,
                    distance_km=distance_km,
                    eta_min=eta_min,
                )
            )
    except nx.NetworkXNoPath:
        return []

    return routes
