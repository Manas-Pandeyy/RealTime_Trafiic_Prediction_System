from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


@dataclass
class RoutePlan:
    best_path: List[str]
    best_eta_min: float
    alternative_paths: List[Tuple[List[str], float]]


def build_sample_road_graph(seed: int = 42) -> nx.Graph:
    rng = np.random.default_rng(seed)
    graph = nx.Graph()
    nodes = ["H1", "H2", "C1", "C2", "C3", "R1", "R2"]

    for node in nodes:
        graph.add_node(node)

    edges = [
        ("H1", "C1"),
        ("H1", "H2"),
        ("H2", "C2"),
        ("C1", "C2"),
        ("C2", "C3"),
        ("C1", "R1"),
        ("R1", "R2"),
        ("R2", "C3"),
        ("H2", "C3"),
        ("C3", "R1"),
    ]
    for u, v in edges:
        distance = float(max(2.0, rng.normal(8, 2.5)))
        graph.add_edge(u, v, distance_km=round(distance, 2), congestion_factor=1.0)
    return graph


def apply_congestion_to_graph(graph: nx.Graph, zone_congestion: Dict[str, str]) -> nx.Graph:
    updated = graph.copy()
    factor_map = {"Low": 1.0, "Medium": 1.35, "High": 1.8}
    for u, v, attrs in updated.edges(data=True):
        c_u = zone_congestion.get(u, "Low")
        c_v = zone_congestion.get(v, "Low")
        mean_factor = (factor_map[c_u] + factor_map[c_v]) / 2
        attrs["congestion_factor"] = mean_factor
        attrs["weight"] = attrs["distance_km"] * mean_factor
    return updated


def eta_for_path(graph: nx.Graph, path: List[str], avg_speed_kmph: float = 35.0) -> float:
    weighted_distance = 0.0
    for i in range(len(path) - 1):
        edge_data = graph[path[i]][path[i + 1]]
        weighted_distance += edge_data["distance_km"] * edge_data.get("congestion_factor", 1.0)
    return round((weighted_distance / max(avg_speed_kmph, 5.0)) * 60, 2)


def optimize_routes(
    graph: nx.Graph, source: str, destination: str, top_k: int = 3
) -> RoutePlan:
    if source not in graph or destination not in graph:
        raise ValueError("Source or destination not present in graph.")

    dijkstra_path = nx.dijkstra_path(graph, source, destination, weight="weight")
    best_eta = eta_for_path(graph, dijkstra_path)

    alternatives = []
    try:
        simple_paths = nx.shortest_simple_paths(graph, source, destination, weight="weight")
        for idx, path in enumerate(simple_paths):
            if idx >= top_k:
                break
            alternatives.append((path, eta_for_path(graph, path)))
    except nx.NetworkXNoPath:
        alternatives = []

    return RoutePlan(
        best_path=dijkstra_path,
        best_eta_min=best_eta,
        alternative_paths=alternatives,
    )


def astar_route(graph: nx.Graph, source: str, destination: str) -> List[str]:
    return nx.astar_path(graph, source, destination, weight="weight")
