# -*- coding: utf-8 -*-
from __future__ import annotations

import networkx as nx
import numpy as np


NODE_SIZE_MIN = 400
NODE_SIZE_MAX = 1200
NODE_SIZE_FALLBACK = 800

LAYOUT_SCALE = {
    "ligue1": 1.55,
    "euro": 1.00,
    "worldcup": 1.18,
}


def node_sizes_from_clustering(graph: nx.Graph) -> list[float]:
    """Map node clustering coefficient to the original Figure2 size rule."""
    clustering = nx.clustering(graph)
    values = [float(clustering.get(node, 0.0)) for node in graph.nodes]
    if not values:
        return []
    low = min(values)
    high = max(values)
    if np.isclose(high, low):
        return [NODE_SIZE_FALLBACK for _ in graph.nodes]
    return [NODE_SIZE_MIN + (value - low) / (high - low) * (NODE_SIZE_MAX - NODE_SIZE_MIN) for value in values]


def get_layout_scale(dataset_key: str) -> float:
    """Return dataset-specific circular layout scale."""
    return LAYOUT_SCALE.get(str(dataset_key).lower(), 1.18)


def get_circular_layout(graph: nx.Graph, dataset_key: str) -> dict:
    """Return Figure2 circular layout using dataset-specific scale."""
    return nx.circular_layout(graph, scale=get_layout_scale(dataset_key))

