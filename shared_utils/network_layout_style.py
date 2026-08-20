# -*- coding: utf-8 -*-
"""Shared network layout and node-size style for Figure 2.

This module intentionally contains only spatial/size rules:

- clustering-coefficient based node size;
- circular network layout with dataset-specific scale.

It does not contain node colors, edge colors, edge widths, legends, or figure
size settings. Those remain controlled by each figure's own plotting/config
code.
"""

from __future__ import annotations

import networkx as nx
import numpy as np

#NODE_SIZE_MIN = 400#节点大小
#NODE_SIZE_MAX = 1200
#NODE_SIZE_FALLBACK = 800

NODE_SIZE_MIN = 350
NODE_SIZE_MAX = 900
NODE_SIZE_FALLBACK = 600

LAYOUT_SCALE = {
    "ligue1": 1.55,
    "euro": 1.00,
    "worldcup": 1.00,
}


def node_sizes_from_clustering(graph: nx.Graph) -> list[float]:
    """Map node clustering coefficient to network node size.

    The mapping follows the Figure2 random-network visual rule:

    size = size_min + normalized_clustering * (size_max - size_min)

    If all clustering coefficients are identical, all nodes use the fallback
    size of 800.
    """
    clustering = nx.clustering(graph)
    values = [float(clustering.get(node, 0.0)) for node in graph.nodes]
    if not values:
        return []
    min_clustering = min(values)
    max_clustering = max(values)
    if np.isclose(max_clustering, min_clustering):
        return [NODE_SIZE_FALLBACK for _ in graph.nodes]
    return [
        NODE_SIZE_MIN
        + (value - min_clustering) / (max_clustering - min_clustering) * (NODE_SIZE_MAX - NODE_SIZE_MIN)
        for value in values
    ]


def get_layout_scale(dataset_key: str) -> float:
    """Return dataset-specific circular layout scale."""
    key = str(dataset_key).lower()
    if key not in LAYOUT_SCALE:
        raise KeyError(f"Unknown dataset layout scale: {dataset_key}")
    return LAYOUT_SCALE[key]


def get_circular_layout(graph: nx.Graph, dataset_key: str) -> dict:
    """Return Figure2 circular layout using dataset-specific scale."""
    return nx.circular_layout(graph, scale=get_layout_scale(dataset_key))
