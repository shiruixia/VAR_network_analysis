from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from utils.network_layout_style import get_circular_layout, node_sizes_from_clustering
from utils.paper_color_config import BACKGROUND_COLOR, EDGE_COLORS, RANDOM_NETWORK_COLORS, REAL_NETWORK_COLORS


def graph_from_edge_list(nodes: list[str], edge_list: pd.DataFrame) -> nx.Graph:
    """Build a graph from a computed random-network edge list."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if edge_list.empty:
        return graph
    for row in edge_list.itertuples(index=False):
        weight = float(getattr(row, "weight", getattr(row, "correlation", 0.0)))
        graph.add_edge(row.source, row.target, spearman_r=weight, abs_r=abs(weight))
    return graph


def graph_from_loaded_edges(nodes: list[str], edge_list: pd.DataFrame) -> nx.Graph:
    """Build a graph from a loaded Figure1 edge list."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if edge_list.empty:
        return graph
    weight_col = "weight" if "weight" in edge_list.columns else "spearman_r"
    keep_col = "keep_strict" if "keep_strict" in edge_list.columns else None
    rows = edge_list.loc[edge_list[keep_col].astype(bool)] if keep_col is not None else edge_list
    for row in rows.itertuples(index=False):
        source = getattr(row, "source")
        target = getattr(row, "target")
        if source not in nodes or target not in nodes:
            continue
        weight = float(getattr(row, weight_col))
        graph.add_edge(source, target, spearman_r=weight, abs_r=abs(weight))
    return graph


def draw_graph_on_axis(ax, graph: nx.Graph, positions: dict[str, np.ndarray], family: dict[str, str]) -> None:
    """Draw one network using the original Figure2 visual rules."""
    positive = [(u, v) for u, v, data in graph.edges(data=True) if data["spearman_r"] >= 0]
    negative = [(u, v) for u, v, data in graph.edges(data=True) if data["spearman_r"] < 0]
    ax.set_facecolor(BACKGROUND_COLOR)
    nx.draw_networkx_nodes(
        graph,
        positions,
        node_size=node_sizes_from_clustering(graph),
        node_color=family["node_fill"],
        edgecolors=family.get("node_edge", "none"),
        linewidths=0,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive,
        width=[1.2 + 7 * graph[u][v]["abs_r"] for u, v in positive],
        edge_color=EDGE_COLORS["positive"],
        alpha=0.78,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative,
        width=[1.2 + 7 * graph[u][v]["abs_r"] for u, v in negative],
        edge_color=EDGE_COLORS["negative"],
        style="dashed",
        alpha=0.78,
        ax=ax,
    )
    coords = list(positions.values())
    if coords:
        x_values = [float(xy[0]) for xy in coords]
        y_values = [float(xy[1]) for xy in coords]
        xpad = 0.18 * (max(x_values) - min(x_values) + 1e-9)
        ypad = 0.18 * (max(y_values) - min(y_values) + 1e-9)
        ax.set_xlim(min(x_values) - xpad, max(x_values) + xpad)
        ax.set_ylim(min(y_values) - ypad, max(y_values) + ypad)
    ax.axis("off")


def save_png_and_eps(fig: plt.Figure, output_path: Path) -> None:
    """Save both PNG and EPS without changing figure styling."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight", facecolor="white")


def draw_single_network(graph: nx.Graph, positions: dict[str, np.ndarray], output_path: Path, family: dict[str, str]) -> Path:
    """Draw one standalone network figure."""
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    draw_graph_on_axis(ax, graph, positions, family)
    fig.tight_layout()
    save_png_and_eps(fig, output_path)
    plt.close(fig)
    return output_path


def draw_pair_network(graph_a: nx.Graph, graph_b: nx.Graph, positions: dict[str, np.ndarray], output_path: Path) -> Path:
    """Draw a random group A/B network pair."""
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 5.2))
    draw_graph_on_axis(axes[0], graph_a, positions, RANDOM_NETWORK_COLORS)
    draw_graph_on_axis(axes[1], graph_b, positions, RANDOM_NETWORK_COLORS)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02, wspace=0.02)
    save_png_and_eps(fig, output_path)
    plt.close(fig)
    return output_path


def positions_for_nodes(nodes: list[str], dataset_key: str) -> dict:
    """Return the dataset-specific circular layout for a node list."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    return get_circular_layout(graph, dataset_key)

