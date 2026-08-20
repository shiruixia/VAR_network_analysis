# -*- coding: utf-8 -*-
"""Shared Figure1 network plotting helpers.

This module is visualization-only. It reads precomputed result workbooks from
``figure1/result`` and draws manuscript network panels. It does not recompute
correlations, p-values, FDR correction, or adjacency matrices.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import networkx as nx
import numpy as np
import pandas as pd

from utils.indicator_config import MATCH_INDICATORS
from utils.paper_color_config import (
    BACKGROUND_COLOR,
    EDGE_COLORS,
    FIGURE1_NODE_COLOR,
    LEGEND_TEXT_COLOR,
    NETWORK_STYLE,
    figure1_node_display_label,
)


FIGURE1_ROOT = Path(__file__).resolve().parents[2]
SHARED_UTILS_DIR = FIGURE1_ROOT.parent / "shared_utils"
if str(SHARED_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_UTILS_DIR))

try:
    from network_layout_style import get_circular_layout, node_sizes_from_clustering
except ModuleNotFoundError:
    def get_circular_layout(graph: nx.Graph, dataset_key: str) -> dict:
        return nx.circular_layout(graph, scale=NETWORK_STYLE["layout_scale"][str(dataset_key).lower()])

    def node_sizes_from_clustering(graph: nx.Graph) -> list[float]:
        return [NETWORK_STYLE["node_size"] for _ in graph.nodes]


POSITIVE_EDGE_COLOR = EDGE_COLORS["positive"]
NEGATIVE_EDGE_COLOR = EDGE_COLORS["negative"]
NETWORK_FACE_COLOR = BACKGROUND_COLOR
PANEL_FIGSIZE = NETWORK_STYLE["figsize"]
AXIS_PADDING = NETWORK_STYLE["axis_padding"]
MIN_EDGE_WIDTH = 1.1
EDGE_WIDTH_SCALE = 6.0


def read_matrix_sheet(workbook: str | Path, sheet_name: str) -> pd.DataFrame:
    """Read a matrix workbook sheet saved by the Figure1 result pipeline."""
    path = Path(workbook)
    if not path.exists():
        raise FileNotFoundError(f"Required Figure1 matrix workbook not found: {path}")
    frame = pd.read_excel(path, sheet_name=sheet_name)
    if "metric" in frame.columns:
        frame = frame.set_index("metric")
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return frame


def read_nodes(correlation_matrix: pd.DataFrame, node_order: list[str] | None = None) -> list[str]:
    """Return the plotted nodes using saved matrix order and the shared nine indicators."""
    nodes_from_matrix = [str(node) for node in correlation_matrix.index.tolist()]
    order = node_order or MATCH_INDICATORS
    available = set(nodes_from_matrix)
    ordered = [node for node in order if node in available]
    extras = [node for node in nodes_from_matrix if node not in ordered]
    nodes = ordered + extras
    if len(nodes) != len(MATCH_INDICATORS) or set(nodes) != set(MATCH_INDICATORS):
        raise ValueError(f"Figure1 panel must contain the shared nine indicators; found: {nodes}")
    return nodes


def read_edge_list(workbook: str | Path, sheet_name: str, nodes: list[str]) -> pd.DataFrame:
    """Read a strict edge-list sheet saved by the Figure1 result pipeline."""
    path = Path(workbook)
    if not path.exists():
        raise FileNotFoundError(f"Required Figure1 edge-list workbook not found: {path}")
    edges = pd.read_excel(path, sheet_name=sheet_name)
    required = {"source", "target"}
    missing = required - set(edges.columns)
    if missing:
        raise ValueError(f"Edge list missing required columns: {sorted(missing)}")
    if "spearman_r" not in edges.columns and "weight" in edges.columns:
        edges["spearman_r"] = edges["weight"]
    if "abs_r" not in edges.columns:
        edges["abs_r"] = edges["spearman_r"].abs()
    if "sign" not in edges.columns:
        edges["sign"] = np.where(edges["spearman_r"] >= 0, "positive", "negative")
    if "keep_strict" not in edges.columns:
        edges["keep_strict"] = True
    node_set = set(nodes)
    return edges.loc[
        edges["source"].astype(str).isin(node_set)
        & edges["target"].astype(str).isin(node_set)
    ].copy()


def graph_from_edge_list(nodes: list[str], edge_list: pd.DataFrame) -> nx.Graph:
    """Build the plotted NetworkX graph from precomputed strict edges."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if edge_list.empty:
        return graph
    selected = edge_list.loc[edge_list["keep_strict"].astype(bool)].copy()
    for row in selected.itertuples(index=False):
        r = float(row.spearman_r)
        abs_r = float(row.abs_r)
        graph.add_edge(
            str(row.source),
            str(row.target),
            spearman_r=r,
            abs_r=abs_r,
            sign="positive" if r >= 0 else "negative",
        )
    return graph


def draw_network_panel(ax, spec, add_panel_label: bool = False) -> dict[str, object]:
    """Draw one before/after VAR network panel from saved result workbooks."""
    correlation = read_matrix_sheet(spec.correlation_workbook, spec.sheet_name)
    nodes = read_nodes(correlation, spec.node_order)
    edges = read_edge_list(spec.edge_workbook, spec.sheet_name, nodes)
    graph = graph_from_edge_list(nodes, edges)
    positions = get_circular_layout(graph, spec.dataset)

    positive = [(u, v) for u, v, data in graph.edges(data=True) if data["spearman_r"] >= 0]
    negative = [(u, v) for u, v, data in graph.edges(data=True) if data["spearman_r"] < 0]
    positive_widths = [MIN_EDGE_WIDTH + EDGE_WIDTH_SCALE * graph[u][v]["abs_r"] for u, v in positive]
    negative_widths = [MIN_EDGE_WIDTH + EDGE_WIDTH_SCALE * graph[u][v]["abs_r"] for u, v in negative]

    ax.set_facecolor(NETWORK_FACE_COLOR)
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=positive,
        width=positive_widths,
        edge_color=POSITIVE_EDGE_COLOR,
        alpha=0.78,
        ax=ax,
    )
    nx.draw_networkx_edges(
        graph,
        positions,
        edgelist=negative,
        width=negative_widths,
        edge_color=NEGATIVE_EDGE_COLOR,
        style="dashed",
        alpha=0.78,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=nodes,
        node_size=node_sizes_from_clustering(graph),
        node_color=[FIGURE1_NODE_COLOR for _ in nodes],
        edgecolors="none",
        linewidths=1.2,
        ax=ax,
    )
    nx.draw_networkx_labels(
        graph,
        positions,
        labels={node: figure1_node_display_label(node) for node in nodes},
        font_size=7,
        font_weight="normal",
        horizontalalignment="center",
        verticalalignment="center",
        font_color="#111827",
        ax=ax,
    )
    ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=8)
    if add_panel_label:
        ax.text(
            -0.08,
            1.04,
            spec.panel,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=22,
            fontweight="bold",
            color="black",
        )
    coords = list(positions.values())
    if coords:
        x_values = [float(xy[0]) for xy in coords]
        y_values = [float(xy[1]) for xy in coords]
        xpad = AXIS_PADDING * (max(x_values) - min(x_values) + 1e-9)
        ypad = AXIS_PADDING * (max(y_values) - min(y_values) + 1e-9)
        ax.set_xlim(min(x_values) - xpad, max(x_values) + xpad)
        ax.set_ylim(min(y_values) - ypad, max(y_values) + ypad)
    ax.set_aspect("equal")
    ax.axis("off")
    return {"nodes": nodes, "graph": graph, "edges": edges}


def draw_shared_legend(ax) -> None:
    """Draw the shared positive/negative edge legend."""
    ax.axis("off")
    ax.text(
        0.02,
        0.64,
        "Edges",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
        color=LEGEND_TEXT_COLOR,
    )
    handles = [
        Line2D([0], [0], color=POSITIVE_EDGE_COLOR, lw=3.0, label="Positive correlation"),
        Line2D([0], [0], color=NEGATIVE_EDGE_COLOR, lw=3.0, linestyle="dashed", label="Negative correlation"),
    ]
    legend = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.56),
        frameon=False,
        fontsize=10.5,
        borderaxespad=0,
        handlelength=2.2,
        labelspacing=1.1,
    )
    for text in legend.get_texts():
        text.set_color(LEGEND_TEXT_COLOR)


def save_single_panel(spec, output_dir: str | Path) -> Path:
    """Save one Figure1 panel as a PNG file."""
    output_path = Path(output_dir) / spec.output_png
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=PANEL_FIGSIZE)
    draw_network_panel(ax, spec, add_panel_label=False)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(output_path.with_suffix(".eps"), format="eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path
