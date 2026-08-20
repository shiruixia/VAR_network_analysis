"""Common Figure1 network analysis helpers."""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd


def strict_adjacency_matrix(edge_list: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    """Build a binary adjacency matrix from edges retained by keep_strict."""
    adjacency = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    if edge_list.empty:
        return adjacency
    selected = edge_list.loc[edge_list["keep_strict"].astype(bool)]
    for row in selected.itertuples(index=False):
        source = str(row.source)
        target = str(row.target)
        if source in adjacency.index and target in adjacency.columns:
            adjacency.at[source, target] = 1
            adjacency.at[target, source] = 1
    return adjacency


def build_graph(nodes: list[str], edge_list: pd.DataFrame, keep_column: str = "keep_strict") -> nx.Graph:
    """Build a NetworkX graph from an edge list and a retain-column."""
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    if edge_list.empty:
        return graph
    for row in edge_list.loc[edge_list[keep_column].astype(bool)].itertuples(index=False):
        abs_r = float(row.abs_r)
        spearman_r = float(row.spearman_r)
        graph.add_edge(
            str(row.source),
            str(row.target),
            spearman_r=spearman_r,
            abs_r=abs_r,
            weight=abs_r,
            signed_weight=spearman_r,
            distance=1.0 / max(abs_r, 1e-12),
            sign=getattr(row, "sign", "positive" if spearman_r >= 0 else "negative"),
        )
    return graph


def compute_node_metrics(graph: nx.Graph, nodes: list[str]) -> pd.DataFrame:
    """Compute node-level network metrics using the existing Figure1 method."""
    betweenness = nx.betweenness_centrality(graph, weight="distance", normalized=True)
    closeness = _componentwise_closeness(graph)
    eigenvector = _componentwise_eigenvector(graph)
    rows = []
    for node in nodes:
        edges = list(graph.edges(node, data=True))
        signed = sum(edge[2]["spearman_r"] for edge in edges)
        absolute = sum(edge[2]["abs_r"] for edge in edges)
        rows.append(
            {
                "node": node,
                "degree": graph.degree(node),
                "weighted_degree_abs": absolute,
                "positive_degree": sum(edge[2]["spearman_r"] > 0 for edge in edges),
                "negative_degree": sum(edge[2]["spearman_r"] < 0 for edge in edges),
                "strength_signed": signed,
                "strength_abs": absolute,
                "betweenness_centrality": betweenness[node],
                "closeness_centrality": closeness[node],
                "eigenvector_centrality": eigenvector[node],
                "clustering_abs": float(nx.clustering(graph, node, weight="weight")),
            }
        )
    return pd.DataFrame(rows).sort_values(["strength_abs", "degree"], ascending=[False, False]).reset_index(drop=True)


def compute_network_metrics(
    var_value: int,
    strict_graph: nx.Graph,
    relaxed_graph: nx.Graph,
    n_matches: int | None = None,
) -> dict[str, float | int | str | None]:
    """Compute graph-level metrics using the existing Figure1 method."""
    components = list(nx.connected_components(strict_graph))
    edge_r = [data["spearman_r"] for _, _, data in strict_graph.edges(data=True)]
    n_nodes = strict_graph.number_of_nodes()
    return {
        "var": var_value,
        "group_label": "Before VAR" if var_value == 0 else "After VAR",
        "n_matches": n_matches,
        "n_nodes": n_nodes,
        "n_edges_strict": strict_graph.number_of_edges(),
        "n_edges_relaxed": relaxed_graph.number_of_edges(),
        "density_strict": nx.density(strict_graph),
        "average_degree_strict": sum(dict(strict_graph.degree()).values()) / n_nodes if n_nodes else 0.0,
        "average_clustering_strict": nx.average_clustering(strict_graph, weight="weight") if n_nodes else 0.0,
        "number_of_components_strict": len(components),
        "largest_component_size": max(map(len, components)) if components else 0,
        "mean_abs_edge_weight_strict": float(np.mean(np.abs(edge_r))) if edge_r else 0.0,
        "mean_signed_edge_weight_strict": float(np.mean(edge_r)) if edge_r else 0.0,
        "positive_edges": sum(value > 0 for value in edge_r),
        "negative_edges": sum(value < 0 for value in edge_r),
    }


def _componentwise_closeness(graph: nx.Graph) -> dict[str, float]:
    values = {node: 0.0 for node in graph.nodes}
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes)
        values.update(nx.closeness_centrality(component, distance="distance"))
    return values


def _componentwise_eigenvector(graph: nx.Graph) -> dict[str, float]:
    values = {node: 0.0 for node in graph.nodes}
    for component_nodes in nx.connected_components(graph):
        component = graph.subgraph(component_nodes)
        if component.number_of_nodes() == 1:
            continue
        try:
            component_values = nx.eigenvector_centrality_numpy(component, weight="abs_r")
        except Exception:
            component_values = nx.eigenvector_centrality(component, weight="abs_r", max_iter=5000)
        values.update(component_values)
    return values
