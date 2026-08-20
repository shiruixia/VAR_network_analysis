from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from utils.excel_utils import (
    write_network_comparison_result,
    write_network_metrics_result,
    write_permutation_result,
    write_random_network_result,
    write_weighted_network_distance_details,
)
from utils.figure2_data_loader import LoadedNetworkResult, load_network_result
from utils.figure2_paths import panels_dir
from utils.match_data_loader import load_match_level_data
from utils.network_comparison import observed_weighted_network_distance, permutation_analysis_from_match_level_data
from utils.network_plotting import draw_pair_network, draw_single_network, graph_from_edge_list, positions_for_nodes
from utils.paper_color_config import REAL_NETWORK_COLORS
from utils.plot_permutation_distribution import plot_permutation_weighted_distance
from utils.random_network import random_edges_from_match_level_data
from utils.weighted_network_distance import build_signed_edge_list


DATASETS = ("ligue1", "euro", "worldcup")
DISPLAY_TITLES = {
    "ligue1": "Ligue 1",
    "euro": "UEFA European Championship",
    "worldcup": "FIFA World Cup",
}
RANDOM_NETWORK_SEED = 123


def _density(graph) -> float:
    n_nodes = graph.number_of_nodes()
    possible_edges = n_nodes * (n_nodes - 1) / 2
    return float(graph.number_of_edges() / possible_edges) if possible_edges else 0.0


def _edge_set(edge_list: pd.DataFrame) -> set[tuple[str, str]]:
    if edge_list.empty:
        return set()
    return {
        tuple(sorted((str(row.source), str(row.target))))
        for row in edge_list.itertuples(index=False)
    }


def adjacency_mismatch_frame(data: LoadedNetworkResult, details: dict[str, object]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    before_plot_edges = build_signed_edge_list(data.before_correlation_matrix, data.before_fdr_matrix, data.nodes)
    after_plot_edges = build_signed_edge_list(data.after_correlation_matrix, data.after_fdr_matrix, data.nodes)
    before_plot_adjacency = pd.DataFrame(0, index=data.nodes, columns=data.nodes, dtype=int)
    after_plot_adjacency = pd.DataFrame(0, index=data.nodes, columns=data.nodes, dtype=int)
    for row in before_plot_edges.itertuples(index=False):
        before_plot_adjacency.loc[row.source, row.target] = before_plot_adjacency.loc[row.target, row.source] = 1
    for row in after_plot_edges.itertuples(index=False):
        after_plot_adjacency.loc[row.source, row.target] = after_plot_adjacency.loc[row.target, row.source] = 1
    for period, loaded_adjacency, recomputed in (
        ("before_var", data.before_adjacency_matrix, before_plot_adjacency),
        ("after_var", data.after_adjacency_matrix, after_plot_adjacency),
    ):
        loaded = loaded_adjacency.loc[data.nodes, data.nodes].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        wd_adjacency = recomputed.loc[data.nodes, data.nodes].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int)
        for i, source in enumerate(data.nodes):
            for j in range(i + 1, len(data.nodes)):
                target = data.nodes[j]
                if int(loaded.loc[source, target]) != int(wd_adjacency.loc[source, target]):
                    rows.append(
                        {
                            "dataset": data.dataset_key,
                            "period": period,
                            "source": source,
                            "target": target,
                            "figure1_adjacency": int(loaded.loc[source, target]),
                            "figure2_plot_adjacency": int(wd_adjacency.loc[source, target]),
                            "audit_scope": "plot_edge_filter_only",
                        }
                    )
    return pd.DataFrame(rows)


def observed_metrics_frame(data: LoadedNetworkResult, observed: dict[str, object]) -> pd.DataFrame:
    before_edges = {
        tuple(sorted((str(row.source), str(row.target))))
        for row in build_signed_edge_list(data.before_correlation_matrix, data.before_fdr_matrix, data.nodes).itertuples(index=False)
    }
    after_edges = {
        tuple(sorted((str(row.source), str(row.target))))
        for row in build_signed_edge_list(data.after_correlation_matrix, data.after_fdr_matrix, data.nodes).itertuples(index=False)
    }
    shared = before_edges & after_edges
    added = after_edges - before_edges
    removed = before_edges - after_edges
    return pd.DataFrame(
        [
            {
                "dataset": data.dataset_key,
                "metric": "weighted_network_distance",
                "distance_input": "full_correlation_matrix",
                "distance_edge_filter": "none",
                "distance_weight_transform": "absolute_spearman",
                "distance_normalization": "separate_max_normalization",
                "plot_edge_filter": "abs_r>=0.10_and_fdr_q<0.05",
                "weighted_network_distance": float(observed["weighted_network_distance"]),
                "global_distance_component": float(observed["global_distance_component"]),
                "global_contribution": float(observed["global_contribution"]),
                "local_heterogeneity_component": float(observed["local_heterogeneity_component"]),
                "local_contribution": float(observed["local_contribution"]),
                "alpha_original_component": float(observed["alpha_original_component"]),
                "alpha_complement_component": float(observed["alpha_complement_component"]),
                "alpha_contribution": float(observed["alpha_contribution"]),
                "WNND_before": float(observed["WNND_before"]),
                "WNND_after": float(observed["WNND_after"]),
                "w_max_before": float(observed["w_max_before"]),
                "w_max_after": float(observed["w_max_after"]),
                "before_edge_number": len(before_edges),
                "after_edge_number": len(after_edges),
                "shared_edge_number": len(shared),
                "added_edge_number": len(added),
                "removed_edge_number": len(removed),
                "disconnected_pair_count_before": int(observed["disconnected_pair_count_before"]),
                "disconnected_pair_count_after": int(observed["disconnected_pair_count_after"]),
                "degenerate_network_before": bool(observed["degenerate_network_before"]),
                "degenerate_network_after": bool(observed["degenerate_network_after"]),
                "frobenius_distance_diagnostic": float(observed["frobenius_distance_diagnostic"]),
                "matrix_similarity": float(observed["matrix_correlation_similarity"]),
                "mean_absolute_delta_r": float(observed["mean_absolute_delta_r"]),
                "sum_abs_delta_r": float(observed["sum_abs_delta_r"]),
                "max_absolute_delta_r": float(observed["max_absolute_delta_r"]),
                "edge_difference_summary": (
                    f"before_edges={len(before_edges)}; after_edges={len(after_edges)}; "
                    f"shared_edges={len(shared)}; added_edges={len(added)}; removed_edges={len(removed)}"
                ),
            }
        ]
    )


def run_real_network(dataset: str) -> list[Path]:
    data = load_network_result(dataset)
    positions = positions_for_nodes(data.nodes, data.dataset_key)
    before_edges = build_signed_edge_list(data.before_correlation_matrix, data.before_fdr_matrix, data.nodes)
    after_edges = build_signed_edge_list(data.after_correlation_matrix, data.after_fdr_matrix, data.nodes)
    before_graph = graph_from_edge_list(data.nodes, before_edges)
    after_graph = graph_from_edge_list(data.nodes, after_edges)

    output_paths = [
        draw_single_network(before_graph, positions, panels_dir() / f"{dataset}_before_real_network.png", REAL_NETWORK_COLORS),
        draw_single_network(after_graph, positions, panels_dir() / f"{dataset}_after_real_network.png", REAL_NETWORK_COLORS),
    ]
    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "period": "before_var",
                "n_nodes": before_graph.number_of_nodes(),
                "n_edges": before_graph.number_of_edges(),
                "network_density": _density(before_graph),
                "source": "Figure1 correlation_matrix.xlsx + fdr_matrix.xlsx, re-filtered in Figure2",
            },
            {
                "dataset": dataset,
                "period": "after_var",
                "n_nodes": after_graph.number_of_nodes(),
                "n_edges": after_graph.number_of_edges(),
                "network_density": _density(after_graph),
                "source": "Figure1 correlation_matrix.xlsx + fdr_matrix.xlsx, re-filtered in Figure2",
            },
        ]
    )
    output_paths.append(
        write_network_metrics_result(
            dataset,
            before_var=data.before_network_metrics,
            after_var=data.after_network_metrics,
            summary=summary,
        )
    )
    return output_paths


def run_random_network(dataset: str) -> list[Path]:
    match_data = load_match_level_data(dataset)
    data = load_network_result(dataset)
    random_edges, _, _ = random_edges_from_match_level_data(match_data, data.nodes, dataset, RANDOM_NETWORK_SEED)

    group_a_edges = random_edges.loc[random_edges["group"].astype(str).eq("random_groupA")].copy()
    group_b_edges = random_edges.loc[random_edges["group"].astype(str).eq("random_groupB")].copy()
    positions = positions_for_nodes(data.nodes, dataset)
    graph_a = graph_from_edge_list(data.nodes, group_a_edges)
    graph_b = graph_from_edge_list(data.nodes, group_b_edges)
    panel_path = draw_pair_network(graph_a, graph_b, positions, panels_dir() / f"{dataset}_random_network_pair.png")

    summary = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "seed": RANDOM_NETWORK_SEED,
                "group": "random_groupA",
                "node_number": graph_a.number_of_nodes(),
                "edge_number": graph_a.number_of_edges(),
                "network_density": _density(graph_a),
                "source": "paper_assets/data",
            },
            {
                "dataset": dataset,
                "seed": RANDOM_NETWORK_SEED,
                "group": "random_groupB",
                "node_number": graph_b.number_of_nodes(),
                "edge_number": graph_b.number_of_edges(),
                "network_density": _density(graph_b),
                "source": "paper_assets/data",
            },
        ]
    )
    workbook = write_random_network_result(dataset, summary=summary, edges=random_edges)
    return [panel_path, workbook]


def run_permutation(dataset: str, n_permutations: int = 1000, random_seed: int = 42) -> list[Path]:
    data = load_network_result(dataset)
    match_data = load_match_level_data(dataset)
    observed = observed_weighted_network_distance(data)
    observed, distribution, summary = permutation_analysis_from_match_level_data(
        match_data,
        data.nodes,
        observed,
        dataset=dataset,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )

    summary_row = summary.iloc[0]
    weighted_panel_path = panels_dir() / f"{dataset}_permutation_weighted_distance.png"
    panel_path = plot_permutation_weighted_distance(
        distribution,
        observed=float(summary_row["observed_value"]),
        empirical_p=float(summary_row["empirical_p_value"]),
        path=weighted_panel_path,
        title=f"{DISPLAY_TITLES[dataset]} permutation distribution",
    )
    legacy_name_panel_path = panels_dir() / f"{dataset}_permutation_distribution.png"
    if legacy_name_panel_path != panel_path:
        legacy_name_panel_path.write_bytes(panel_path.read_bytes())
        legacy_name_panel_path.with_suffix(".eps").write_bytes(panel_path.with_suffix(".eps").read_bytes())

    mismatches = adjacency_mismatch_frame(data, observed)
    permutation_workbook = write_permutation_result(dataset, summary, distribution)
    comparison_workbook = write_network_comparison_result(dataset, observed_metrics_frame(data, observed), mismatches)
    details_workbook = write_weighted_network_distance_details(dataset, observed)
    return [panel_path, permutation_workbook, comparison_workbook, details_workbook]


def run_full_dataset(dataset: str) -> list[Path]:
    return [*run_real_network(dataset), *run_random_network(dataset), *run_permutation(dataset)]


def run_all_datasets() -> list[Path]:
    outputs: list[Path] = []
    for dataset in DATASETS:
        outputs.extend(run_full_dataset(dataset))
    return outputs
