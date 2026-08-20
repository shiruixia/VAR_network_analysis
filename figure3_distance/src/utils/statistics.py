from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .weighted_network_distance import weighted_network_distance_details


B = 1000
RANDOM_SEED = 42
DISTANCE_INPUT = "full_correlation_matrix"
DISTANCE_EDGE_FILTER = "none"
DISTANCE_WEIGHT_TRANSFORM = "absolute_spearman"
DISTANCE_NORMALIZATION = "separate_max_normalization"
METHOD_DESCRIPTION = (
    "Figure3 cumulative node-inclusion analysis using Table1-ranked indicators, "
    "complete Spearman correlation matrices, absolute correlation similarity weights, "
    "separate maximum-weight normalization, weighted network distance, and 1000 "
    "VAR-label permutations."
)

WD_SCALAR_KEYS = [
    "weighted_network_distance",
    "global_distance_component",
    "global_contribution",
    "local_heterogeneity_component",
    "local_contribution",
    "alpha_original_component",
    "alpha_complement_component",
    "alpha_contribution",
    "WNND_before",
    "WNND_after",
    "w_max_before",
    "w_max_after",
    "edge_count_before",
    "edge_count_after",
    "disconnected_pair_count_before",
    "disconnected_pair_count_after",
    "degenerate_network_before",
    "degenerate_network_after",
    "distance_input",
    "distance_edge_filter",
    "distance_weight_transform",
    "distance_normalization",
]


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle values excluding the diagonal."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def compute_matrix_similarity(before_matrix: np.ndarray | pd.DataFrame, after_matrix: np.ndarray | pd.DataFrame) -> float:
    """Compute upper-triangle correlation-vector similarity as a diagnostic value."""
    before_vector = upper_triangle(np.asarray(before_matrix, dtype=float))
    after_vector = upper_triangle(np.asarray(after_matrix, dtype=float))
    valid = np.isfinite(before_vector) & np.isfinite(after_vector)
    before_vector = before_vector[valid]
    after_vector = after_vector[valid]
    if len(before_vector) < 2:
        return 0.0
    if np.std(before_vector) == 0 or np.std(after_vector) == 0:
        return 0.0
    return float(np.corrcoef(before_vector, after_vector)[0, 1])


def spearman_correlation_matrix(values: np.ndarray, nodes: list[str] | None = None) -> pd.DataFrame:
    """Compute a complete Spearman correlation matrix for observed and permuted groups."""
    array = np.asarray(values, dtype=float)
    if array.ndim != 2:
        raise ValueError("Spearman input must be a 2D array.")
    n_nodes = int(array.shape[1])
    if n_nodes < 2:
        raise ValueError("Figure3 node-inclusion analysis requires at least two nodes.")
    labels = list(nodes) if nodes is not None else [str(index) for index in range(n_nodes)]
    if len(labels) != n_nodes:
        raise ValueError("Node label count must match the Spearman input width.")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        result = spearmanr(array, axis=0)
    statistic = np.asarray(result.statistic, dtype=float)
    if statistic.ndim == 0 and n_nodes == 2:
        value = float(statistic)
        matrix = np.array([[1.0, value], [value, 1.0]], dtype=float)
    elif statistic.shape == (n_nodes, n_nodes):
        matrix = statistic.astype(float, copy=True)
    else:
        matrix = np.eye(n_nodes, dtype=float)
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    pair = spearmanr(array[:, i], array[:, j])
                value = float(pair.statistic)
                matrix[i, j] = matrix[j, i] = value
    matrix[~np.isfinite(matrix)] = 0.0
    np.fill_diagonal(matrix, 1.0)
    return pd.DataFrame(matrix, index=labels, columns=labels)


def empirical_p_value(observed: float, null_values: np.ndarray) -> float:
    """Compute add-one empirical p-value for WD permutation tests."""
    null = np.asarray(null_values, dtype=float)
    return float((1 + np.sum(null >= observed)) / (len(null) + 1))


def _var_column(data: pd.DataFrame, var_column: str = "VAR") -> str:
    if var_column in data.columns:
        return var_column
    if var_column == "VAR" and "var" in data.columns:
        return "var"
    raise ValueError(f"VAR column not found: {var_column}")


def prepare_step_data(data: pd.DataFrame, nodes: list[str], var_column: str = "VAR") -> pd.DataFrame:
    """Return numeric complete-case VAR + node data for one Figure3 node set."""
    var_col = _var_column(data, var_column)
    required = [var_col, *nodes]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Match-level data missing required column(s): {missing}")
    subset = data[required].copy()
    for column in required:
        subset[column] = pd.to_numeric(subset[column], errors="coerce")
    subset = subset.dropna(subset=required).reset_index(drop=True)
    subset[var_col] = subset[var_col].astype(int)
    if not set(subset[var_col].unique()).issubset({0, 1}):
        raise ValueError("VAR values must be 0/1.")
    if var_col != "VAR":
        subset = subset.rename(columns={var_col: "VAR"})
    return subset


def _as_labeled_matrix(matrix: np.ndarray | pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    values = np.asarray(matrix, dtype=float)
    if values.shape != (len(nodes), len(nodes)):
        raise ValueError("Correlation matrix shape does not match the node set.")
    return pd.DataFrame(values, index=nodes, columns=nodes)


def compute_weighted_network_distance(
    before_matrix: np.ndarray | pd.DataFrame,
    after_matrix: np.ndarray | pd.DataFrame,
    nodes: list[str],
) -> dict[str, object]:
    """Compute Figure3 WD using full correlation matrices and no edge filtering."""
    before = _as_labeled_matrix(before_matrix, nodes)
    after = _as_labeled_matrix(after_matrix, nodes)
    return weighted_network_distance_details(before, None, after, None, nodes)


def _finite_upper(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.to_numpy(dtype=float)
    upper = values[np.triu_indices_from(values, k=1)]
    return upper[np.isfinite(upper)]


def _binary_path_summary(details: dict[str, object], period: str) -> dict[str, object]:
    binary = details[f"{period}_binary_shortest"]
    finite = _finite_upper(binary)
    if len(finite) == 0:
        return {
            f"binary_mean_shortest_{period}": np.nan,
            f"binary_diameter_{period}": np.nan,
        }
    return {
        f"binary_mean_shortest_{period}": float(np.mean(finite)),
        f"binary_diameter_{period}": float(np.max(finite)),
    }


def _curve_scalars(details: dict[str, object], number_of_nodes: int) -> dict[str, object]:
    possible_edges = int(number_of_nodes * (number_of_nodes - 1) / 2)
    edge_before = int(details["edge_count_before"])
    edge_after = int(details["edge_count_after"])
    disconnected_before = int(details["disconnected_pair_count_before"])
    disconnected_after = int(details["disconnected_pair_count_after"])
    scalars = {key: details[key] for key in WD_SCALAR_KEYS if key in details}
    scalars.update(
        {
            "possible_edges": possible_edges,
            "density_before": float(edge_before / possible_edges) if possible_edges else np.nan,
            "density_after": float(edge_after / possible_edges) if possible_edges else np.nan,
            "complete_graph_before": bool(edge_before == possible_edges and disconnected_before == 0),
            "complete_graph_after": bool(edge_after == possible_edges and disconnected_after == 0),
        }
    )
    scalars.update(_binary_path_summary(details, "before"))
    scalars.update(_binary_path_summary(details, "after"))
    return scalars


def _observed_details(values: np.ndarray, labels: np.ndarray, nodes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    before = values[labels == 0]
    after = values[labels == 1]
    if len(before) == 0 or len(after) == 0:
        raise ValueError("Both VAR groups must contain observations.")
    before_matrix = spearman_correlation_matrix(before, nodes)
    after_matrix = spearman_correlation_matrix(after, nodes)
    return before_matrix, after_matrix, compute_weighted_network_distance(before_matrix, after_matrix, nodes)


def run_permutation_test(
    data: pd.DataFrame,
    nodes: list[str],
    var_column: str = "VAR",
    n_permutations: int = B,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run a Figure3 WD permutation test for one cumulative node set."""
    subset = prepare_step_data(data, nodes, var_column)
    values = subset[nodes].to_numpy(dtype=float)
    labels = subset["VAR"].to_numpy(dtype=int)
    group0_size = int(np.sum(labels == 0))
    group1_size = int(np.sum(labels == 1))
    if len(labels) != len(values):
        raise AssertionError("Label and value lengths differ after complete-case cleaning.")
    if group0_size + group1_size != len(values):
        raise AssertionError("VAR group sizes do not sum to the cleaned sample size.")

    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    for permutation_index in range(1, n_permutations + 1):
        permuted_labels = rng.permutation(labels)
        permutation_group0 = int(np.sum(permuted_labels == 0))
        permutation_group1 = int(np.sum(permuted_labels == 1))
        if permutation_group0 != group0_size or permutation_group1 != group1_size:
            raise AssertionError("A permutation changed the fixed VAR group sizes.")
        _before_matrix, _after_matrix, details = _observed_details(values, permuted_labels, nodes)
        records.append(
            {
                "permutation_index": permutation_index,
                "weighted_network_distance_perm": float(details["weighted_network_distance"]),
                "global_component_perm": float(details["global_distance_component"]),
                "local_component_perm": float(details["local_heterogeneity_component"]),
                "alpha_original_component_perm": float(details["alpha_original_component"]),
                "alpha_complement_component_perm": float(details["alpha_complement_component"]),
                "permutation_group0_n": permutation_group0,
                "permutation_group1_n": permutation_group1,
                "edge_count_group0": int(details["edge_count_before"]),
                "edge_count_group1": int(details["edge_count_after"]),
                "disconnected_pair_count_group0": int(details["disconnected_pair_count_before"]),
                "disconnected_pair_count_group1": int(details["disconnected_pair_count_after"]),
                "degenerate_network_group0": bool(details["degenerate_network_before"]),
                "degenerate_network_group1": bool(details["degenerate_network_after"]),
                "random_seed": int(random_seed),
            }
        )
    return pd.DataFrame(records)


def compute_node_inclusion_results(
    match_data: pd.DataFrame,
    path_definitions: pd.DataFrame,
    dataset: str,
    var_column: str = "VAR",
    n_permutations: int = B,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute Figure3 cumulative node-inclusion WD and permutation summaries."""
    curve_rows: list[dict[str, object]] = []
    permutation_rows: list[dict[str, object]] = []
    distribution_rows: list[pd.DataFrame] = []
    for row in path_definitions.sort_values("n_nodes").itertuples(index=False):
        nodes = str(row.node_set_full).split(" + ")
        number_of_nodes = int(row.n_nodes)
        subset = prepare_step_data(match_data, nodes, var_column)
        values = subset[nodes].to_numpy(dtype=float)
        labels = subset["VAR"].to_numpy(dtype=int)
        group0_size = int(np.sum(labels == 0))
        group1_size = int(np.sum(labels == 1))
        if len(labels) != len(values) or group0_size + group1_size != len(values):
            raise AssertionError("Cleaned labels and values are inconsistent.")

        before_matrix, after_matrix, details = _observed_details(values, labels, nodes)
        seed = int(random_seed + int(row.step))
        distribution = run_permutation_test(
            subset,
            nodes,
            var_column="VAR",
            n_permutations=n_permutations,
            random_seed=seed,
        )
        observed_distance = float(details["weighted_network_distance"])
        null_values = distribution["weighted_network_distance_perm"].to_numpy(dtype=float)
        empirical_p = empirical_p_value(observed_distance, null_values)
        null_std = float(np.std(null_values, ddof=1)) if len(null_values) > 1 else 0.0
        curve_scalars = _curve_scalars(details, number_of_nodes)
        curve_rows.append(
            {
                "dataset": dataset,
                "step": int(row.step),
                "number_of_nodes": number_of_nodes,
                "added_node": str(row.node_added),
                "node_set_full": str(row.node_set_full),
                **curve_scalars,
                "matrix_similarity": compute_matrix_similarity(before_matrix, after_matrix),
                "n_matches": int(len(subset)),
                "n_no_var": group0_size,
                "n_with_var": group1_size,
                "random_seed": seed,
            }
        )
        permutation_rows.append(
            {
                "dataset": dataset,
                "step": int(row.step),
                "number_of_nodes": number_of_nodes,
                "added_node": str(row.node_added),
                "node_set_full": str(row.node_set_full),
                "observed_weighted_network_distance": observed_distance,
                "null_mean": float(np.mean(null_values)),
                "null_std": null_std,
                "null_95_percentile": float(np.percentile(null_values, 95)),
                "empirical_p_value": empirical_p,
                "significant_at_0_05": bool(empirical_p < 0.05),
                "n_permutations": int(n_permutations),
                "random_seed": seed,
                "n_matches": int(len(subset)),
                "n_no_var": group0_size,
                "n_with_var": group1_size,
            }
        )
        distribution = distribution.copy()
        distribution.insert(0, "number_of_nodes", number_of_nodes)
        distribution.insert(0, "step", int(row.step))
        distribution.insert(0, "dataset", dataset)
        distribution_rows.append(distribution)
    curve = pd.DataFrame(curve_rows)
    permutation_summary = pd.DataFrame(permutation_rows)
    permutation_distribution = pd.concat(distribution_rows, ignore_index=True) if distribution_rows else pd.DataFrame()
    return curve, permutation_summary, permutation_distribution


def build_statistics_summary(
    dataset: str,
    match_data: pd.DataFrame,
    node_order: pd.DataFrame,
    method_description: str = METHOD_DESCRIPTION,
    var_column: str = "VAR",
) -> pd.DataFrame:
    """Build a dataset-level Figure3 statistics summary."""
    var_col = _var_column(match_data, var_column)
    var_values = pd.to_numeric(match_data[var_col], errors="coerce")
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "n_matches": int(var_values.isin([0, 1]).sum()),
                "n_no_var": int(var_values.eq(0).sum()),
                "n_with_var": int(var_values.eq(1).sum()),
                "node_number": int(len(node_order)),
                "method_description": method_description,
                "distance_input": DISTANCE_INPUT,
                "distance_edge_filter": DISTANCE_EDGE_FILTER,
                "distance_weight_transform": DISTANCE_WEIGHT_TRANSFORM,
                "distance_normalization": DISTANCE_NORMALIZATION,
                "interpretation": "WD trajectory under cumulative inclusion of ranked indicators",
            }
        ]
    )


def build_parameter_table(n_permutations: int = B, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Return reproducibility parameters for Figure3 WD analysis."""
    return pd.DataFrame(
        [
            {"parameter": "n_permutations_per_step", "value": int(n_permutations)},
            {"parameter": "base_random_seed", "value": int(random_seed)},
            {"parameter": "step_seed_rule", "value": "base_random_seed + step"},
            {"parameter": "distance_input", "value": DISTANCE_INPUT},
            {"parameter": "distance_edge_filter", "value": DISTANCE_EDGE_FILTER},
            {"parameter": "distance_weight_transform", "value": DISTANCE_WEIGHT_TRANSFORM},
            {"parameter": "distance_normalization", "value": DISTANCE_NORMALIZATION},
            {"parameter": "wd_scale_note", "value": "WD is not divided by node count or possible pair count."},
            {
                "parameter": "n_equals_2_note",
                "value": "With separate-max normalization, two nonzero single-edge networks both normalize to edge weight 1, so WD is 0.",
            },
        ]
    )
