from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse.csgraph import shortest_path


ABS_R_THRESHOLD = 0.10
FDR_THRESHOLD = 0.05
EPS = 1e-12


@dataclass(frozen=True)
class ShortestPathResult:
    weighted: pd.DataFrame
    binary: pd.DataFrame


@dataclass(frozen=True)
class NetworkProfile:
    absolute_weight: pd.DataFrame
    normalized_weight: pd.DataFrame
    adjacency: pd.DataFrame
    edge_length: pd.DataFrame
    weighted_shortest: pd.DataFrame
    binary_shortest: pd.DataFrame
    scaled_distance: pd.DataFrame
    discrete_distance: pd.DataFrame
    distance_probability: pd.DataFrame
    mu: pd.DataFrame
    alpha: pd.DataFrame
    alpha_complement: pd.DataFrame
    wnnd: float
    w_max: float
    edge_count: int
    disconnected_pair_count: int
    degenerate_network: bool
    finite_distance_categories: list[str]


def _matrix(frame: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    matrix = frame.loc[nodes, nodes].apply(pd.to_numeric, errors="coerce").astype(float).copy()
    matrix.index = nodes
    matrix.columns = nodes
    return matrix


def _zero_diagonal(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for i in range(min(output.shape)):
        output.iat[i, i] = 0.0
    return output


def build_absolute_weight_matrix(
    correlation: pd.DataFrame,
    nodes: list[str],
) -> pd.DataFrame:
    """Return full WD input W0 = abs(R), preserving all finite nonzero correlations."""
    corr = _matrix(correlation, nodes)
    values = np.abs(corr.to_numpy(dtype=float))
    values[~np.isfinite(values)] = 0.0
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=nodes, columns=nodes)


def build_signed_edge_list(
    correlation: pd.DataFrame,
    fdr: pd.DataFrame,
    nodes: list[str],
    abs_r_threshold: float = ABS_R_THRESHOLD,
    fdr_threshold: float = FDR_THRESHOLD,
) -> pd.DataFrame:
    """Return strict edges for drawing, preserving the sign of Spearman r."""
    corr = _matrix(correlation, nodes)
    q = _matrix(fdr, nodes)
    rows: list[dict[str, object]] = []
    for i, source in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            r_value = corr.loc[source, target]
            q_value = q.loc[source, target]
            keep = (
                np.isfinite(r_value)
                and np.isfinite(q_value)
                and abs(float(r_value)) >= abs_r_threshold
                and float(q_value) < fdr_threshold
            )
            if keep:
                r_float = float(r_value)
                rows.append(
                    {
                        "source": source,
                        "target": target,
                        "correlation": r_float,
                        "weight": r_float,
                        "spearman_r": r_float,
                        "abs_r": abs(r_float),
                        "fdr_q_value": float(q_value),
                        "edge_sign": "positive" if r_float >= 0 else "negative",
                        "sign": "positive" if r_float >= 0 else "negative",
                        "edge_weight": abs(r_float),
                        "keep_strict": True,
                    }
                )
    return pd.DataFrame(rows)


def build_adjacency_matrix(weight: pd.DataFrame) -> pd.DataFrame:
    adjacency = (weight.astype(float) > 0).astype(int)
    for i in range(min(adjacency.shape)):
        adjacency.iat[i, i] = 0
    return adjacency


def normalize_weight_matrix(weight: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Normalize nonzero W0 values by each network's own maximum retained weight."""
    w = weight.astype(float).copy()
    upper = w.to_numpy(dtype=float)[np.triu_indices_from(w, k=1)]
    positives = upper[upper > 0]
    if len(positives) == 0:
        return _zero_diagonal(w * 0.0), 0.0
    w_max = float(np.max(positives))
    normalized = w / w_max
    normalized = normalized.clip(lower=0.0, upper=1.0)
    return _zero_diagonal(normalized), w_max


def build_edge_length_matrix(normalized_weight: pd.DataFrame) -> pd.DataFrame:
    """Convert normalized similarity weights to lengths C_ij = 1 / W_ij."""
    w = normalized_weight.astype(float)
    values = np.full(w.shape, np.inf, dtype=float)
    positive = w.to_numpy(dtype=float) > 0
    values[positive] = 1.0 / w.to_numpy(dtype=float)[positive]
    np.fill_diagonal(values, 0.0)
    return pd.DataFrame(values, index=w.index, columns=w.columns)


def all_pairs_shortest_paths(edge_length: pd.DataFrame) -> ShortestPathResult:
    """Calculate weighted and binary all-pairs shortest path matrices."""
    c = edge_length.astype(float)
    edge_values = np.ascontiguousarray(c.to_numpy(dtype=float))
    weighted_values = shortest_path(edge_values, directed=False, unweighted=False)
    binary_cost = np.where(np.isfinite(edge_values) & ~np.eye(len(c), dtype=bool), 1.0, np.inf)
    np.fill_diagonal(binary_cost, 0.0)
    binary_values = shortest_path(np.ascontiguousarray(binary_cost), directed=False, unweighted=False)
    return ShortestPathResult(
        weighted=pd.DataFrame(weighted_values, index=c.index, columns=c.columns),
        binary=pd.DataFrame(binary_values, index=c.index, columns=c.columns),
    )


def _finite_upper_values(matrix: pd.DataFrame) -> np.ndarray:
    arr = matrix.to_numpy(dtype=float)
    upper = arr[np.triu_indices_from(arr, k=1)]
    return upper[np.isfinite(upper)]


def _disconnected_pair_count(matrix: pd.DataFrame) -> int:
    arr = matrix.to_numpy(dtype=float)
    upper = arr[np.triu_indices_from(arr, k=1)]
    return int(np.sum(~np.isfinite(upper)))


def _scale_and_discretize(shortest: ShortestPathResult) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    weighted_finite = _finite_upper_values(shortest.weighted)
    binary_finite = _finite_upper_values(shortest.binary)
    if len(weighted_finite) == 0 or len(binary_finite) == 0:
        scaled = shortest.weighted.copy()
        discrete = shortest.weighted.copy()
        return scaled, discrete, True

    mean_weighted = float(np.mean(weighted_finite))
    mean_binary = float(np.mean(binary_finite))
    if mean_weighted <= 0 or not np.isfinite(mean_weighted):
        scaled = shortest.weighted.copy()
        discrete = shortest.weighted.copy()
        return scaled, discrete, True

    scaled_values = shortest.weighted.to_numpy(dtype=float) * mean_binary / mean_weighted
    finite = np.isfinite(scaled_values)
    discrete_values = np.full(scaled_values.shape, np.inf, dtype=float)
    discrete_values[finite] = np.ceil(scaled_values[finite] - EPS)
    np.fill_diagonal(scaled_values, 0.0)
    np.fill_diagonal(discrete_values, 0.0)
    return (
        pd.DataFrame(scaled_values, index=shortest.weighted.index, columns=shortest.weighted.columns),
        pd.DataFrame(discrete_values, index=shortest.weighted.index, columns=shortest.weighted.columns),
        False,
    )


def distance_probability_profile(
    weighted_shortest: pd.DataFrame,
    discrete_distance: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return node-level distance probabilities and their network mean."""
    nodes = list(discrete_distance.index)
    finite_values = discrete_distance.to_numpy(dtype=float)
    finite_nonzero = finite_values[np.isfinite(finite_values) & (finite_values > 0)]
    max_distance = int(np.max(finite_nonzero)) if len(finite_nonzero) else 0
    categories = [str(value) for value in range(1, max_distance + 1)]
    has_disconnected = _disconnected_pair_count(weighted_shortest) > 0
    if has_disconnected:
        categories.append("disconnected")

    records: list[list[float]] = []
    denominator = max(len(nodes) - 1, 1)
    weighted_values = weighted_shortest.to_numpy(dtype=float)
    for i, _node in enumerate(nodes):
        row: list[float] = []
        for category in categories:
            if category == "disconnected":
                count = int(np.sum(~np.isfinite(weighted_values[i, :]))) 
            else:
                count = int(np.sum(finite_values[i, :] == int(category)))
            row.append(count / denominator)
        if not categories:
            row = [1.0]
            categories = ["disconnected"]
        records.append(row)

    probability = pd.DataFrame(records, index=nodes, columns=categories)
    mu = pd.DataFrame([probability.mean(axis=0).to_numpy(dtype=float)], columns=probability.columns, index=["mu"])
    return probability, mu, categories


def _multi_distribution_js(probability: pd.DataFrame) -> float:
    p = probability.to_numpy(dtype=float)
    if p.size == 0:
        return 0.0
    mu = np.mean(p, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where((p > 0) & (mu > 0), p * np.log(p / mu), 0.0)
    value = float(np.mean(np.sum(terms, axis=1)))
    return 0.0 if abs(value) < 1e-15 else max(value, 0.0)


def _wnnd(probability: pd.DataFrame, finite_categories: list[str], degenerate: bool) -> float:
    if degenerate:
        return 0.0
    m = len([category for category in finite_categories if category != "disconnected"])
    if m <= 0:
        return 0.0
    return float(_multi_distribution_js(probability) / np.log(m + 1))


def weighted_alpha_centrality_distribution(normalized_weight: pd.DataFrame) -> pd.DataFrame:
    """Return sorted weighted alpha-centrality probability distribution."""
    w = normalized_weight.to_numpy(dtype=float)
    n = w.shape[0]
    upper = w[np.triu_indices_from(w, k=1)]
    positives = upper[upper > 0]
    labels = [f"alpha_{i}" for i in range(1, n + 1)] + ["residual"]
    if len(positives) == 0:
        return pd.DataFrame({"probability": [0.0] * n + [1.0]}, index=labels)

    mean_omega = float(np.mean(positives))
    strength = np.sum(w, axis=1)
    beta = strength / ((n - 1) * mean_omega)
    alpha = 1.0 / n
    x = np.linalg.solve(np.eye(n) - alpha * w, beta)
    x_sorted = np.sort(x)
    probabilities = np.r_[x_sorted / (n**2), 1.0 - float(np.sum(x_sorted) / (n**2))]
    if np.min(probabilities) < -1e-10:
        raise ValueError(f"Alpha-centrality distribution has materially negative probability: {probabilities}")
    probabilities = np.clip(probabilities, 0.0, 1.0)
    total = float(np.sum(probabilities))
    if not np.isfinite(total) or abs(total - 1.0) > 1e-8:
        raise ValueError(f"Alpha-centrality probabilities do not sum to 1: {total}")
    probabilities[-1] += 1.0 - float(np.sum(probabilities))
    return pd.DataFrame({"probability": probabilities}, index=labels)


def jensen_shannon_divergence(p: np.ndarray | pd.Series | pd.DataFrame, q: np.ndarray | pd.Series | pd.DataFrame) -> float:
    """Return two-distribution JS divergence using natural logs."""
    p_values = np.asarray(p, dtype=float).ravel()
    q_values = np.asarray(q, dtype=float).ravel()
    if p_values.shape != q_values.shape:
        raise ValueError("JS divergence requires aligned probability vectors.")
    p_sum = float(np.sum(p_values))
    q_sum = float(np.sum(q_values))
    if p_sum <= 0 or q_sum <= 0:
        raise ValueError("JS divergence requires nonempty probability vectors.")
    p_values = p_values / p_sum
    q_values = q_values / q_sum
    m = 0.5 * (p_values + q_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        p_terms = np.where(p_values > 0, p_values * np.log(p_values / m), 0.0)
        q_terms = np.where(q_values > 0, q_values * np.log(q_values / m), 0.0)
    value = float(0.5 * np.sum(p_terms) + 0.5 * np.sum(q_terms))
    if value < 0 and abs(value) < 1e-15:
        return 0.0
    return max(value, 0.0)


def _profile_from_weight(absolute_weight: pd.DataFrame) -> NetworkProfile:
    normalized, w_max = normalize_weight_matrix(absolute_weight)
    adjacency = build_adjacency_matrix(normalized)
    edge_length = build_edge_length_matrix(normalized)
    shortest = all_pairs_shortest_paths(edge_length)
    scaled, discrete, no_finite_paths = _scale_and_discretize(shortest)
    probability, mu, categories = distance_probability_profile(shortest.weighted, discrete)
    edge_count = int(np.sum(np.triu(adjacency.to_numpy(dtype=int), k=1)))
    degenerate = edge_count == 0 or no_finite_paths
    alpha = weighted_alpha_centrality_distribution(normalized)
    complement = 1.0 - normalized
    for i in range(min(complement.shape)):
        complement.iat[i, i] = 0.0
    alpha_complement = weighted_alpha_centrality_distribution(complement)
    return NetworkProfile(
        absolute_weight=absolute_weight,
        normalized_weight=normalized,
        adjacency=adjacency,
        edge_length=edge_length,
        weighted_shortest=shortest.weighted,
        binary_shortest=shortest.binary,
        scaled_distance=scaled,
        discrete_distance=discrete,
        distance_probability=probability,
        mu=mu,
        alpha=alpha,
        alpha_complement=alpha_complement,
        wnnd=_wnnd(probability, categories, degenerate),
        w_max=w_max,
        edge_count=edge_count,
        disconnected_pair_count=_disconnected_pair_count(shortest.weighted),
        degenerate_network=degenerate,
        finite_distance_categories=categories,
    )


def _align_probability_rows(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    left_finite = [column for column in left.columns if column != "disconnected"]
    right_finite = [column for column in right.columns if column != "disconnected"]
    finite = sorted(set(left_finite) | set(right_finite), key=lambda value: int(value))
    columns = finite + (["disconnected"] if "disconnected" in set(left.columns) | set(right.columns) else [])
    return left.reindex(columns=columns, fill_value=0.0), right.reindex(columns=columns, fill_value=0.0)


def weighted_network_distance_details(
    before_correlation: pd.DataFrame,
    before_fdr: pd.DataFrame | None,
    after_correlation: pd.DataFrame,
    after_fdr: pd.DataFrame | None,
    nodes: list[str],
) -> dict[str, object]:
    """Calculate WD and return scalar contributions plus intermediate matrices."""
    before_corr = _matrix(before_correlation, nodes)
    after_corr = _matrix(after_correlation, nodes)
    before_q = _matrix(before_fdr, nodes) if before_fdr is not None else pd.DataFrame(np.nan, index=nodes, columns=nodes)
    after_q = _matrix(after_fdr, nodes) if after_fdr is not None else pd.DataFrame(np.nan, index=nodes, columns=nodes)
    before_profile = _profile_from_weight(build_absolute_weight_matrix(before_corr, nodes))
    after_profile = _profile_from_weight(build_absolute_weight_matrix(after_corr, nodes))

    before_mu_aligned, after_mu_aligned = _align_probability_rows(before_profile.mu, after_profile.mu)
    global_component = float(np.sqrt(jensen_shannon_divergence(before_mu_aligned, after_mu_aligned) / np.log(2)))
    local_component = float(abs(np.sqrt(before_profile.wnnd) - np.sqrt(after_profile.wnnd)))
    alpha_original_component = float(
        np.sqrt(jensen_shannon_divergence(before_profile.alpha["probability"], after_profile.alpha["probability"]) / np.log(2))
    )
    alpha_complement_component = float(
        np.sqrt(
            jensen_shannon_divergence(
                before_profile.alpha_complement["probability"],
                after_profile.alpha_complement["probability"],
            )
            / np.log(2)
        )
    )
    global_contribution = 0.45 * global_component
    local_contribution = 0.45 * local_component
    alpha_contribution = 0.05 * (alpha_original_component + alpha_complement_component)
    distance = global_contribution + local_contribution + alpha_contribution

    return {
        "weighted_network_distance": float(distance),
        "global_distance_component": global_component,
        "global_contribution": global_contribution,
        "local_heterogeneity_component": local_component,
        "local_contribution": local_contribution,
        "alpha_original_component": alpha_original_component,
        "alpha_complement_component": alpha_complement_component,
        "alpha_contribution": alpha_contribution,
        "WNND_before": float(before_profile.wnnd),
        "WNND_after": float(after_profile.wnnd),
        "w_max_before": float(before_profile.w_max),
        "w_max_after": float(after_profile.w_max),
        "edge_count_before": int(before_profile.edge_count),
        "edge_count_after": int(after_profile.edge_count),
        "disconnected_pair_count_before": int(before_profile.disconnected_pair_count),
        "disconnected_pair_count_after": int(after_profile.disconnected_pair_count),
        "degenerate_network_before": bool(before_profile.degenerate_network),
        "degenerate_network_after": bool(after_profile.degenerate_network),
        "distance_input": "full_correlation_matrix",
        "distance_edge_filter": "none",
        "distance_weight_transform": "absolute_spearman",
        "distance_normalization": "separate_max_normalization",
        "plot_edge_filter": "abs_r>=0.10_and_fdr_q<0.05",
        "before_correlation": before_corr,
        "after_correlation": after_corr,
        "before_fdr": before_q,
        "after_fdr": after_q,
        "before_absolute_weight": before_profile.absolute_weight,
        "after_absolute_weight": after_profile.absolute_weight,
        "before_normalized_weight": before_profile.normalized_weight,
        "after_normalized_weight": after_profile.normalized_weight,
        "before_adjacency": before_profile.adjacency,
        "after_adjacency": after_profile.adjacency,
        "before_edge_length": before_profile.edge_length,
        "after_edge_length": after_profile.edge_length,
        "before_weighted_shortest": before_profile.weighted_shortest,
        "after_weighted_shortest": after_profile.weighted_shortest,
        "before_binary_shortest": before_profile.binary_shortest,
        "after_binary_shortest": after_profile.binary_shortest,
        "before_scaled_distance": before_profile.scaled_distance,
        "after_scaled_distance": after_profile.scaled_distance,
        "before_discrete_distance": before_profile.discrete_distance,
        "after_discrete_distance": after_profile.discrete_distance,
        "before_distance_probability": before_profile.distance_probability,
        "after_distance_probability": after_profile.distance_probability,
        "before_mu": before_profile.mu,
        "after_mu": after_profile.mu,
        "before_alpha": before_profile.alpha,
        "after_alpha": after_profile.alpha,
        "before_alpha_complement": before_profile.alpha_complement,
        "after_alpha_complement": after_profile.alpha_complement,
    }


def weighted_network_distance(
    before_correlation: pd.DataFrame,
    before_fdr: pd.DataFrame | None,
    after_correlation: pd.DataFrame,
    after_fdr: pd.DataFrame | None,
    nodes: list[str],
) -> float:
    return float(
        weighted_network_distance_details(
            before_correlation,
            before_fdr,
            after_correlation,
            after_fdr,
            nodes,
        )["weighted_network_distance"]
    )
