from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.figure2_data_loader import LoadedNetworkResult
from utils.match_data_loader import clean_indicator_var_data
from utils.random_network import benjamini_hochberg_adjust
from utils.weighted_network_distance import (
    ABS_R_THRESHOLD,
    FDR_THRESHOLD,
    weighted_network_distance_details,
)


RANDOM_SEED = 42
N_PERMUTATIONS = 1000


def upper_values(matrix: np.ndarray) -> np.ndarray:
    """Return upper-triangle values excluding the diagonal."""
    return matrix[np.triu_indices_from(matrix, k=1)]


def correlation_matrix(values: np.ndarray) -> np.ndarray:
    """Calculate a Spearman indicator correlation matrix."""
    result = spearmanr(values, axis=0, nan_policy="omit").statistic
    result = np.asarray(result, dtype=float)
    if result.ndim == 0:
        raise ValueError("Correlation matrix could not be calculated.")
    np.fill_diagonal(result, 1.0)
    return result


def spearman_correlation_pvalue_fdr(values: np.ndarray, nodes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate Spearman r, raw p, and BH-FDR matrices for one network."""
    n = len(nodes)
    corr = pd.DataFrame(np.eye(n), index=nodes, columns=nodes, dtype=float)
    pvalues = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes, dtype=float)
    fdr = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes, dtype=float)
    raw_pvalues: list[float] = []
    raw_pairs: list[tuple[str, str]] = []
    frame = pd.DataFrame(values, columns=nodes)

    for i, source in enumerate(nodes):
        for j in range(i + 1, n):
            target = nodes[j]
            pair = frame[[source, target]].dropna()
            if len(pair) < 4 or pair[source].nunique() < 2 or pair[target].nunique() < 2:
                r_value, p_value = np.nan, np.nan
            else:
                result = spearmanr(pair[source], pair[target])
                r_value, p_value = float(result.statistic), float(result.pvalue)
            corr.loc[source, target] = corr.loc[target, source] = r_value
            pvalues.loc[source, target] = pvalues.loc[target, source] = p_value
            if np.isfinite(p_value):
                raw_pairs.append((source, target))
                raw_pvalues.append(float(p_value))

    if raw_pvalues:
        adjusted = benjamini_hochberg_adjust(raw_pvalues)
        for (source, target), q_value in zip(raw_pairs, adjusted):
            fdr.loc[source, target] = fdr.loc[target, source] = float(q_value)
    else:
        fdr.loc[:, :] = np.nan
    for i in range(min(fdr.shape)):
        fdr.iat[i, i] = 0.0
    return corr, pvalues, fdr


def spearman_correlation_frame(values: np.ndarray, nodes: list[str]) -> pd.DataFrame:
    """Calculate a complete Spearman correlation matrix for WD."""
    corr = correlation_matrix(values)
    return pd.DataFrame(corr, index=nodes, columns=nodes)


def difference_statistics(corr0: np.ndarray, corr1: np.ndarray) -> dict[str, float]:
    """Calculate diagnostic full-matrix difference metrics."""
    delta = corr1 - corr0
    upper0 = upper_values(corr0)
    upper1 = upper_values(corr1)
    upper_delta = upper_values(delta)
    similarity = (
        float(np.corrcoef(upper0, upper1)[0, 1])
        if np.std(upper0) > 0 and np.std(upper1) > 0
        else np.nan
    )
    return {
        "frobenius_distance_diagnostic": float(np.linalg.norm(delta, ord="fro")),
        "mean_absolute_delta_r": float(np.nanmean(np.abs(upper_delta))),
        "sum_abs_delta_r": float(np.nansum(np.abs(upper_delta))),
        "max_absolute_delta_r": float(np.nanmax(np.abs(upper_delta))),
        "matrix_correlation_similarity": similarity,
    }


def observed_weighted_network_distance(data: LoadedNetworkResult) -> dict[str, object]:
    """Calculate observed Figure2 WD from Figure1 correlation and FDR matrices."""
    details = weighted_network_distance_details(
        data.before_correlation_matrix,
        data.before_fdr_matrix,
        data.after_correlation_matrix,
        data.after_fdr_matrix,
        data.nodes,
    )
    diagnostics = difference_statistics(
        data.before_correlation_matrix.loc[data.nodes, data.nodes].to_numpy(dtype=float),
        data.after_correlation_matrix.loc[data.nodes, data.nodes].to_numpy(dtype=float),
    )
    details.update(diagnostics)
    return details


def matrix_similarity(corr0: np.ndarray, corr1: np.ndarray) -> float:
    """Return upper-triangle correlation similarity between two matrices."""
    return difference_statistics(corr0, corr1)["matrix_correlation_similarity"]


def _distribution_record(permutation_id: int, details: dict[str, object]) -> dict[str, object]:
    return {
        "permutation_id": permutation_id,
        "weighted_network_distance": float(details["weighted_network_distance"]),
        "global_distance_component": float(details["global_distance_component"]),
        "local_heterogeneity_component": float(details["local_heterogeneity_component"]),
        "alpha_original_component": float(details["alpha_original_component"]),
        "alpha_complement_component": float(details["alpha_complement_component"]),
        "edge_count_group0": int(details["edge_count_before"]),
        "edge_count_group1": int(details["edge_count_after"]),
        "disconnected_pair_count_group0": int(details["disconnected_pair_count_before"]),
        "disconnected_pair_count_group1": int(details["disconnected_pair_count_after"]),
        "degenerate_network_group0": bool(details["degenerate_network_before"]),
        "degenerate_network_group1": bool(details["degenerate_network_after"]),
    }


def run_permutations(
    values: np.ndarray,
    group0_size: int,
    nodes: list[str],
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Run label permutations, recomputing complete WD for each split."""
    rng = np.random.default_rng(random_seed)
    records: list[dict[str, object]] = []
    n_samples = len(values)
    group1_size = n_samples - group0_size
    assert group0_size + group1_size == n_samples
    for permutation_id in range(1, n_permutations + 1):
        labels = rng.permutation(np.r_[np.zeros(group0_size, dtype=int), np.ones(group1_size, dtype=int)])
        assert len(labels) == len(values)
        corr0 = spearman_correlation_frame(values[labels == 0], nodes)
        corr1 = spearman_correlation_frame(values[labels == 1], nodes)
        details = weighted_network_distance_details(corr0, None, corr1, None, nodes)
        records.append(_distribution_record(permutation_id, details))
    return pd.DataFrame(records)


def build_permutation_results(
    dataset: str,
    observed_metrics: dict[str, object],
    distribution: pd.DataFrame,
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Summarize observed WD against the permutation null distribution."""
    observed = float(observed_metrics["weighted_network_distance"])
    null = distribution["weighted_network_distance"].to_numpy(dtype=float)
    empirical_p = (int(np.sum(null >= observed)) + 1) / (len(null) + 1)
    return pd.DataFrame(
        [
            {
                "dataset": dataset,
                "metric": "weighted_network_distance",
                "description": (
                    "Weighted Network Distance using abs(r)>=0.10 and BH-FDR q<0.05 "
                    "inside each observed or permuted network"
                ),
                "observed_value": observed,
                "null_mean": float(np.mean(null)),
                "null_std": float(np.std(null, ddof=1)),
                "null_95_percentile": float(np.quantile(null, 0.95)),
                "empirical_p_value": float(empirical_p),
                "significant_at_0_05": bool(empirical_p < 0.05),
                "n_permutations": int(n_permutations),
                "random_seed": int(random_seed),
                "distribution_source": "paper_assets/data",
                "distance_input": "full_correlation_matrix",
                "distance_edge_filter": "none",
                "distance_weight_transform": "absolute_spearman",
                "distance_normalization": "separate_max_normalization",
                "plot_edge_filter": "abs_r>=0.10_and_fdr_q<0.05",
                "plot_abs_r_threshold": float(ABS_R_THRESHOLD),
                "plot_fdr_threshold": float(FDR_THRESHOLD),
            }
        ]
    )


def permutation_analysis_from_values(
    values: np.ndarray,
    group0_size: int,
    nodes: list[str],
    observed_metrics: dict[str, object],
    dataset: str = "unknown",
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run Figure2 WD permutation analysis from raw indicator values."""
    distribution = run_permutations(values, group0_size, nodes, n_permutations, random_seed)
    summary = build_permutation_results(dataset, observed_metrics, distribution, n_permutations, random_seed)
    return observed_metrics, distribution, summary


def permutation_analysis_from_match_level_data(
    match_data: pd.DataFrame,
    nodes: list[str],
    observed_metrics: dict[str, object],
    dataset: str = "unknown",
    n_permutations: int = N_PERMUTATIONS,
    random_seed: int = RANDOM_SEED,
) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run Figure2 WD permutation analysis from standardized match-level data."""
    values_frame, labels = clean_indicator_var_data(match_data, nodes)
    values = values_frame.to_numpy(dtype=float)
    group0_size = int(labels.eq(0).sum())
    group1_size = int(labels.eq(1).sum())
    assert len(labels) == len(values)
    assert group0_size + group1_size == len(values)
    return permutation_analysis_from_values(
        values,
        group0_size,
        nodes,
        observed_metrics,
        dataset=dataset,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )


def permutation_analysis(data: LoadedNetworkResult) -> tuple[dict[str, object], pd.DataFrame, pd.DataFrame]:
    """Run observed WD plus random permutation distribution for one dataset."""
    if data.indicator_data is None:
        raise ValueError(
            "Permutation analysis requires raw indicator data generated by Figure2; "
            "the Phase 1 loader reads only Figure1 result workbooks."
        )
    return permutation_analysis_from_match_level_data(
        data.indicator_data,
        data.nodes,
        observed_weighted_network_distance(data),
        dataset=data.dataset_key,
    )
