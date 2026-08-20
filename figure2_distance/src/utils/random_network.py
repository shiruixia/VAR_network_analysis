from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from utils.match_data_loader import clean_indicator_var_data


STRICT_ABS_R = 0.10
STRICT_FDR = 0.05


def benjamini_hochberg_adjust(pvalues: list[float]) -> np.ndarray:
    """Adjust p-values using the same Figure2 Benjamini-Hochberg rule."""
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    if n == 0:
        return np.asarray([], dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted_ranked = ranked * n / np.arange(1, n + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted


def _var_column(df: pd.DataFrame) -> str:
    if "var" in df.columns:
        return "var"
    if "VAR" in df.columns:
        return "VAR"
    raise ValueError("Figure2 randomization requires a VAR/var column in match-level data.")


def split_by_permuted_var_group_sizes(df: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle rows while preserving original before/after VAR group sizes."""
    var_column = _var_column(df)
    group_a_size = int(pd.to_numeric(df[var_column], errors="coerce").eq(0).sum())
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(df))
    group_a = df.iloc[indices[:group_a_size]].reset_index(drop=True)
    group_b = df.iloc[indices[group_a_size:]].reset_index(drop=True)
    return group_a, group_b


def split_cleaned_by_permuted_var_group_sizes(df: pd.DataFrame, nodes: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Shuffle cleaned rows while preserving cleaned VAR=0/VAR=1 group sizes."""
    values, labels = clean_indicator_var_data(df, nodes)
    group_a_size = int(labels.eq(0).sum())
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(values))
    group_a = values.iloc[indices[:group_a_size]].reset_index(drop=True)
    group_b = values.iloc[indices[group_a_size:]].reset_index(drop=True)
    assert group_a_size + len(group_b) == len(values)
    return group_a, group_b


def compute_random_network_matrices(data: pd.DataFrame, nodes: list[str]) -> dict[str, pd.DataFrame]:
    """Construct random split Spearman/FDR matrices using the original Figure2 thresholds."""
    n = len(nodes)
    corr = pd.DataFrame(np.eye(n), index=nodes, columns=nodes, dtype=float)
    pvalues = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes, dtype=float)
    fdr = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes, dtype=float)
    raw_pvalues: list[float] = []
    raw_pairs: list[tuple[str, str]] = []

    for i, source in enumerate(nodes):
        for j in range(i + 1, n):
            target = nodes[j]
            pair = data[[source, target]].apply(pd.to_numeric, errors="coerce").dropna()
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

    adjacency = pd.DataFrame(0, index=nodes, columns=nodes, dtype=int)
    edge_rows = []
    for i, source in enumerate(nodes):
        for j in range(i + 1, n):
            target = nodes[j]
            r_value = corr.loc[source, target]
            p_value = pvalues.loc[source, target]
            q_value = fdr.loc[source, target]
            keep = bool(np.isfinite(r_value) and abs(r_value) >= STRICT_ABS_R and np.isfinite(q_value) and q_value < STRICT_FDR)
            if keep:
                adjacency.loc[source, target] = adjacency.loc[target, source] = 1
                edge_rows.append(
                    {
                        "source": source,
                        "target": target,
                        "correlation": float(r_value),
                        "weight": float(r_value),
                        "spearman_r": float(r_value),
                        "abs_r": abs(float(r_value)),
                        "p_value": float(p_value) if np.isfinite(p_value) else np.nan,
                        "fdr_q_value": float(q_value) if np.isfinite(q_value) else np.nan,
                        "edge_sign": "positive" if r_value >= 0 else "negative",
                        "sign": "positive" if r_value >= 0 else "negative",
                        "edge_weight": abs(float(r_value)),
                        "keep_strict": True,
                    }
                )

    return {
        "correlation_matrix": corr,
        "pvalue_matrix": pvalues,
        "fdr_matrix": fdr,
        "adjacency_matrix": adjacency,
        "edge_list": pd.DataFrame(edge_rows),
    }


def random_edges_from_match_level_data(
    match_data: pd.DataFrame,
    nodes: list[str],
    dataset: str,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, pd.DataFrame]]:
    """Create random group A/B edge lists from match-level data using the original random split."""
    group_a, group_b = split_cleaned_by_permuted_var_group_sizes(match_data, nodes, seed)
    matrices_a = compute_random_network_matrices(group_a, nodes)
    matrices_b = compute_random_network_matrices(group_b, nodes)
    edges_a = matrices_a["edge_list"].copy()
    edges_b = matrices_b["edge_list"].copy()
    edges_a["dataset"] = dataset
    edges_b["dataset"] = dataset
    edges_a["group"] = "random_groupA"
    edges_b["group"] = "random_groupB"
    edges_a["seed"] = seed
    edges_b["seed"] = seed
    return pd.concat([edges_a, edges_b], ignore_index=True), matrices_a, matrices_b
