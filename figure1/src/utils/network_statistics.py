"""Shared Spearman/FDR/edge-list functions for Figure1 network data."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
try:
    from statsmodels.stats.multitest import multipletests
except ModuleNotFoundError:
    def multipletests(pvals, method="fdr_bh"):
        """Fallback Benjamini-Hochberg FDR adjustment matching statsmodels output use."""
        if method != "fdr_bh":
            raise ValueError(f"Unsupported fallback multipletests method: {method}")
        pvalues = np.asarray(pvals, dtype=float)
        n_tests = len(pvalues)
        order = np.argsort(pvalues)
        ranked = pvalues[order]
        adjusted_ranked = ranked * n_tests / np.arange(1, n_tests + 1)
        adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
        adjusted_ranked = np.clip(adjusted_ranked, 0.0, 1.0)
        adjusted = np.empty_like(adjusted_ranked)
        adjusted[order] = adjusted_ranked
        return None, adjusted, None, None


def compute_spearman_matrix(data: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    corr, _ = _compute_pairwise_spearman(data, nodes)
    return corr


def compute_pvalue_matrix(data: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    _, pvalues = _compute_pairwise_spearman(data, nodes)
    return pvalues


def compute_fdr_matrix(pvalue_matrix: pd.DataFrame) -> pd.DataFrame:
    nodes = list(pvalue_matrix.index)
    fdr = pd.DataFrame(np.zeros((len(nodes), len(nodes))), index=nodes, columns=nodes, dtype=float)
    raw_pvalues: list[float] = []
    raw_pairs: list[tuple[str, str]] = []
    for i, source in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            p_value = pvalue_matrix.loc[source, target]
            if np.isfinite(p_value):
                raw_pairs.append((source, target))
                raw_pvalues.append(float(p_value))
            else:
                fdr.loc[source, target] = fdr.loc[target, source] = np.nan
    if raw_pvalues:
        adjusted = multipletests(raw_pvalues, method="fdr_bh")[1]
        for (source, target), value in zip(raw_pairs, adjusted):
            fdr.loc[source, target] = fdr.loc[target, source] = float(value)
    return fdr


def build_edge_list(
    corr: pd.DataFrame,
    pvalues: pd.DataFrame,
    fdr: pd.DataFrame,
    strict_abs_r: float = 0.10,
    strict_fdr: float = 0.05,
    relaxed_abs_r: float = 0.10,
) -> pd.DataFrame:
    nodes = list(corr.index)
    rows = []
    for i, source in enumerate(nodes):
        for j in range(i + 1, len(nodes)):
            target = nodes[j]
            r = float(corr.loc[source, target]) if np.isfinite(corr.loc[source, target]) else np.nan
            p = float(pvalues.loc[source, target]) if np.isfinite(pvalues.loc[source, target]) else np.nan
            q = float(fdr.loc[source, target]) if np.isfinite(fdr.loc[source, target]) else np.nan
            abs_r = abs(r) if np.isfinite(r) else np.nan
            significant = bool(np.isfinite(q) and q < strict_fdr)
            rows.append(
                {
                    "source": source,
                    "target": target,
                    "spearman_r": r,
                    "abs_r": abs_r,
                    "p_value": p,
                    "p_fdr": q,
                    "sign": "positive" if np.isfinite(r) and r > 0 else "negative" if np.isfinite(r) and r < 0 else "zero_or_missing",
                    "keep_strict": bool(np.isfinite(abs_r) and abs_r >= strict_abs_r and significant),
                    "keep_relaxed": bool(np.isfinite(abs_r) and abs_r >= relaxed_abs_r),
                }
            )
    return pd.DataFrame(rows)


def _compute_pairwise_spearman(data: pd.DataFrame, nodes: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    n = len(nodes)
    corr = pd.DataFrame(np.eye(n), index=nodes, columns=nodes, dtype=float)
    pvalues = pd.DataFrame(np.zeros((n, n)), index=nodes, columns=nodes, dtype=float)
    for i, source in enumerate(nodes):
        for j in range(i + 1, n):
            target = nodes[j]
            pair = data[[source, target]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(pair) < 4 or pair[source].nunique() < 2 or pair[target].nunique() < 2:
                r, p = np.nan, np.nan
            else:
                result = spearmanr(pair[source], pair[target])
                r, p = float(result.statistic), float(result.pvalue)
            corr.loc[source, target] = corr.loc[target, source] = r
            pvalues.loc[source, target] = pvalues.loc[target, source] = p
    return corr, pvalues
