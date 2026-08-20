"""Run the Figure1 standardized network pipeline for World Cup."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


DATASET = "worldcup"


def _add_src_to_path() -> Path:
    src_dir = Path(__file__).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


_add_src_to_path()

from utils.indicator_config import MATCH_INDICATORS
from utils.pipeline_utils import (
    figure1_root,
    indicator_frame_from_matches,
    load_standardized_match_file,
    split_before_after_var,
    standardized_data_dir,
)
from utils.network_statistics import (
    build_edge_list,
    compute_fdr_matrix,
    compute_pvalue_matrix,
    compute_spearman_matrix,
)
from utils.network_analysis import (
    build_graph,
    compute_network_metrics,
    strict_adjacency_matrix,
)
from utils.excel_utils import write_dataset_network_outputs


def _compute_group_outputs(group: pd.DataFrame, var_value: int) -> dict[str, object]:
    """Compute Figure1 matrices, edges, and network metrics for one VAR group."""
    indicator_data = group.loc[:, MATCH_INDICATORS].copy()
    correlation = compute_spearman_matrix(indicator_data, MATCH_INDICATORS)
    p_values = compute_pvalue_matrix(indicator_data, MATCH_INDICATORS)
    fdr = compute_fdr_matrix(p_values)
    edges = build_edge_list(correlation, p_values, fdr)
    adjacency = strict_adjacency_matrix(edges, MATCH_INDICATORS)
    strict_graph = build_graph(MATCH_INDICATORS, edges, keep_column="keep_strict")
    relaxed_graph = build_graph(MATCH_INDICATORS, edges, keep_column="keep_relaxed")
    metrics = pd.DataFrame(
        [
            compute_network_metrics(
                var_value=var_value,
                strict_graph=strict_graph,
                relaxed_graph=relaxed_graph,
                n_matches=len(group),
            )
        ]
    )
    return {
        "correlation": correlation,
        "p_values": p_values,
        "fdr": fdr,
        "edges": edges,
        "adjacency": adjacency,
        "metrics": metrics,
    }


def run_pipeline() -> dict[str, object]:
    """Run the dataset pipeline and write standard Figure1 result workbooks."""
    matches = load_standardized_match_file(DATASET)
    indicators = indicator_frame_from_matches(matches)
    before_var, after_var = split_before_after_var(indicators)

    before_outputs = _compute_group_outputs(before_var, var_value=0)
    after_outputs = _compute_group_outputs(after_var, var_value=1)

    output_dir = figure1_root() / "result" / DATASET
    output_files = write_dataset_network_outputs(
        output_dir=output_dir,
        correlation_matrices={
            "before_var": before_outputs["correlation"],
            "after_var": after_outputs["correlation"],
        },
        p_value_matrices={
            "before_var": before_outputs["p_values"],
            "after_var": after_outputs["p_values"],
        },
        fdr_matrices={
            "before_var": before_outputs["fdr"],
            "after_var": after_outputs["fdr"],
        },
        adjacency_matrices={
            "before_var": before_outputs["adjacency"],
            "after_var": after_outputs["adjacency"],
        },
        edge_lists={
            "before_var": before_outputs["edges"],
            "after_var": after_outputs["edges"],
        },
        network_metrics={
            "before_var": before_outputs["metrics"],
            "after_var": after_outputs["metrics"],
        },
    )

    return {
        "dataset": DATASET,
        "input_file": standardized_data_dir() / f"{DATASET}_matches.xlsx",
        "output_dir": output_dir,
        "output_files": list(output_files.values()),
        "total_count": len(indicators),
        "before_count": len(before_var),
        "after_count": len(after_var),
    }


def main() -> None:
    result = run_pipeline()
    print(f"{result['dataset']} pipeline complete")
    print(f"input: {result['input_file']}")
    print(f"total matches: {result['total_count']}")
    print(f"before_var: {result['before_count']}")
    print(f"after_var: {result['after_count']}")
    print(f"output_dir: {result['output_dir']}")


if __name__ == "__main__":
    main()
