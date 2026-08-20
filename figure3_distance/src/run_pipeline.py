from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils.excel_utils import (
    write_distance_curve,
    write_node_inclusion_order,
    write_permutation_results,
    write_statistics_summary,
)
from utils.figure3_paths import get_data_path, get_results_path
from utils.node_selection import build_node_inclusion_path, get_node_inclusion_order
from utils.statistics import (
    B,
    METHOD_DESCRIPTION,
    RANDOM_SEED,
    build_parameter_table,
    build_statistics_summary,
    compute_node_inclusion_results,
)


DATASETS = ("ligue1", "euro", "worldcup")


def run_dataset(dataset: str, n_permutations: int = B, random_seed: int = RANDOM_SEED) -> list[Path]:
    """Run the Figure3 WD pipeline for one dataset using paper_assets inputs."""
    match_data = pd.read_excel(get_data_path(dataset))
    node_order = get_node_inclusion_order(dataset)
    path_definitions = build_node_inclusion_path(node_order)
    weighted_curve, permutation_summary, permutation_distribution = compute_node_inclusion_results(
        match_data=match_data,
        path_definitions=path_definitions,
        dataset=dataset,
        n_permutations=n_permutations,
        random_seed=random_seed,
    )
    summary = build_statistics_summary(dataset, match_data, node_order, METHOD_DESCRIPTION)
    parameters = build_parameter_table(n_permutations=n_permutations, random_seed=random_seed)
    output_dir = get_results_path(dataset)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        write_node_inclusion_order(output_dir / "node_inclusion_order.xlsx", node_order, path_definitions),
        write_distance_curve(output_dir / "weighted_network_distance_curve.xlsx", weighted_curve),
        write_permutation_results(
            output_dir / "permutation_results.xlsx",
            permutation_summary,
            permutation_distribution,
        ),
        write_statistics_summary(output_dir / "statistics_summary.xlsx", summary, parameters),
    ]


def run_all(n_permutations: int = B, random_seed: int = RANDOM_SEED) -> list[Path]:
    """Run the Figure3 WD pipeline for Ligue 1, EURO, and World Cup."""
    outputs: list[Path] = []
    for dataset in DATASETS:
        outputs.extend(run_dataset(dataset, n_permutations=n_permutations, random_seed=random_seed))
    return outputs


def main() -> list[Path]:
    return run_all()


if __name__ == "__main__":
    for path in main():
        print(path)
