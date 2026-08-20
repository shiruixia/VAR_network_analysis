from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from utils.feature_config import FEATURES
from utils.figure2_paths import figure1_result_root


@dataclass(frozen=True)
class DatasetConfig:
    key: str
    display_name: str
    source_dir: Path
    correlation_matrix: Path
    p_value_matrix: Path
    fdr_matrix: Path
    adjacency_matrix: Path
    edge_list: Path
    network_metrics: Path
    features: tuple[str, ...]


def dataset_config(dataset_key: str) -> DatasetConfig:
    """Return Figure2 dataset workbook paths, all sourced from Figure1 results."""
    names = {"ligue1": "Ligue1", "euro": "EURO", "worldcup": "WorldCup"}
    if dataset_key not in names:
        raise KeyError(f"Unknown Figure2 dataset: {dataset_key}")
    source_dir = figure1_result_root() / dataset_key
    return DatasetConfig(
        key=dataset_key,
        display_name=names[dataset_key],
        source_dir=source_dir,
        correlation_matrix=source_dir / "correlation_matrix.xlsx",
        p_value_matrix=source_dir / "p_value_matrix.xlsx",
        fdr_matrix=source_dir / "fdr_matrix.xlsx",
        adjacency_matrix=source_dir / "adjacency_matrix.xlsx",
        edge_list=source_dir / "edge_list.xlsx",
        network_metrics=source_dir / "network_metrics.xlsx",
        features=tuple(FEATURES),
    )


def dataset_configs() -> dict[str, DatasetConfig]:
    """Return all Figure2 dataset path configs."""
    return {key: dataset_config(key) for key in ("ligue1", "euro", "worldcup")}
