from __future__ import annotations

from pathlib import Path


VALID_DATASETS = {"ligue1", "euro", "worldcup"}


def get_figure3_root() -> Path:
    """Return the Figure3 project root."""
    return Path(__file__).resolve().parents[2]


def get_paper_assets_root() -> Path:
    """Return the paper_assets root that owns all Figure3 inputs."""
    return get_figure3_root().parent


def _validate_dataset(dataset: str) -> str:
    key = str(dataset).lower()
    if key not in VALID_DATASETS:
        raise KeyError(f"Unknown Figure3 dataset: {dataset}")
    return key


def get_data_path(dataset: str) -> Path:
    """Return standardized match-level data path for a dataset."""
    key = _validate_dataset(dataset)
    return get_paper_assets_root() / "data" / f"{key}_matches.xlsx"


def get_table1_result_path(dataset: str) -> Path:
    """Return Table1 result path for a dataset."""
    key = _validate_dataset(dataset)
    return get_paper_assets_root() / "table1" / "results" / f"table1_{key}.xlsx"


def get_results_path(dataset: str) -> Path:
    """Return Figure3 result directory for a dataset."""
    key = _validate_dataset(dataset)
    return get_figure3_root() / "results" / key


def get_panel_path() -> Path:
    """Return Figure3 panel output directory."""
    return get_figure3_root() / "panels"
