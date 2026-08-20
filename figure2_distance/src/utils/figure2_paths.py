from __future__ import annotations

from pathlib import Path


def figure2_root() -> Path:
    """Return the Figure2 project root."""
    return Path(__file__).resolve().parents[2]


def paper_assets_root() -> Path:
    """Return the article/paper_assets directory."""
    return figure2_root().parent


def match_data_root() -> Path:
    """Return the standardized match-level input directory."""
    return paper_assets_root() / "data"


def figure1_result_root() -> Path:
    """Return the only Figure1 result input root used for real networks."""
    return paper_assets_root() / "figure1" / "result"


def results_root() -> Path:
    """Return the Figure2 numerical result directory."""
    return figure2_root() / "results"


def dataset_results_dir(dataset: str) -> Path:
    """Return the standardized Figure2 result directory for one dataset."""
    return results_root() / dataset


def legacy_results_root() -> Path:
    """Return the archived Figure2 result directory for legacy files."""
    return figure2_root() / "src" / "_legacy" / "results"


def table_output_dir() -> Path:
    """Return archived legacy table outputs. Main pipelines must not read this."""
    return legacy_results_root() / "tables"


def panels_dir() -> Path:
    """Return the Figure2 image output directory."""
    return figure2_root() / "panels"


def random_network_examples_dir() -> Path:
    """Return the image output directory for random network examples."""
    return panels_dir()


def ensure_output_dirs() -> None:
    """Create standard Figure2 output directories without recreating results/tables."""
    for path in (results_root(), panels_dir()):
        path.mkdir(parents=True, exist_ok=True)
    for dataset in ("ligue1", "euro", "worldcup"):
        dataset_results_dir(dataset).mkdir(parents=True, exist_ok=True)
