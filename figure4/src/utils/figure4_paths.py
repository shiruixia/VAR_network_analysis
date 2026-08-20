from __future__ import annotations

from pathlib import Path


def figure4_root() -> Path:
    """Return the Figure4 project root directory."""

    return Path(__file__).resolve().parents[2]


def paper_assets_root() -> Path:
    """Return the shared paper_assets directory."""

    return figure4_root().parent


def results_dir(dataset: str | None = None) -> Path:
    """Return Figure4 results directory, optionally scoped to a dataset."""

    base = figure4_root() / "results"
    return base if dataset is None else base / str(dataset).lower()


def panels_dir() -> Path:
    """Return Figure4 panels directory."""

    return figure4_root() / "panels"


def data_dir() -> Path:
    """Return standardized match-level data directory."""

    return paper_assets_root() / "data"


def table1_results_dir() -> Path:
    """Return Table1 results directory used for significant indicators."""

    return paper_assets_root() / "table1" / "results"


def match_data_path(dataset: str) -> Path:
    """Return standardized match-level workbook path for a dataset."""

    return data_dir() / f"{str(dataset).lower()}_matches.xlsx"


def table1_result_path(dataset: str) -> Path:
    """Return Table1 workbook path for a dataset."""

    return table1_results_dir() / f"table1_{str(dataset).lower()}.xlsx"
