from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_excel_workbook(path: str | Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """Write a Figure3 multi-sheet Excel workbook."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Figure3 Excel output must use .xlsx: {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=str(sheet_name)[:31], index=False)
    return output_path


def write_node_inclusion_order(path: str | Path, order: pd.DataFrame, path_definitions: pd.DataFrame | None = None) -> Path:
    """Save node inclusion order workbook."""
    sheets = {"node_order": order.copy()}
    if path_definitions is not None:
        sheets["path_definitions"] = path_definitions.copy()
    return write_excel_workbook(path, sheets)


def write_distance_curve(path: str | Path, curve: pd.DataFrame) -> Path:
    """Save the weighted network distance trajectory workbook."""
    return write_excel_workbook(path, {"weighted_distance_curve": curve.copy()})


def write_permutation_results(
    path: str | Path,
    permutation_summary: pd.DataFrame,
    permutation_distribution: pd.DataFrame | None = None,
) -> Path:
    """Save Figure3 permutation summary and distribution workbook."""
    sheets = {"permutation_summary": permutation_summary.copy()}
    if permutation_distribution is not None:
        sheets["permutation_distribution"] = permutation_distribution.copy()
    return write_excel_workbook(path, sheets)


def write_statistics_summary(path: str | Path, summary: pd.DataFrame, parameters: pd.DataFrame | None = None) -> Path:
    """Save Figure3 statistics summary workbook."""
    sheets = {"statistics_summary": summary.copy()}
    if parameters is not None:
        sheets["parameters"] = parameters.copy()
    return write_excel_workbook(path, sheets)
