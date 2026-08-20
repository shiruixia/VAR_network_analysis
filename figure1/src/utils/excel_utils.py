"""Excel writers for Figure1 network data outputs."""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_matrix_excel(matrix: pd.DataFrame, path: str | Path, sheet_name: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    output = matrix_sheet(matrix)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        output.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return path


def write_table_excel(table: pd.DataFrame, path: str | Path, sheet_name: str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name=sheet_name[:31])
    return path


def matrix_sheet(matrix: pd.DataFrame) -> pd.DataFrame:
    output = matrix.copy()
    output.insert(0, "metric", output.index)
    return output.reset_index(drop=True)


def edge_sheet(edge_list: pd.DataFrame) -> pd.DataFrame:
    output = edge_list.copy()
    if "weight" not in output.columns and "spearman_r" in output.columns:
        output["weight"] = output["spearman_r"]
    columns = ["source", "target", "weight", "sign"] + [
        column for column in output.columns if column not in {"source", "target", "weight", "sign"}
    ]
    return output.loc[:, [column for column in columns if column in output.columns]]


def write_dataset_network_outputs(
    output_dir: str | Path,
    correlation_matrices: dict[str, pd.DataFrame],
    p_value_matrices: dict[str, pd.DataFrame],
    fdr_matrices: dict[str, pd.DataFrame],
    adjacency_matrices: dict[str, pd.DataFrame],
    edge_lists: dict[str, pd.DataFrame],
    network_metrics: dict[str, pd.DataFrame],
) -> dict[str, Path]:
    """Write standard Figure1 network output workbooks with before/after sheets."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    outputs = {
        "correlation_matrix": output_path / "correlation_matrix.xlsx",
        "p_value_matrix": output_path / "p_value_matrix.xlsx",
        "fdr_matrix": output_path / "fdr_matrix.xlsx",
        "adjacency_matrix": output_path / "adjacency_matrix.xlsx",
        "edge_list": output_path / "edge_list.xlsx",
        "network_metrics": output_path / "network_metrics.xlsx",
    }
    _write_matrix_workbook(outputs["correlation_matrix"], correlation_matrices)
    _write_matrix_workbook(outputs["p_value_matrix"], p_value_matrices)
    _write_matrix_workbook(outputs["fdr_matrix"], fdr_matrices)
    _write_matrix_workbook(outputs["adjacency_matrix"], adjacency_matrices)
    _write_table_workbook(outputs["edge_list"], {key: edge_sheet(value) for key, value in edge_lists.items()})
    _write_table_workbook(outputs["network_metrics"], network_metrics)
    return outputs


def _write_matrix_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in ["before_var", "after_var"]:
            matrix_sheet(sheets[sheet_name]).to_excel(writer, index=False, sheet_name=sheet_name)


def _write_table_workbook(path: Path, sheets: dict[str, pd.DataFrame]) -> None:
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for sheet_name in ["before_var", "after_var"]:
            sheets[sheet_name].to_excel(writer, index=False, sheet_name=sheet_name)
