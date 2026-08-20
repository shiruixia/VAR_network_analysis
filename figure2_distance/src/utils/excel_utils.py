from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.figure2_paths import dataset_results_dir, results_root


def ensure_results_dirs() -> None:
    """Create standardized Figure2 numerical result directories."""
    results_root().mkdir(parents=True, exist_ok=True)
    for dataset in ("ligue1", "euro", "worldcup"):
        dataset_results_dir(dataset).mkdir(parents=True, exist_ok=True)


def _as_dataframe(data: pd.DataFrame | dict | list[dict]) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if isinstance(data, dict):
        return pd.DataFrame([data])
    return pd.DataFrame(data)


def write_excel_workbook(path: str | Path, sheets: dict[str, pd.DataFrame]) -> Path:
    """Write a multi-sheet Excel workbook."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Figure2 Excel output must use .xlsx: {output_path}")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, index=False, sheet_name=str(sheet_name)[:31])
    return output_path


def write_labeled_excel_workbook(path: str | Path, sheets: dict[str, pd.DataFrame], index_sheets: set[str] | None = None) -> Path:
    """Write an Excel workbook, preserving row labels for selected matrix sheets."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".xlsx":
        raise ValueError(f"Figure2 Excel output must use .xlsx: {output_path}")
    index_sheets = index_sheets or set()
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet_name, frame in sheets.items():
            frame.to_excel(writer, index=sheet_name in index_sheets, sheet_name=str(sheet_name)[:31])
    return output_path


def write_csv(path: str | Path, frame: pd.DataFrame) -> Path:
    """Write a CSV result under Figure2 results when explicitly requested."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() != ".csv":
        raise ValueError(f"Figure2 CSV output must use .csv: {output_path}")
    frame.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def write_network_comparison_result(
    dataset: str,
    observed_metrics: pd.DataFrame | dict,
    adjacency_mismatches: pd.DataFrame | None = None,
) -> Path:
    """Write standardized observed network-comparison metrics."""
    frame = _as_dataframe(observed_metrics)
    sheets = {"observed_metrics": frame}
    if adjacency_mismatches is not None:
        sheets["adjacency_mismatches"] = adjacency_mismatches.copy()
    return write_excel_workbook(
        dataset_results_dir(dataset) / "network_comparison.xlsx",
        sheets,
    )


def write_weighted_network_distance_details(dataset: str, details: dict[str, object]) -> Path:
    """Write WD scalar and intermediate matrix details for one dataset."""
    scalar_rows = []
    for key, value in details.items():
        if not isinstance(value, pd.DataFrame):
            scalar_rows.append({"metric": key, "value": value})
    sheets: dict[str, pd.DataFrame] = {"summary": pd.DataFrame(scalar_rows)}
    sheet_names = [
        "before_correlation",
        "after_correlation",
        "before_fdr",
        "after_fdr",
        "before_absolute_weight",
        "after_absolute_weight",
        "before_normalized_weight",
        "after_normalized_weight",
        "before_adjacency",
        "after_adjacency",
        "before_edge_length",
        "after_edge_length",
        "before_weighted_shortest",
        "after_weighted_shortest",
        "before_binary_shortest",
        "after_binary_shortest",
        "before_scaled_distance",
        "after_scaled_distance",
        "before_discrete_distance",
        "after_discrete_distance",
        "before_distance_probability",
        "after_distance_probability",
        "before_mu",
        "after_mu",
        "before_alpha",
        "after_alpha",
        "before_alpha_complement",
        "after_alpha_complement",
    ]
    index_sheets = set(sheet_names)
    for sheet_name in sheet_names:
        frame = details.get(sheet_name)
        if isinstance(frame, pd.DataFrame):
            sheets[sheet_name] = frame.copy()
    return write_labeled_excel_workbook(
        dataset_results_dir(dataset) / "weighted_network_distance_details.xlsx",
        sheets,
        index_sheets=index_sheets,
    )


def write_permutation_result(
    dataset: str,
    permutation_summary: pd.DataFrame,
    distribution: pd.DataFrame,
) -> Path:
    """Write standardized permutation summary and null distribution."""
    summary = permutation_summary.copy()
    if "null_sd" in summary.columns and "null_std" not in summary.columns:
        summary = summary.rename(columns={"null_sd": "null_std"})
    return write_excel_workbook(
        dataset_results_dir(dataset) / "permutation_results.xlsx",
        {
            "permutation_summary": summary,
            "distribution": distribution.copy(),
        },
    )


def write_random_network_result(
    dataset: str,
    summary: pd.DataFrame,
    edges: pd.DataFrame | None = None,
) -> Path:
    """Write standardized random-network example results."""
    sheets: dict[str, pd.DataFrame] = {"summary": summary.copy()}
    if edges is not None:
        sheets["edges"] = edges.copy()
    return write_excel_workbook(dataset_results_dir(dataset) / "random_network_summary.xlsx", sheets)


def write_network_metrics_result(
    dataset: str,
    before_var: pd.DataFrame,
    after_var: pd.DataFrame,
    summary: pd.DataFrame | None = None,
) -> Path:
    """Write standardized before/after network metrics."""
    sheets: dict[str, pd.DataFrame] = {
        "before_var": before_var.copy(),
        "after_var": after_var.copy(),
    }
    if summary is not None:
        sheets["summary"] = summary.copy()
    return write_excel_workbook(dataset_results_dir(dataset) / "network_metrics.xlsx", sheets)


def standard_result_workbook(filename: str, sheets: dict[str, pd.DataFrame], subdir: str | None = None) -> Path:
    """Compatibility wrapper for standardized dataset result workbooks."""
    if subdir is None:
        raise ValueError("Figure2 result workbooks must be written under a dataset subdirectory.")
    return write_excel_workbook(dataset_results_dir(subdir) / filename, sheets)
