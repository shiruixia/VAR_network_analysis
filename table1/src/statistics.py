# -*- coding: utf-8 -*-
"""Statistical helpers for manuscript Table 1.

This module contains all Table 1 statistical logic:

- Mann-Whitney U test
- tie-corrected Z approximation
- eta-squared effect size
- mean, median, and IQR interval summaries
- Excel Table 1 workbook generation
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu


FEATURES = [
    "first_half_time",
    "second_half_time",
    "total_time",
    "goals",
    "yellow_cards",
    "red_cards",
    "fouls",
    "offsides",
    "penalties",
]

TABLE_COLUMNS = [
    "Indicator",
    "No VAR Mean",
    "No VAR Median",
    "No VAR IQR Low",
    "No VAR IQR High",
    "With VAR Mean",
    "With VAR Median",
    "With VAR IQR Low",
    "With VAR IQR High",
    "Z",
    "p-value",
    "Effect size",
]

DISPLAY_NAMES = {
    "first_half_time": "First-half time",
    "second_half_time": "Second-half time",
    "total_time": "Total time",
    "goals": "Goals",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "fouls": "Fouls",
    "offsides": "Offsides",
    "penalties": "Penalties",
}

LEGACY_TO_CANONICAL = {
    "total_goals": "goals",
    "total_yellow_cards": "yellow_cards",
    "total_red_cards": "red_cards",
    "total_fouls": "fouls",
    "total_offsides": "offsides",
    "total_penalties": "penalties",
}


def normalize_indicator_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with legacy total_* indicators exposed as canonical names."""
    normalized = df.copy()
    forbidden = [str(column) for column in normalized.columns if "ranking" in str(column).lower()]
    if forbidden:
        raise ValueError(f"Forbidden ranking variables found in Table1 input: {sorted(forbidden)}")
    for legacy, canonical in LEGACY_TO_CANONICAL.items():
        if canonical not in normalized.columns and legacy in normalized.columns:
            normalized[canonical] = normalized[legacy]
    return normalized


def resolve_var_column(df: pd.DataFrame) -> str:
    """Find the VAR grouping column used by Figure1 indicator data."""
    for candidate in ["var", "VAR"]:
        if candidate in df.columns:
            return candidate
    raise ValueError("Table1 input must contain a var or VAR grouping column")


def summarize_group(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    """Compute n, mean, median, q25, and q75 after dropping missing values."""
    series = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if series.empty:
        return {"n": 0, "mean": np.nan, "median": np.nan, "q25": np.nan, "q75": np.nan}
    return {
        "n": int(series.shape[0]),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "q25": float(series.quantile(0.25)),
        "q75": float(series.quantile(0.75)),
    }


def tie_corrected_z(x: np.ndarray, y: np.ndarray, u_statistic: float) -> float:
    """Compute the tie-corrected normal Z approximation for Mann-Whitney U."""
    n1, n2 = len(x), len(y)
    combined = np.concatenate([x, y])
    _, counts = np.unique(combined, return_counts=True)
    n = n1 + n2
    tie_term = np.sum(counts**3 - counts)
    variance = n1 * n2 / 12 * ((n + 1) - tie_term / max(n * (n - 1), 1))
    if variance <= 0:
        return 0.0
    mean_u = n1 * n2 / 2
    return float((u_statistic - mean_u) / math.sqrt(variance))


def mann_whitney_test(no_var: pd.Series | np.ndarray, with_var: pd.Series | np.ndarray) -> dict[str, float]:
    """Run a two-sided asymptotic Mann-Whitney U test and return U, Z, and p."""
    x = pd.to_numeric(pd.Series(no_var), errors="coerce").dropna().to_numpy(dtype=float)
    y = pd.to_numeric(pd.Series(with_var), errors="coerce").dropna().to_numpy(dtype=float)
    if len(x) == 0 or len(y) == 0:
        raise ValueError("Mann-Whitney U test requires non-empty No VAR and With VAR groups")
    test = mannwhitneyu(x, y, alternative="two-sided", method="asymptotic")
    z_raw = tie_corrected_z(x, y, float(test.statistic))
    return {
        "U statistic": float(test.statistic),
        "Z": float(-z_raw),
        "p-value": float(test.pvalue),
        "N No VAR": int(len(x)),
        "N With VAR": int(len(y)),
        "N Total": int(len(x) + len(y)),
    }


def calculate_effect_size(z_score: float, total_n: int) -> float:
    """Calculate eta squared using eta_squared = Z^2 / (N - 1)."""
    return float(float(z_score) ** 2 / max(int(total_n) - 1, 1))


def generate_table1(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate a Table 1 dataframe and a per-indicator audit dataframe."""
    normalized = normalize_indicator_columns(df)
    var_column = resolve_var_column(normalized)
    missing_features = [feature for feature in FEATURES if feature not in normalized.columns]
    if missing_features:
        raise ValueError(f"Table1 input is missing required indicators: {missing_features}")

    group_values = pd.to_numeric(normalized[var_column], errors="coerce")
    if not set(group_values.dropna().unique()).issubset({0, 1}):
        raise ValueError(f"VAR column must contain only 0/1 values: {var_column}")

    table_rows: list[dict[str, float | str]] = []
    audit_rows: list[dict[str, float | int | str]] = []
    for feature in FEATURES:
        numeric = pd.to_numeric(normalized[feature], errors="coerce")
        no_var = numeric.loc[group_values.eq(0)]
        with_var = numeric.loc[group_values.eq(1)]
        no_var_summary = summarize_group(no_var)
        with_var_summary = summarize_group(with_var)
        test = mann_whitney_test(no_var, with_var)
        effect_size = calculate_effect_size(test["Z"], test["N Total"])

        label = DISPLAY_NAMES[feature]
        table_rows.append(
            {
                "Indicator": label,
                "No VAR Mean": no_var_summary["mean"],
                "No VAR Median": no_var_summary["median"],
                "No VAR IQR Low": no_var_summary["q25"],
                "No VAR IQR High": no_var_summary["q75"],
                "With VAR Mean": with_var_summary["mean"],
                "With VAR Median": with_var_summary["median"],
                "With VAR IQR Low": with_var_summary["q25"],
                "With VAR IQR High": with_var_summary["q75"],
                "Z": test["Z"],
                "p-value": test["p-value"],
                "Effect size": effect_size,
            }
        )
        audit_rows.append(
            {
                "Indicator": label,
                "source_variable": feature,
                "N No VAR": test["N No VAR"],
                "N With VAR": test["N With VAR"],
                "N Total": test["N Total"],
                "Missing No VAR": int(no_var.isna().sum()),
                "Missing With VAR": int(with_var.isna().sum()),
                "Missing Total": int(numeric.isna().sum()),
                "U statistic": test["U statistic"],
            }
        )

    table_df = pd.DataFrame(table_rows, columns=TABLE_COLUMNS)
    table_df = table_df.sort_values(by="p-value", ascending=True).reset_index(drop=True)
    return table_df, pd.DataFrame(audit_rows)


def write_table1_workbook(
    output_path: str | Path,
    table: pd.DataFrame,
    audit: pd.DataFrame | None = None,
    extra_sheets: dict[str, pd.DataFrame] | None = None,
) -> Path:
    """Write a formatted Table 1 workbook."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, index=False, sheet_name="Table1")
        if audit is not None:
            audit.to_excel(writer, index=False, sheet_name="Audit")
        for sheet_name, frame in (extra_sheets or {}).items():
            frame.to_excel(writer, index=False, sheet_name=sheet_name[:31])

        workbook = writer.book
        for worksheet in workbook.worksheets:
            worksheet.freeze_panes = "A2"
            for cell in worksheet[1]:
                cell.font = cell.font.copy(bold=True)
            for column_cells in worksheet.columns:
                header = str(column_cells[0].value or "")
                max_length = max(len(str(cell.value)) if cell.value is not None else 0 for cell in column_cells)
                worksheet.column_dimensions[column_cells[0].column_letter].width = min(max(max_length + 2, 12), 22)
                if header in {"p-value", "Effect size", "Z", "U statistic"}:
                    for cell in column_cells[1:]:
                        cell.number_format = "0.000"
                elif "Mean" in header or "Median" in header or "IQR" in header:
                    for cell in column_cells[1:]:
                        cell.number_format = "0.00"
    return path
