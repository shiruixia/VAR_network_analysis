from __future__ import annotations

import pandas as pd

from .figure4_paths import table1_result_path


REQUIRED_COLUMNS = ["Indicator", "p-value", "Effect size"]

INDICATOR_NAME_MAP = {
    "first_half_time": "first_half_time",
    "first half time": "first_half_time",
    "first-half time": "first_half_time",
    "second_half_time": "second_half_time",
    "second half time": "second_half_time",
    "second-half time": "second_half_time",
    "total_time": "total_time",
    "total time": "total_time",
    "goals": "goals",
    "total goals": "goals",
    "yellow_cards": "yellow_cards",
    "yellow cards": "yellow_cards",
    "total yellow cards": "yellow_cards",
    "red_cards": "red_cards",
    "red cards": "red_cards",
    "total red cards": "red_cards",
    "fouls": "fouls",
    "total fouls": "fouls",
    "offsides": "offsides",
    "total offsides": "offsides",
    "penalties": "penalties",
    "total penalties": "penalties",
}


def normalize_indicator_name(value: object) -> str:
    """Convert Table1 display labels to standardized match-data column names."""

    key = str(value).strip().replace("_", " ").lower()
    key = " ".join(key.split())
    if key not in INDICATOR_NAME_MAP:
        raise ValueError(f"Unknown Table1 indicator name: {value}")
    return INDICATOR_NAME_MAP[key]


def load_table1_result(dataset: str) -> pd.DataFrame:
    """Read one Table1 workbook without recomputing any statistical test."""

    path = table1_result_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Table1 result file not found: {path}")
    frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Table1 result missing required column(s): {missing}")
    frame = frame.copy()
    frame["dataset"] = str(dataset).lower()
    frame["Indicator"] = frame["Indicator"].map(normalize_indicator_name)
    frame["p-value"] = pd.to_numeric(frame["p-value"], errors="coerce")
    frame["Effect size"] = pd.to_numeric(frame["Effect size"], errors="coerce")
    optional = [column for column in ["Z"] if column in frame.columns]
    return frame[["dataset", "Indicator", "p-value", "Effect size", *optional]]


def load_significant_indicators(dataset: str, alpha: float = 0.05) -> pd.DataFrame:
    """Return Table1 indicators with p-value below alpha."""

    frame = load_table1_result(dataset)
    significant = frame.loc[frame["p-value"].lt(alpha)].copy()
    significant = significant.sort_values("p-value", ascending=True).reset_index(drop=True)
    return significant


def significant_indicator_names(dataset: str, alpha: float = 0.05) -> list[str]:
    """Return significant indicator names from Table1."""

    return load_significant_indicators(dataset, alpha)["Indicator"].astype(str).tolist()
