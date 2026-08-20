"""Figure1 standardized-data pipeline helpers."""
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

try:
    from utils.indicator_config import MATCH_INDICATORS
except ModuleNotFoundError:
    from indicator_config import MATCH_INDICATORS


DATASETS = {"ligue1", "euro", "worldcup"}
TEAM_COLUMNS = ["home_team", "away_team"]
VAR_COLUMN = "VAR"
STANDARDIZED_COLUMNS = [*TEAM_COLUMNS, *MATCH_INDICATORS, VAR_COLUMN]


def figure1_root() -> Path:
    """Return the figure1 project root."""
    return Path(__file__).resolve().parents[2]


def paper_assets_root() -> Path:
    """Return the paper_assets root."""
    return figure1_root().parent


def standardized_data_dir() -> Path:
    """Return the standardized paper_assets/data directory."""
    return paper_assets_root() / "data"


def add_src_to_path(current_file: str | Path) -> Path:
    """Add figure1/src to sys.path and return it."""
    src_dir = Path(current_file).resolve().parents[1]
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    return src_dir


def load_standardized_match_file(dataset: str) -> pd.DataFrame:
    """Load one standardized match-level workbook from paper_assets/data."""
    dataset_key = str(dataset).lower()
    if dataset_key not in DATASETS:
        raise ValueError(f"Unknown Figure1 dataset: {dataset}")
    path = standardized_data_dir() / f"{dataset_key}_matches.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Standardized Figure1 match file not found: {path}")
    frame = pd.read_excel(path)
    return validate_standardized_match_data(frame)


def validate_standardized_match_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and return a copy of the standardized Figure1 match data."""
    missing = [column for column in STANDARDIZED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Standardized Figure1 match data missing columns: {missing}")
    output = df.loc[:, STANDARDIZED_COLUMNS].copy()
    if output[STANDARDIZED_COLUMNS].isna().any().any():
        missing_counts = output.isna().sum()
        missing_counts = missing_counts.loc[missing_counts.gt(0)].to_dict()
        raise ValueError(f"Standardized Figure1 match data contains missing values: {missing_counts}")
    for column in TEAM_COLUMNS:
        output[column] = output[column].astype(str)
    for indicator in MATCH_INDICATORS:
        output[indicator] = pd.to_numeric(output[indicator], errors="raise")
    output[VAR_COLUMN] = pd.to_numeric(output[VAR_COLUMN], errors="raise").astype(int)
    invalid_var = sorted(set(output[VAR_COLUMN].unique()) - {0, 1})
    if invalid_var:
        raise ValueError(f"VAR column must contain only 0/1 values; found: {invalid_var}")
    return output


def indicator_frame_from_matches(df: pd.DataFrame) -> pd.DataFrame:
    """Extract the VAR column and nine indicators needed for network analysis."""
    validated = validate_standardized_match_data(df)
    return validated.loc[:, [VAR_COLUMN, *MATCH_INDICATORS]].copy()


def split_before_after_var(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return before-VAR and after-VAR dataframes from a VAR-coded dataframe."""
    if VAR_COLUMN not in df.columns:
        raise ValueError("Cannot split data without VAR column")
    var_values = pd.to_numeric(df[VAR_COLUMN], errors="raise").astype(int)
    invalid_var = sorted(set(var_values.unique()) - {0, 1})
    if invalid_var:
        raise ValueError(f"VAR column must contain only 0/1 values; found: {invalid_var}")
    before_var = df.loc[var_values.eq(0)].copy().reset_index(drop=True)
    after_var = df.loc[var_values.eq(1)].copy().reset_index(drop=True)
    return before_var, after_var
