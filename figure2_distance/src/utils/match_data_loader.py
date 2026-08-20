from __future__ import annotations

import pandas as pd

from utils.feature_config import FEATURES, assert_no_retired_rank_features
from utils.figure2_paths import match_data_root


DATASET_FILES = {
    "ligue1": "ligue1_matches.xlsx",
    "euro": "euro_matches.xlsx",
    "worldcup": "worldcup_matches.xlsx",
}

REQUIRED_COLUMNS = [
    "home_team",
    "away_team",
    *FEATURES,
    "VAR",
]


def load_match_level_data(dataset: str) -> pd.DataFrame:
    """Load standardized match-level data for Figure2 random/permutation analysis."""
    if dataset not in DATASET_FILES:
        raise KeyError(f"Unknown Figure2 dataset: {dataset}")
    path = match_data_root() / DATASET_FILES[dataset]
    if not path.exists():
        raise FileNotFoundError(f"Required match-level data file not found: {path}")

    frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required column(s): {missing}")

    assert_no_retired_rank_features(list(frame.columns), f"{dataset} match-level data")
    var_values = pd.to_numeric(frame["VAR"], errors="coerce")
    unique_var = set(var_values.dropna().astype(int).tolist())
    if var_values.isna().any() or not unique_var.issubset({0, 1}):
        raise ValueError(f"{path} has invalid VAR values; expected only 0/1.")

    output = frame.copy()
    output["VAR"] = var_values.astype(int)
    output["var"] = output["VAR"]
    return output


def clean_indicator_var_data(match_data: pd.DataFrame, nodes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    """Coerce nodes and VAR together, drop incomplete rows, and return aligned values/labels."""
    var_column = "var" if "var" in match_data.columns else "VAR" if "VAR" in match_data.columns else None
    if var_column is None:
        raise ValueError("Figure2 analysis requires a VAR/var column in match-level data.")
    missing = [node for node in nodes if node not in match_data.columns]
    if missing:
        raise ValueError(f"Match-level data missing node column(s): {missing}")

    cleaned = match_data[[*nodes, var_column]].copy()
    for column in [*nodes, var_column]:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    cleaned = cleaned.dropna(subset=[*nodes, var_column]).copy()
    cleaned[var_column] = cleaned[var_column].astype(int)
    invalid = sorted(set(cleaned[var_column].tolist()) - {0, 1})
    if invalid:
        raise ValueError(f"Figure2 analysis found invalid VAR label(s): {invalid}")

    values = cleaned[nodes].astype(float)
    labels = cleaned[var_column].astype(int)
    group0_size = int(labels.eq(0).sum())
    group1_size = int(labels.eq(1).sum())
    assert len(labels) == len(values)
    assert group0_size + group1_size == len(values)
    return values, labels
