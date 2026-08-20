from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from utils.feature_config import FEATURES, assert_no_retired_rank_features, canonical_feature
from utils.figure2_paths import figure1_result_root


DISPLAY_NAMES = {
    "ligue1": "Ligue1",
    "euro": "EURO",
    "worldcup": "WorldCup",
}

RESULT_WORKBOOKS = {
    "correlation": "correlation_matrix.xlsx",
    "p_value": "p_value_matrix.xlsx",
    "fdr": "fdr_matrix.xlsx",
    "adjacency": "adjacency_matrix.xlsx",
    "edge_list": "edge_list.xlsx",
    "network_metrics": "network_metrics.xlsx",
}


@dataclass(frozen=True)
class LoadedNetworkResult:
    dataset_key: str
    display_name: str
    nodes: list[str]
    before_correlation_matrix: pd.DataFrame
    after_correlation_matrix: pd.DataFrame
    before_p_value_matrix: pd.DataFrame
    after_p_value_matrix: pd.DataFrame
    before_fdr_matrix: pd.DataFrame
    after_fdr_matrix: pd.DataFrame
    before_adjacency_matrix: pd.DataFrame
    after_adjacency_matrix: pd.DataFrame
    before_edge_list: pd.DataFrame
    after_edge_list: pd.DataFrame
    before_network_metrics: pd.DataFrame
    after_network_metrics: pd.DataFrame
    source_dir: Path
    indicator_data: pd.DataFrame | None = None


def _require_workbook(source_dir: Path, filename: str) -> Path:
    path = source_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Required Figure1 result workbook not found: {path}")
    return path


def _read_matrix_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    frame = pd.read_excel(path, sheet_name=sheet_name)
    if "metric" in frame.columns:
        frame = frame.set_index("metric")
    elif len(frame.columns) > 0 and (str(frame.columns[0]).startswith("Unnamed") or frame.columns[0] in {"index", "node", "node_name"}):
        frame = frame.set_index(frame.columns[0])
    frame.index = frame.index.astype(str)
    frame.columns = frame.columns.astype(str)
    return _canonical_matrix(frame)


def _read_table_workbook(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def _canonical_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    names = list(map(str, frame.index)) + list(map(str, frame.columns))
    assert_no_retired_rank_features(names, "Figure2 network matrix")
    rename_map = {name: feature for name in set(names) if (feature := canonical_feature(name))}
    renamed = frame.rename(index=rename_map, columns=rename_map)
    missing = [feature for feature in FEATURES if feature not in renamed.index or feature not in renamed.columns]
    if missing:
        raise ValueError(f"Figure2 network matrix missing shared feature(s): {missing}")
    return renamed.loc[FEATURES, FEATURES].apply(pd.to_numeric, errors="coerce")


def _canonical_edge_list(edge_list: pd.DataFrame, nodes: list[str]) -> pd.DataFrame:
    if edge_list.empty or not {"source", "target"}.issubset(edge_list.columns):
        return pd.DataFrame(columns=["source", "target", "weight", "spearman_r", "abs_r", "sign", "keep_strict"])
    frame = edge_list.copy()
    assert_no_retired_rank_features(
        frame["source"].astype(str).tolist() + frame["target"].astype(str).tolist(),
        "Figure2 edge list",
    )
    frame["source"] = frame["source"].map(lambda value: canonical_feature(value) or str(value))
    frame["target"] = frame["target"].map(lambda value: canonical_feature(value) or str(value))
    node_set = set(nodes)
    frame = frame.loc[frame["source"].isin(node_set) & frame["target"].isin(node_set)].copy()
    if "spearman_r" not in frame.columns:
        for candidate in ("weight", "correlation", "r"):
            if candidate in frame.columns:
                frame["spearman_r"] = pd.to_numeric(frame[candidate], errors="coerce")
                break
    if "spearman_r" not in frame.columns:
        frame["spearman_r"] = pd.NA
    if "weight" not in frame.columns:
        frame["weight"] = frame["spearman_r"]
    if "abs_r" not in frame.columns:
        frame["abs_r"] = pd.to_numeric(frame["spearman_r"], errors="coerce").abs()
    if "sign" not in frame.columns:
        frame["sign"] = frame["spearman_r"].map(lambda value: "positive" if pd.notna(value) and float(value) >= 0 else "negative")
    if "keep_strict" not in frame.columns:
        frame["keep_strict"] = True
    return frame


def _load_dataset(dataset_key: str) -> LoadedNetworkResult:
    if dataset_key not in DISPLAY_NAMES:
        raise KeyError(f"Unknown Figure2 dataset: {dataset_key}")
    source_dir = figure1_result_root() / dataset_key
    if not source_dir.exists():
        raise FileNotFoundError(f"Figure1 result directory not found: {source_dir}")

    corr_path = _require_workbook(source_dir, RESULT_WORKBOOKS["correlation"])
    p_path = _require_workbook(source_dir, RESULT_WORKBOOKS["p_value"])
    fdr_path = _require_workbook(source_dir, RESULT_WORKBOOKS["fdr"])
    adjacency_path = _require_workbook(source_dir, RESULT_WORKBOOKS["adjacency"])
    edge_path = _require_workbook(source_dir, RESULT_WORKBOOKS["edge_list"])
    metrics_path = _require_workbook(source_dir, RESULT_WORKBOOKS["network_metrics"])

    before_corr = _read_matrix_workbook(corr_path, "before_var")
    after_corr = _read_matrix_workbook(corr_path, "after_var")
    nodes = list(FEATURES)

    return LoadedNetworkResult(
        dataset_key=dataset_key,
        display_name=DISPLAY_NAMES[dataset_key],
        nodes=nodes,
        before_correlation_matrix=before_corr,
        after_correlation_matrix=after_corr,
        before_p_value_matrix=_read_matrix_workbook(p_path, "before_var"),
        after_p_value_matrix=_read_matrix_workbook(p_path, "after_var"),
        before_fdr_matrix=_read_matrix_workbook(fdr_path, "before_var"),
        after_fdr_matrix=_read_matrix_workbook(fdr_path, "after_var"),
        before_adjacency_matrix=_read_matrix_workbook(adjacency_path, "before_var"),
        after_adjacency_matrix=_read_matrix_workbook(adjacency_path, "after_var"),
        before_edge_list=_canonical_edge_list(_read_table_workbook(edge_path, "before_var"), nodes),
        after_edge_list=_canonical_edge_list(_read_table_workbook(edge_path, "after_var"), nodes),
        before_network_metrics=_read_table_workbook(metrics_path, "before_var"),
        after_network_metrics=_read_table_workbook(metrics_path, "after_var"),
        source_dir=source_dir,
    )


def load_ligue1_network_result() -> LoadedNetworkResult:
    return load_network_result("ligue1")


def load_euro_network_result() -> LoadedNetworkResult:
    return load_network_result("euro")


def load_worldcup_network_result() -> LoadedNetworkResult:
    return load_network_result("worldcup")


def load_all_network_results() -> list[LoadedNetworkResult]:
    return [load_network_result(dataset) for dataset in ("ligue1", "euro", "worldcup")]


def load_network_result(dataset: str) -> LoadedNetworkResult:
    """Load one standardized Figure1 network result for Figure2."""
    return _load_dataset(dataset)
