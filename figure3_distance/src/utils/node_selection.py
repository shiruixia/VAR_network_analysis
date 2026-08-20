from __future__ import annotations

import pandas as pd

from utils.figure3_paths import get_table1_result_path


INDICATOR_NAME_MAP = {
    "First-half time": "first_half_time",
    "Second-half time": "second_half_time",
    "Total time": "total_time",
    "Goals": "goals",
    "Total goals": "goals",
    "Yellow cards": "yellow_cards",
    "Total yellow cards": "yellow_cards",
    "Red cards": "red_cards",
    "Total red cards": "red_cards",
    "Fouls": "fouls",
    "Total fouls": "fouls",
    "Offsides": "offsides",
    "Total offsides": "offsides",
    "Penalties": "penalties",
    "Total penalties": "penalties",
}

MECHANISM_TIE_ORDER = [
    "total_time",
    "first_half_time",
    "second_half_time",
    "offsides",
    "penalties",
    "goals",
    "yellow_cards",
    "red_cards",
    "fouls",
]

REQUIRED_COLUMNS = ["Indicator", "Z", "p-value", "Effect size"]


def standardize_node_name(value: object) -> str:
    """Map Table1 indicator labels to Figure3 node names."""
    text = str(value).strip()
    if text in INDICATOR_NAME_MAP:
        return INDICATOR_NAME_MAP[text]
    normalized = text.lower().replace("-", "_").replace(" ", "_")
    return INDICATOR_NAME_MAP.get(text.title(), normalized)


def load_table1_statistics(dataset: str) -> pd.DataFrame:
    """Read Table1 statistics used for Figure3 node ranking."""
    path = get_table1_result_path(dataset)
    if not path.exists():
        raise FileNotFoundError(f"Table1 result file not found: {path}")
    frame = pd.read_excel(path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{path} missing required column(s): {missing}")
    return frame.copy()


def build_node_inclusion_order(table1_frame: pd.DataFrame) -> pd.DataFrame:
    """Create effect-size-ranked Figure3 node order from Table1 output."""
    missing = [column for column in REQUIRED_COLUMNS if column not in table1_frame.columns]
    if missing:
        raise ValueError(f"Table1 frame missing required column(s): {missing}")

    frame = table1_frame.copy()
    frame["node"] = frame["Indicator"].map(standardize_node_name)
    frame["Z"] = pd.to_numeric(frame["Z"], errors="coerce")
    frame["p_value"] = pd.to_numeric(frame["p-value"], errors="coerce")
    frame["effect_size"] = pd.to_numeric(frame["Effect size"], errors="coerce")
    frame["abs_Z"] = frame["Z"].abs()
    tie_rank = {node: index for index, node in enumerate(MECHANISM_TIE_ORDER)}
    frame["mechanism_tie_rank"] = frame["node"].map(tie_rank).fillna(len(tie_rank)).astype(int)
    frame = frame.sort_values(
        ["effect_size", "abs_Z", "mechanism_tie_rank"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    frame["rank"] = range(1, len(frame) + 1)
    return frame[["rank", "node", "p_value", "effect_size", "Z"]]


def get_node_inclusion_order(dataset: str) -> pd.DataFrame:
    """Read Table1 and return the standard Figure3 node inclusion order."""
    return build_node_inclusion_order(load_table1_statistics(dataset))


def build_node_inclusion_path(order: pd.DataFrame, min_nodes: int = 2) -> pd.DataFrame:
    """Return cumulative node sets beginning with the first min_nodes nodes."""
    if "node" not in order.columns:
        raise ValueError("Node order must contain a node column.")
    nodes = order.sort_values("rank")["node"].astype(str).tolist() if "rank" in order.columns else order["node"].astype(str).tolist()
    rows = []
    for end_index in range(min_nodes, len(nodes) + 1):
        node_set = nodes[:end_index]
        rows.append(
            {
                "step": end_index - min_nodes + 1,
                "n_nodes": end_index,
                "node_added": node_set[-1],
                "node_set_full": " + ".join(node_set),
            }
        )
    return pd.DataFrame(rows)
