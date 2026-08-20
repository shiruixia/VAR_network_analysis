from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
SRC_DIR = CURRENT_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.excel_utils import write_team_indicator_changes
from utils.figure4_paths import match_data_path, results_dir
from utils.table1_loader import load_significant_indicators
from utils.team_changes import calculate_team_changes


DATASET = "euro"


def run_pipeline() -> Path:
    """Run the Figure4 team-change pipeline for one dataset."""

    significant = load_significant_indicators(DATASET)
    significant = significant[["Indicator", "p-value", "Effect size"]].copy()
    indicators = significant["Indicator"].astype(str).tolist()

    data_path = match_data_path(DATASET)
    if not data_path.exists():
        raise FileNotFoundError(f"Standardized match data file not found: {data_path}")
    match_data = pd.read_excel(data_path)

    team_changes, _team_changes_long, common_teams_audit = calculate_team_changes(
        match_data,
        indicators,
        DATASET,
    )
    common_teams = (
        common_teams_audit.loc[
            common_teams_audit["included"].eq("yes"),
            ["team", "n_before", "n_after"],
        ]
        .rename(columns={"n_before": "before_count", "n_after": "after_count"})
        .reset_index(drop=True)
    )

    output_dir = results_dir(DATASET)
    output_path = output_dir / "team_indicator_changes.xlsx"
    write_team_indicator_changes(
        output_path=output_path,
        significant_indicators=significant,
        team_changes=team_changes,
        common_teams=common_teams,
    )
    return output_path


def main() -> None:
    output = run_pipeline()
    print(output)


if __name__ == "__main__":
    main()
