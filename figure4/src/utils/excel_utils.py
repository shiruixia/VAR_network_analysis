from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_team_indicator_changes(
    output_path: str | Path,
    significant_indicators: pd.DataFrame,
    team_changes: pd.DataFrame,
    common_teams: pd.DataFrame,
    team_changes_long: pd.DataFrame | None = None,
    barplot_data: pd.DataFrame | None = None,
    heatmap_matrix: pd.DataFrame | None = None,
    metadata: pd.DataFrame | None = None,
) -> Path:
    """Write standardized Figure4 team-indicator change workbook."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        significant_indicators.to_excel(writer, index=False, sheet_name="significant_indicators")
        team_changes.to_excel(writer, index=False, sheet_name="team_changes")
        common_teams.to_excel(writer, index=False, sheet_name="common_teams")
        if team_changes_long is not None:
            team_changes_long.to_excel(writer, index=False, sheet_name="team_changes_long")
        if barplot_data is not None:
            barplot_data.to_excel(writer, index=False, sheet_name="barplot_data")
        if heatmap_matrix is not None:
            heatmap_matrix.to_excel(writer, index=False, sheet_name="heatmap_matrix")
        if metadata is not None:
            metadata.to_excel(writer, index=False, sheet_name="metadata")
    return path
