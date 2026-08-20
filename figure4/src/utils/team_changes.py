from __future__ import annotations

import numpy as np
import pandas as pd


BASE_COLUMNS = ["home_team", "away_team", "VAR"]
MATCH_INDICATORS = [
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


def validate_match_data(df: pd.DataFrame, indicators: list[str]) -> None:
    """Validate standardized match-level data required for Figure4."""

    required = [*BASE_COLUMNS, *indicators]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Match data missing required column(s): {missing}")
    var_values = set(pd.Series(df["VAR"]).dropna().astype(int).unique().tolist())
    if not var_values.issubset({0, 1}):
        raise ValueError(f"VAR column must contain only 0/1 values, found: {sorted(var_values)}")


def split_var_groups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split standardized match data into before and after VAR groups."""

    before = df.loc[pd.to_numeric(df["VAR"], errors="coerce").eq(0)].copy()
    after = df.loc[pd.to_numeric(df["VAR"], errors="coerce").eq(1)].copy()
    return before, after


def expand_team_appearances(df: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Create one team-appearance row for each home and away team."""

    validate_match_data(df, indicators)
    frames = []
    for side in ["home_team", "away_team"]:
        part = df[[side, "VAR", *indicators]].copy()
        part = part.rename(columns={side: "team"})
        part["team"] = part["team"].astype(str).str.strip()
        frames.append(part)
    team_rows = pd.concat(frames, ignore_index=True)
    for indicator in indicators:
        team_rows[indicator] = pd.to_numeric(team_rows[indicator], errors="coerce")
    team_rows["VAR"] = pd.to_numeric(team_rows["VAR"], errors="coerce").astype("Int64")
    return team_rows


def find_common_teams(team_rows: pd.DataFrame) -> tuple[list[str], pd.DataFrame]:
    """Find teams appearing before and after VAR, and return an audit table."""

    before_counts = (
        team_rows.loc[team_rows["VAR"].eq(0), "team"]
        .dropna()
        .astype(str)
        .value_counts()
    )
    after_counts = (
        team_rows.loc[team_rows["VAR"].eq(1), "team"]
        .dropna()
        .astype(str)
        .value_counts()
    )
    all_teams = sorted(set(before_counts.index).union(after_counts.index))
    common = sorted(set(before_counts.index).intersection(after_counts.index))
    audit = pd.DataFrame(
        {
            "team": all_teams,
            "before": ["yes" if team in before_counts.index else "no" for team in all_teams],
            "after": ["yes" if team in after_counts.index else "no" for team in all_teams],
            "n_before": [int(before_counts.get(team, 0)) for team in all_teams],
            "n_after": [int(after_counts.get(team, 0)) for team in all_teams],
            "included": ["yes" if team in common else "no" for team in all_teams],
        }
    )
    return common, audit


def calculate_team_level_means(team_rows: pd.DataFrame, indicators: list[str]) -> pd.DataFrame:
    """Calculate team-level means for before and after VAR groups."""

    grouped = (
        team_rows.groupby(["team", "VAR"], dropna=False)[indicators]
        .mean(numeric_only=True)
        .reset_index()
    )
    return grouped


def calculate_team_changes(
    match_df: pd.DataFrame,
    indicators: list[str],
    dataset: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calculate team-by-indicator changes for common teams only."""

    validate_match_data(match_df, indicators)
    team_rows = expand_team_appearances(match_df, indicators)
    common_teams, common_audit = find_common_teams(team_rows)
    included_rows = team_rows.loc[team_rows["team"].isin(common_teams)].copy()

    long_records = []
    for team in common_teams:
        team_data = included_rows.loc[included_rows["team"].eq(team)]
        for indicator in indicators:
            before = team_data.loc[team_data["VAR"].eq(0), indicator].dropna()
            after = team_data.loc[team_data["VAR"].eq(1), indicator].dropna()
            mean_before = float(before.mean()) if len(before) else np.nan
            mean_after = float(after.mean()) if len(after) else np.nan
            difference = mean_after - mean_before if np.isfinite(mean_before) and np.isfinite(mean_after) else np.nan
            long_records.append(
                {
                    "dataset": str(dataset).lower(),
                    "team": team,
                    "indicator": indicator,
                    "n_before": int(len(before)),
                    "n_after": int(len(after)),
                    "mean_before": mean_before,
                    "mean_after": mean_after,
                    "difference": difference,
                }
            )
    long_df = pd.DataFrame(long_records)
    if long_df.empty:
        matrix = pd.DataFrame(columns=["team", *indicators])
    else:
        matrix = long_df.pivot(index="team", columns="indicator", values="difference")
        matrix = matrix.reindex(columns=indicators).reset_index()
    return matrix, long_df, common_audit


def build_barplot_data(team_changes_long: pd.DataFrame, color_map: dict[str, str] | None = None) -> pd.DataFrame:
    """Build sorted barplot data for each significant indicator."""

    if team_changes_long.empty:
        return pd.DataFrame(columns=["dataset", "indicator", "team", "difference", "plot_order", "label_flag", "color_hex"])
    rows = []
    color_map = color_map or {}
    for (_dataset, indicator), group in team_changes_long.groupby(["dataset", "indicator"], sort=False):
        ordered = group.sort_values("difference").reset_index(drop=True).copy()
        ordered["plot_order"] = np.arange(1, len(ordered) + 1)
        label_teams = set(ordered.head(2)["team"].tolist() + ordered.tail(2)["team"].tolist())
        ordered["label_flag"] = ordered["team"].isin(label_teams)
        ordered["color_hex"] = ordered["team"].map(color_map).fillna("#999999")
        rows.append(ordered[["dataset", "indicator", "team", "difference", "plot_order", "label_flag", "color_hex"]])
    return pd.concat(rows, ignore_index=True)


def build_heatmap_matrix(team_changes: pd.DataFrame) -> pd.DataFrame:
    """Sort a team-by-indicator matrix for heatmap plotting."""

    if team_changes.empty or list(team_changes.columns) == ["team"]:
        return team_changes.copy()
    indicator_cols = [column for column in team_changes.columns if column != "team"]
    matrix = team_changes.copy()
    matrix["_abs_mean_change"] = matrix[indicator_cols].abs().mean(axis=1)
    matrix = matrix.sort_values("_abs_mean_change", ascending=False).drop(columns=["_abs_mean_change"])
    return matrix.reset_index(drop=True)
