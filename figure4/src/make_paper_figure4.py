# -*- coding: utf-8 -*-
"""Create manuscript Figure4 split outputs from existing Figure4 results only.

This script is visualization-only. It reads only
``figure4/results/{dataset}/team_indicator_changes.xlsx`` and does not read
match-level data, read Table1 files, or recompute statistical tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import string
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils.figure4_paths import figure4_root, panels_dir, results_dir
from utils.plotting_utils import draw_barplot, draw_heatmap


DATASETS = ["ligue1", "euro", "worldcup"]
DATASET_LABELS = {
    "ligue1": "Ligue 1",
    "euro": "UEFA Euro",
    "worldcup": "World Cup",
}
INDICATOR_LABELS = {
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
INDICATOR_ORDER = [
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


@dataclass(frozen=True)
class PanelOutput:
    label: str
    title: str
    path: Path
    kind: str
    dataset: str


def result_workbook(dataset: str) -> Path:
    """Return the Figure4 dataset result workbook."""

    path = results_dir(dataset) / "team_indicator_changes.xlsx"
    if not path.exists():
        raise FileNotFoundError(f"Figure4 result workbook not found: {path}")
    return path


def load_dataset_result(dataset: str) -> dict[str, pd.DataFrame]:
    """Load Figure4 result sheets for one dataset."""

    workbook = result_workbook(dataset)
    return {
        "significant_indicators": pd.read_excel(workbook, sheet_name="significant_indicators"),
        "team_changes": pd.read_excel(workbook, sheet_name="team_changes"),
        "common_teams": pd.read_excel(workbook, sheet_name="common_teams"),
    }


def indicator_label(indicator: str) -> str:
    """Return paper label for an indicator column."""

    return INDICATOR_LABELS.get(str(indicator), str(indicator).replace("_", " ").title())


def label_sequence() -> list[str]:
    """Return enough panel labels for dynamic Figure4 panels."""

    return list(string.ascii_uppercase)


def ordered_indicator_columns(matrix: pd.DataFrame) -> list[str]:
    """Return team-change indicator columns in the shared paper order."""

    available = [column for column in matrix.columns if column != "team"]
    ordered = [indicator for indicator in INDICATOR_ORDER if indicator in available]
    ordered.extend(column for column in available if column not in ordered)
    return ordered


def global_heatmap_columns(dataset_results: dict[str, dict[str, pd.DataFrame]]) -> list[str]:
    """Return the union of heatmap indicators using one shared order."""

    present: set[str] = set()
    for result in dataset_results.values():
        present.update(ordered_indicator_columns(result["team_changes"]))
    columns = [indicator for indicator in INDICATOR_ORDER if indicator in present]
    columns.extend(sorted(column for column in present if column not in columns))
    return columns


def global_heatmap_vmax(dataset_results: dict[str, dict[str, pd.DataFrame]]) -> float:
    """Calculate a shared heatmap color range from saved team-change matrices."""

    max_abs = 0.0
    for result in dataset_results.values():
        matrix = result["team_changes"]
        indicators = ordered_indicator_columns(matrix)
        if indicators:
            values = matrix[indicators].astype(float).to_numpy()
            if values.size:
                max_abs = max(max_abs, float(pd.Series(values.ravel()).abs().max()))
    return max_abs if max_abs > 0 else 1.0


def save_split_figure(fig: plt.Figure, png_path: str | Path) -> tuple[Path, Path, Path]:
    """Save a split Figure4 manuscript figure as PNG, PDF, and EPS."""

    path = Path(png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    pdf_path = path.with_suffix(".pdf")
    fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight", facecolor="white")
    eps_path = path.with_suffix(".eps")
    fig.savefig(eps_path, format="eps", dpi=300, facecolor="white")
    return path, pdf_path, eps_path


def generate_heatmap_panels(
    dataset_results: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    output_dir: Path,
) -> list[PanelOutput]:
    """Generate one heatmap panel per dataset."""

    outputs: list[PanelOutput] = []
    vmax = global_heatmap_vmax(dataset_results)
    for index, dataset in enumerate(DATASETS):
        label = labels[index]
        matrix = dataset_results[dataset]["team_changes"]
        title = f"{DATASET_LABELS[dataset]} team changes"
        output_path = output_dir / f"figure4_panel_{label}_{dataset}_heatmap.png"
        if not (output_path.exists() and output_path.with_suffix(".eps").exists()):
            ordered_matrix = matrix[["team", *ordered_indicator_columns(matrix)]].copy()
            draw_heatmap(ordered_matrix, output_path, title=title, vmax=vmax)
        outputs.append(PanelOutput(label, title, output_path, "heatmap", dataset))
    return outputs


def color_for_team(team: str, team_order: list[str]) -> str:
    """Assign a stable color within one dataset."""

    palette = plt.get_cmap("tab20")
    index = team_order.index(team) if team in team_order else 0
    rgba = palette(index % 20)
    return matplotlib.colors.to_hex(rgba)


def generate_barplot_panels(
    dataset_results: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    output_dir: Path,
    start_index: int,
) -> list[PanelOutput]:
    """Generate one barplot panel for each significant indicator."""

    outputs: list[PanelOutput] = []
    label_index = start_index
    for dataset in DATASETS:
        significant = dataset_results[dataset]["significant_indicators"]
        matrix = dataset_results[dataset]["team_changes"]
        team_order = sorted(matrix["team"].dropna().astype(str).tolist())
        for indicator in significant["Indicator"].astype(str).tolist():
            if indicator not in matrix.columns:
                continue
            label = labels[label_index]
            label_index += 1
            plot_data = matrix[["team", indicator]].rename(columns={indicator: "difference"}).copy()
            plot_data["color_hex"] = [
                color_for_team(str(team), team_order) for team in plot_data["team"].astype(str)
            ]
            title = f"{DATASET_LABELS[dataset]}: {indicator_label(indicator)}"
            output_path = output_dir / f"figure4_panel_{label}_{dataset}_{indicator}_bar.png"
            if not (output_path.exists() and output_path.with_suffix(".eps").exists()):
                draw_barplot(plot_data, output_path, title=title)
            outputs.append(PanelOutput(label, title, output_path, "barplot", dataset))
    return outputs


def heatmap_values_for_axis(matrix: pd.DataFrame, columns: list[str]) -> tuple[list[str], np.ndarray]:
    """Return teams and values aligned to the shared heatmap columns."""

    teams = matrix["team"].astype(str).tolist()
    aligned = matrix.set_index("team").reindex(columns=columns)
    return teams, aligned.astype(float).to_numpy()


def draw_heatmap_axis(
    ax,
    matrix: pd.DataFrame,
    columns: list[str],
    title: str,
    vmax: float,
):
    """Draw one vector heatmap axis for the split heatmap manuscript figure."""

    teams, values = heatmap_values_for_axis(matrix, columns)
    masked_values = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#F2F2F2")
    image = ax.pcolormesh(
        np.arange(len(columns) + 1),
        np.arange(len(teams) + 1),
        masked_values,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        shading="flat",
        edgecolors="white",
        linewidth=0.25,
    )
    ax.set_facecolor("#F2F2F2")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_xticks(np.arange(len(columns)) + 0.5)
    ax.set_xticklabels([indicator_label(column) for column in columns], rotation=38, ha="right", fontsize=9)
    ax.set_yticks(np.arange(len(teams)) + 0.5)
    ax.set_yticklabels(teams, fontsize=8)
    ax.set_xlim(0, len(columns))
    ax.set_ylim(len(teams), 0)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def compose_heatmap_figure(dataset_results: dict[str, dict[str, pd.DataFrame]]) -> Path:
    """Compose the three dataset heatmaps as a standalone manuscript figure."""

    columns = global_heatmap_columns(dataset_results)
    vmax = global_heatmap_vmax(dataset_results)
    max_teams = max(len(result["team_changes"]) for result in dataset_results.values())
    fig_height = max(7.2, 0.34 * max_teams + 2.1)
    fig, axes = plt.subplots(1, 3, figsize=(22, fig_height), constrained_layout=False)
    image = None
    labels = ["A", "B", "C"]
    titles = {
        "ligue1": "Ligue1 team-level changes",
        "euro": "UEFA Euro team-level changes",
        "worldcup": "World Cup team-level changes",
    }
    for ax, dataset, label in zip(axes, DATASETS, labels):
        image = draw_heatmap_axis(
            ax,
            dataset_results[dataset]["team_changes"],
            columns,
            titles[dataset],
            vmax,
        )
        ax.text(
            -0.08,
            1.03,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=22,
            fontweight="bold",
            color="black",
        )

    fig.subplots_adjust(left=0.045, right=0.9, top=0.92, bottom=0.18, wspace=0.25)
    cbar_ax = fig.add_axes([0.925, 0.25, 0.014, 0.58])
    cbar = fig.colorbar(image, cax=cbar_ax)
    cbar.set_label("Mean after VAR - mean before VAR", fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    output = figure4_root() / "figure4_heatmaps.png"
    _, _, _ = save_split_figure(fig, output)
    plt.close(fig)
    return output


def barplot_data_for_indicator(
    dataset_results: dict[str, dict[str, pd.DataFrame]],
    dataset: str,
    indicator: str,
) -> pd.DataFrame:
    """Return sorted team-change data for one dataset indicator."""

    matrix = dataset_results[dataset]["team_changes"]
    team_order = sorted(matrix["team"].dropna().astype(str).tolist())
    plot_data = matrix[["team", indicator]].rename(columns={indicator: "difference"}).copy()
    plot_data["color_hex"] = [
        color_for_team(str(team), team_order) for team in plot_data["team"].astype(str)
    ]
    return plot_data.sort_values("difference").reset_index(drop=True)


def draw_barplot_axis(ax, plot_data: pd.DataFrame, title: str, label: str) -> None:
    """Draw one barplot axis for the split team-change manuscript figure."""

    x = np.arange(len(plot_data))
    colors = plot_data["color_hex"].tolist() if "color_hex" in plot_data.columns else ["#999999"] * len(plot_data)
    ax.bar(x, plot_data["difference"], color=colors, width=0.82, edgecolor="none")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_title(title, fontsize=12.5, fontweight="bold", pad=7)
    ax.set_ylabel("Mean after VAR - mean before VAR", fontsize=9.5)
    ax.set_xlabel("Teams sorted by difference", fontsize=9.5)
    ax.set_xticks([])
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.text(
        0.018,
        0.982,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=18,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 0.18},
    )


def team_change_specs(
    dataset_results: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    start_index: int,
) -> list[tuple[str, str, str]]:
    """Return label, dataset, and indicator triples for team-change panels."""

    specs: list[tuple[str, str, str]] = []
    label_index = start_index
    for dataset in DATASETS:
        significant = dataset_results[dataset]["significant_indicators"]
        matrix = dataset_results[dataset]["team_changes"]
        for indicator in significant["Indicator"].astype(str).tolist():
            if indicator not in matrix.columns:
                continue
            specs.append((labels[label_index], dataset, indicator))
            label_index += 1
    return specs


def compose_team_changes_figure(
    dataset_results: dict[str, dict[str, pd.DataFrame]],
    labels: list[str],
    start_index: int,
) -> Path:
    """Compose the team-change barplots as a standalone manuscript figure."""

    specs = team_change_specs(dataset_results, labels, start_index)
    columns = 4
    rows = (len(specs) + columns - 1) // columns
    fig = plt.figure(figsize=(22, 4.6 * max(rows, 1)))
    grid = fig.add_gridspec(rows, columns, wspace=0.26, hspace=0.36)

    for index, (label, dataset, indicator) in enumerate(specs):
        row = index // columns
        col = index % columns
        ax = fig.add_subplot(grid[row, col])
        plot_data = barplot_data_for_indicator(dataset_results, dataset, indicator)
        title = f"{DATASET_LABELS[dataset]}: {indicator_label(indicator)}"
        draw_barplot_axis(ax, plot_data, title, label)

    empty_slots = rows * columns - len(specs)
    if empty_slots:
        start = len(specs)
        for index in range(start, start + empty_slots):
            row = index // columns
            col = index % columns
            ax = fig.add_subplot(grid[row, col])
            ax.axis("off")

    fig.subplots_adjust(left=0.035, right=0.99, top=0.975, bottom=0.045)
    output = figure4_root() / "figure4_team_changes.png"
    _, _, _ = save_split_figure(fig, output)
    plt.close(fig)
    return output


def ensure_eps_for_existing_panel_pngs() -> None:
    """Create EPS sidecars for any existing Figure4 panel PNGs without EPS."""

    for png_path in sorted(panels_dir().glob("figure4_panel_*.png")):
        eps_path = png_path.with_suffix(".eps")
        if eps_path.exists():
            continue
        image = mpimg.imread(png_path)
        height, width = image.shape[:2]
        fig = plt.figure(figsize=(width / 300, height / 300), dpi=300)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.imshow(image)
        ax.axis("off")
        fig.savefig(eps_path, format="eps", dpi=300, bbox_inches="tight", pad_inches=0, facecolor="white")
        plt.close(fig)


def main() -> list[Path]:
    """Generate Figure4 panels and the split manuscript figures from results only."""

    output_dir = panels_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_results = {dataset: load_dataset_result(dataset) for dataset in DATASETS}
    labels = label_sequence()
    heatmap_panels = generate_heatmap_panels(dataset_results, labels, output_dir)
    barplot_panels = generate_barplot_panels(
        dataset_results,
        labels,
        output_dir,
        start_index=0,
    )
    heatmap_output = compose_heatmap_figure(dataset_results)
    team_changes_output = compose_team_changes_figure(
        dataset_results,
        labels,
        start_index=0,
    )
    ensure_eps_for_existing_panel_pngs()

    panel_outputs = [panel.path for panel in [*heatmap_panels, *barplot_panels]]
    outputs = [*panel_outputs]
    outputs.extend(path.with_suffix(".pdf") for path in panel_outputs)
    outputs.extend(path.with_suffix(".eps") for path in panel_outputs)
    outputs.extend(
        [
            heatmap_output,
            heatmap_output.with_suffix(".pdf"),
            heatmap_output.with_suffix(".eps"),
            team_changes_output,
            team_changes_output.with_suffix(".pdf"),
            team_changes_output.with_suffix(".eps"),
        ]
    )
    for output in outputs:
        print(output)
    return outputs


if __name__ == "__main__":
    main()
