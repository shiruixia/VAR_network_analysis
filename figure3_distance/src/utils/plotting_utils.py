from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .figure3_paths import get_panel_path, get_results_path


HEADER_COLOR = "#419BBE"
SIGNIFICANT_ROW_COLOR = "#D9EBF2"
CURVE_COLOR = "#419BBE"
GRID_COLOR = "#D9D9D9"
TEXT_COLOR = "#222222"

plt.rcParams.update(
    {
        "font.family": ["DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
    }
)

DISPLAY_LABELS = {
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

CURVE_DISPLAY_NODES = [5, 6, 7, 8, 9]
CURVE_X_LIMITS = (4.65, 9.35)
LABEL_OFFSETS = {
    "ligue1": {
        5: (10, 18),
        6: (22, 28),
        7: (0, 28),
        8: (0, -26),
        9: (-10, 18),
    },
    "euro": {
        5: (10, -24),
        6: (-18, -30),
        7: (0, -34),
        8: (8, 18),
        9: (-10, -24),
    },
    "worldcup": {
        5: (10, 18),
        6: (14, -34),
        7: (-36, 26),
        8: (10, 18),
        9: (-10, 18),
    },
}


@dataclass(frozen=True)
class DatasetFigure3Results:
    """Container for Figure3 result workbooks already produced by pipelines."""

    dataset: str
    node_order: pd.DataFrame
    weighted_distance_curve: pd.DataFrame
    permutation_results: pd.DataFrame
    statistics_summary: pd.DataFrame


@dataclass(frozen=True)
class Figure3PanelSpec:
    """Panel metadata used by the paper-level Figure3 composer."""

    dataset: str
    table_label: str
    curve_label: str
    display_name: str


def display_label(value: object) -> str:
    """Return publication-style labels without changing result data."""

    text = str(value)
    return DISPLAY_LABELS.get(text, text.replace("_", " ").title())


def save_panel(fig: plt.Figure, output_path: str | Path) -> Path:
    """Save a Figure3 panel as PNG, PDF, and EPS."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), format="pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".eps"), format="eps", bbox_inches="tight", facecolor="white")
    return path


def _wrap_label(label: object, width: int = 18) -> str:
    return "\n".join(textwrap.wrap(str(label), width=width, break_long_words=False))


def _validate_columns(frame: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"{label} missing column(s): {missing}")


def load_dataset_results(dataset: str) -> DatasetFigure3Results:
    """Read the four Figure3 result workbooks for one dataset."""

    result_dir = get_results_path(dataset)
    node_order_path = result_dir / "node_inclusion_order.xlsx"
    weighted_distance_path = result_dir / "weighted_network_distance_curve.xlsx"
    permutation_path = result_dir / "permutation_results.xlsx"
    summary_path = result_dir / "statistics_summary.xlsx"

    for path in [node_order_path, weighted_distance_path, permutation_path, summary_path]:
        if not path.exists():
            raise FileNotFoundError(f"Figure3 result file not found: {path}")

    node_order = pd.read_excel(node_order_path, sheet_name="node_order")
    weighted_distance_curve = pd.read_excel(weighted_distance_path, sheet_name="weighted_distance_curve")
    permutation_results = pd.read_excel(permutation_path, sheet_name="permutation_summary")
    statistics_summary = pd.read_excel(summary_path, sheet_name="statistics_summary")

    _validate_columns(node_order, ["rank", "node", "p_value", "effect_size", "Z"], str(node_order_path))
    _validate_columns(
        weighted_distance_curve,
        ["number_of_nodes", "added_node", "weighted_network_distance"],
        str(weighted_distance_path),
    )
    _validate_columns(
        permutation_results,
        ["observed_weighted_network_distance", "null_mean", "null_std", "empirical_p_value"],
        str(permutation_path),
    )
    _validate_columns(
        statistics_summary,
        ["dataset", "n_matches", "n_no_var", "n_with_var", "node_number", "method_description"],
        str(summary_path),
    )

    return DatasetFigure3Results(
        dataset=dataset,
        node_order=node_order,
        weighted_distance_curve=weighted_distance_curve,
        permutation_results=permutation_results,
        statistics_summary=statistics_summary,
    )



def is_significant_pvalue(value: object) -> bool:
    """Return True when a p-value representation indicates p < 0.05."""

    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    has_star = "*" in text
    numeric_text = text.replace("*", "").replace("<", "").strip()
    try:
        numeric = float(numeric_text)
    except ValueError:
        return has_star
    return has_star or numeric < 0.05


def significant_node_count(order: pd.DataFrame) -> int:
    """Count significant nodes in the stored inclusion-order table."""

    if order is None or order.empty:
        return 0
    if "significant" in order.columns:
        return int(order["significant"].astype(bool).sum())
    pvalue_column = None
    for candidate in ["p_value", "p-value", "pvalue", "P_value", "P-value"]:
        if candidate in order.columns:
            pvalue_column = candidate
            break
    if pvalue_column is None:
        return 0
    return int(order[pvalue_column].map(is_significant_pvalue).sum())


def divider_x_from_node_order(order: pd.DataFrame) -> float | None:
    """Return the x-position separating significant and non-significant nodes."""

    if order is None or order.empty:
        return None
    count = significant_node_count(order)
    total_nodes = len(order)
    if count >= 2 and count < total_nodes:
        return float(count) + 0.5
    return None

def _format_p_value(value: object) -> str:
    if pd.isna(value):
        return ""
    numeric = float(value)
    if numeric < 0.001:
        return "<0.001*"
    suffix = "*" if numeric < 0.05 else ""
    return f"{numeric:.3f}{suffix}"


def plot_node_order_table(
    order: pd.DataFrame,
    output_path: str | Path,
    title: str = "Node inclusion order",
) -> Path:
    """Plot a Figure3 node order table without reading raw data."""

    required = ["rank", "node", "p_value", "effect_size"]
    _validate_columns(order, required, "node inclusion order")

    display = order.copy()
    display["node"] = display["node"].map(display_label)
    display["p_value"] = display["p_value"].map(_format_p_value)
    display["effect_size"] = display["effect_size"].map(
        lambda value: f"{float(value):.2f}" if pd.notna(value) else ""
    )
    table_data = display[["rank", "node", "p_value", "effect_size"]].values.tolist()
    headers = ["Rank", "Node", "p-value", "Effect size"]
    colors = [
        [SIGNIFICANT_ROW_COLOR] * len(headers)
        if pd.notna(row.get("p_value")) and float(row.get("p_value")) < 0.05
        else ["white"] * len(headers)
        for _, row in order.iterrows()
    ]

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.axis("off")
    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    table = ax.table(
        cellText=table_data,
        colLabels=headers,
        cellColours=colors,
        cellLoc="center",
        colLoc="center",
        loc="center",
        colWidths=[0.12, 0.40, 0.20, 0.20],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(12.5)
    table.scale(1.0, 1.72)

    for (row_index, _col_index), cell in table.get_celld().items():
        cell.set_edgecolor(GRID_COLOR)
        cell.set_linewidth(0.8)
        if row_index == 0:
            cell.set_facecolor(HEADER_COLOR)
            cell.set_text_props(
                color="white",
                weight="bold",
                ha="center",
                va="center",
                fontfamily="Arial",
            )
        else:
            cell.set_text_props(ha="center", va="center", color=TEXT_COLOR, fontfamily="Arial")

    fig.tight_layout(pad=0.4)
    path = save_panel(fig, output_path)
    plt.close(fig)
    return path


def plot_weighted_distance_curve(
    curve: pd.DataFrame,
    output_path: str | Path,
    title: str = "Weighted network distance trajectory",
    node_order: pd.DataFrame | None = None,
    dataset: str | None = None,
) -> Path:
    """Plot a Figure3 WD trajectory from N=5 while retaining full Excel results."""

    required = ["number_of_nodes", "added_node", "weighted_network_distance"]
    _validate_columns(curve, required, "weighted network distance curve")
    data = curve.sort_values("number_of_nodes").copy()
    # The curve starts at N=5 for visual presentation, while the first four
    # ranked indicators remain included in every displayed network.
    plot_data = data.loc[data["number_of_nodes"].between(5, 9)].copy()
    plotted_nodes = plot_data["number_of_nodes"].astype(int).tolist()
    if plotted_nodes != CURVE_DISPLAY_NODES:
        raise ValueError(f"Figure3 curve panel must plot nodes {CURVE_DISPLAY_NODES}, got {plotted_nodes}")

    fig, ax = plt.subplots(figsize=(8.4, 6.4))
    ax.plot(
        plot_data["number_of_nodes"],
        plot_data["weighted_network_distance"],
        color=CURVE_COLOR,
        marker="o",
        markerfacecolor=CURVE_COLOR,
        markeredgecolor=CURVE_COLOR,
        linewidth=2.4,
        markersize=7.5,
    )
    divider_x = divider_x_from_node_order(node_order)
    if divider_x is not None and CURVE_DISPLAY_NODES[0] <= divider_x <= CURVE_DISPLAY_NODES[-1]:
        ax.axvline(
            divider_x,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=1.0,
            zorder=1,
        )

    ax.set_title(title, fontsize=17, fontweight="bold", pad=10)
    ax.set_xlabel("Number of nodes", fontsize=13)
    ax.set_ylabel("Weighted network distance (WD)", fontsize=13)
    ax.set_xticks(CURVE_DISPLAY_NODES)
    ax.set_xlim(*CURVE_X_LIMITS)
    ax.grid(True, axis="y", color=GRID_COLOR, linewidth=0.8, alpha=1.0)
    ax.grid(False, axis="x")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_values = plot_data["weighted_network_distance"].astype(float).to_numpy()
    y_span = max(float(np.max(y_values) - np.min(y_values)), 0.01)
    offsets = LABEL_OFFSETS.get(str(dataset), {})
    min_x = float(plot_data["number_of_nodes"].min())
    max_x = float(plot_data["number_of_nodes"].max())
    for row in plot_data.itertuples(index=False):
        x_value = int(getattr(row, "number_of_nodes"))
        dx, dy = offsets.get(x_value, (0, 18))
        va = "bottom" if dy >= 0 else "top"
        ha = "center"
        if x_value <= min_x:
            ha = "left"
        elif x_value >= max_x:
            ha = "right"
        label = f"+ {display_label(getattr(row, 'added_node'))}"
        ax.annotate(
            _wrap_label(label, width=18),
            xy=(getattr(row, "number_of_nodes"), getattr(row, "weighted_network_distance")),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=10.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 0.20},
        )

    ax.set_ylim(0.0, 0.3)
    fig.tight_layout()
    path = save_panel(fig, output_path)
    plt.close(fig)
    return path


def generate_dataset_panels(spec: Figure3PanelSpec, output_dir: str | Path | None = None) -> list[Path]:
    """Generate the node-order and WD trajectory panels for one dataset."""

    results = load_dataset_results(spec.dataset)
    panel_dir = Path(output_dir) if output_dir is not None else get_panel_path()
    panel_dir.mkdir(parents=True, exist_ok=True)

    table_path = plot_node_order_table(
        results.node_order,
        panel_dir / f"figure3_panel_{spec.table_label}_{spec.dataset}.png",
        f"{spec.display_name} node inclusion order",
    )
    curve_path = plot_weighted_distance_curve(
        results.weighted_distance_curve,
        panel_dir / f"figure3_panel_{spec.curve_label}_{spec.dataset}.png",
        spec.display_name,
        node_order=results.node_order,
        dataset=spec.dataset,
    )
    return [table_path, curve_path]


def compose_main_figure(panel_paths: list[str | Path], output_path: str | Path) -> Path:
    """Compose six saved Figure3 panels into the manuscript Figure3 image."""

    labels = ["A", "B", "C", "D", "E", "F"]
    if len(panel_paths) != len(labels):
        raise ValueError(f"Figure3 main figure requires six panels, got {len(panel_paths)}")

    fig, axes = plt.subplots(
        3,
        2,
        figsize=(15.8, 18.2),
        gridspec_kw={"width_ratios": [1.0, 1.55]},
    )

    for ax, panel_path, label in zip(axes.flat, panel_paths, labels):
        image = plt.imread(Path(panel_path))
        ax.imshow(image)
        ax.axis("off")
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=28,
            fontweight="bold",
            color="black",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 1.0, "pad": 0.22},
        )

    fig.subplots_adjust(left=0.015, right=0.985, top=0.99, bottom=0.01, wspace=0.02, hspace=0.04)
    path = save_panel(fig, output_path)
    plt.close(fig)
    return path
