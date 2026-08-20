# -*- coding: utf-8 -*-
"""Assemble manuscript Figure 2 from existing panel images.

This is a layout-only script. It reads PNG panel files from ``figure2/panels``
and writes ``figure2_main.png`` plus ``figure2_main.eps``. It does not compute
network metrics, random networks, permutation distributions, or weighted
network distance statistics.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


FIGURE2_ROOT = Path(__file__).resolve().parents[1]
PANELS_DIR = FIGURE2_ROOT / "panels"
MAIN_PNG = FIGURE2_ROOT / "figure2_main.png"
MAIN_EPS = FIGURE2_ROOT / "figure2_main.eps"


@dataclass(frozen=True)
class PanelSpec:
    label: str
    title: str
    filename: str


PANEL_SPECS = [
    PanelSpec("A", "Ligue 1 before VAR", "ligue1_before_real_network.png"),
    PanelSpec("B", "Ligue 1 after VAR", "ligue1_after_real_network.png"),
    PanelSpec("C", "Ligue 1 random split", "ligue1_random_network_pair.png"),
    PanelSpec("D", "Ligue 1 WD distribution", "ligue1_permutation_weighted_distance.png"),
    PanelSpec("E", "EURO before VAR", "euro_before_real_network.png"),
    PanelSpec("F", "EURO after VAR", "euro_after_real_network.png"),
    PanelSpec("G", "EURO random split", "euro_random_network_pair.png"),
    PanelSpec("H", "EURO WD distribution", "euro_permutation_weighted_distance.png"),
    PanelSpec("I", "World Cup before VAR", "worldcup_before_real_network.png"),
    PanelSpec("J", "World Cup after VAR", "worldcup_after_real_network.png"),
    PanelSpec("K", "World Cup random split", "worldcup_random_network_pair.png"),
    PanelSpec("L", "World Cup WD distribution", "worldcup_permutation_weighted_distance.png"),
]


def _panel_path(spec: PanelSpec) -> Path:
    path = PANELS_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(f"Figure2 panel image not found: {path}")
    return path


def draw_panel(ax, spec: PanelSpec) -> None:
    """Draw one saved panel image with Figure1-style label and title."""
    image = mpimg.imread(_panel_path(spec))
    ax.imshow(image)
    ax.set_title(spec.title, fontsize=16, fontweight="bold", pad=8)
    ax.text(
        -0.055,
        1.035,
        spec.label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=22,
        fontweight="bold",
        color="black",
    )
    ax.axis("off")


def save_main_figure() -> tuple[Path, Path]:
    """Save the combined Figure2 manuscript panel layout."""
    fig = plt.figure(figsize=(22.0, 13.8))
    grid = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.18, 1.18, 2.05, 2.25],
        wspace=0.16,
        hspace=0.22,
    )
    for index, spec in enumerate(PANEL_SPECS):
        row, column = divmod(index, 4)
        ax = fig.add_subplot(grid[row, column])
        draw_panel(ax, spec)
    fig.savefig(MAIN_PNG, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(MAIN_EPS, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return MAIN_PNG, MAIN_EPS


def main() -> None:
    outputs = save_main_figure()
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
