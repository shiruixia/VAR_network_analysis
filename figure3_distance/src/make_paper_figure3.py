# -*- coding: utf-8 -*-
"""Create manuscript Figure3 from existing Figure3 pipeline results.

This script is visualization-only. It reads only figure3/results/{dataset}
workbooks and does not rerun node ranking, WD, permutation, or effect-size
calculations.
"""

from __future__ import annotations

from pathlib import Path
import sys


CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from utils.figure3_paths import get_figure3_root, get_panel_path
from utils.plotting_utils import Figure3PanelSpec, compose_main_figure, generate_dataset_panels


PANEL_SPECS = [
    Figure3PanelSpec("ligue1", "A", "B", "Ligue 1"),
    Figure3PanelSpec("euro", "C", "D", "UEFA European Championship"),
    Figure3PanelSpec("worldcup", "E", "F", "FIFA World Cup"),
]


def main() -> list[Path]:
    """Generate Figure3 panels and the final manuscript Figure3."""

    panel_dir = get_panel_path()
    panel_dir.mkdir(parents=True, exist_ok=True)

    panel_paths: list[Path] = []
    for spec in PANEL_SPECS:
        panel_paths.extend(generate_dataset_panels(spec, panel_dir))

    main_figure = compose_main_figure(panel_paths, get_figure3_root() / "figure3_main.png")
    return [*panel_paths, main_figure]


if __name__ == "__main__":
    for output in main():
        print(output)
