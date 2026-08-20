from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


HISTOGRAM_COLOR = "#8DB3C7"
OBSERVED_COLOR = "#C4473A"
FIG_AXIS_LABEL_SIZE = 12
FIG_TICK_SIZE = 10
FIG_LEGEND_SIZE = 10


def plot_permutation_weighted_distance(
    distribution: pd.DataFrame,
    observed: float,
    empirical_p: float,
    path: Path,
    title: str | None = None,
) -> Path:
    """Plot the permutation histogram for Weighted Network Distance."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.hist(distribution["weighted_network_distance"], bins=35, color=HISTOGRAM_COLOR, edgecolor="white")
    ax.axvline(observed, color=OBSERVED_COLOR, linewidth=2.5, label=f"Observed WD = {observed:.3f}")
    if title:
        ax.set_title(title)
    ax.set_xlabel("Weighted network distance (WD)", fontsize=FIG_AXIS_LABEL_SIZE)
    ax.set_ylabel("Permutation count", fontsize=FIG_AXIS_LABEL_SIZE)
    ax.tick_params(axis="both", labelsize=FIG_TICK_SIZE)
    ax.legend(frameon=False, fontsize=FIG_LEGEND_SIZE)
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".eps"), format="eps", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
