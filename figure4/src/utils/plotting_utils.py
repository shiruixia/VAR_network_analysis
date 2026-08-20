from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_png_eps(fig: plt.Figure, png_path: str | Path) -> tuple[Path, Path]:
    """Save a matplotlib figure as PNG and EPS sidecar."""

    path = Path(png_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    eps_path = path.with_suffix(".eps")
    fig.savefig(eps_path, format="eps", bbox_inches="tight", facecolor="white")
    return path, eps_path


def draw_heatmap(
    matrix: pd.DataFrame,
    output_path: str | Path,
    title: str,
    value_label: str = "Mean after VAR - mean before VAR",
    cmap: str = "RdBu_r",
    vmax: float | None = None,
) -> Path:
    """Draw a team-by-indicator heatmap from a prepared matrix."""

    if "team" not in matrix.columns:
        raise ValueError("Heatmap matrix must contain a 'team' column")
    teams = matrix["team"].astype(str).tolist()
    indicators = [column for column in matrix.columns if column != "team"]
    values = matrix[indicators].astype(float).to_numpy() if indicators else np.empty((len(teams), 0))
    if vmax is None:
        vmax = float(np.nanmax(np.abs(values))) if values.size else 1.0
        vmax = vmax if vmax > 0 else 1.0

    fig_height = max(4.8, 0.32 * max(len(teams), 1) + 1.6)
    fig, ax = plt.subplots(figsize=(7.8, fig_height))
    if indicators:
        image = ax.imshow(values, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(np.arange(len(indicators)))
        ax.set_xticklabels(indicators, rotation=35, ha="right", fontsize=9.5)
        ax.set_yticks(np.arange(len(teams)))
        ax.set_yticklabels(teams, fontsize=8.5)
        cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.025)
        cbar.set_label(value_label, fontsize=10)
        cbar.ax.tick_params(labelsize=8.5)
    else:
        ax.text(0.5, 0.5, "No significant indicators", ha="center", va="center", fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    fig.tight_layout()
    save_png_eps(fig, output_path)
    plt.close(fig)
    return Path(output_path)


def draw_barplot(
    data: pd.DataFrame,
    output_path: str | Path,
    title: str,
    ylabel: str = "Mean after VAR - mean before VAR",
) -> Path:
    """Draw one sorted team-change barplot from prepared barplot data."""

    required = {"team", "difference"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Barplot data missing required column(s): {sorted(missing)}")
    plot_data = data.sort_values("difference").reset_index(drop=True)
    colors = plot_data["color_hex"].tolist() if "color_hex" in plot_data.columns else ["#999999"] * len(plot_data)

    fig, ax = plt.subplots(figsize=(8.2, 5.3))
    x = np.arange(len(plot_data))
    ax.bar(x, plot_data["difference"], color=colors, width=0.82, edgecolor="none")
    ax.axhline(0, color="#333333", linewidth=1.0)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=8)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_xlabel("Teams sorted by difference", fontsize=11)
    ax.set_xticks([])
    ax.grid(True, axis="y", color="#E3E3E3", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    save_png_eps(fig, output_path)
    plt.close(fig)
    return Path(output_path)
