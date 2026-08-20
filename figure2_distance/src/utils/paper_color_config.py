# -*- coding: utf-8 -*-
from __future__ import annotations


indicator_colors = {
    "first_half_time": "#8dd3c7",
    "second_half_time": "#ffffb3",
    "total_time": "#bebada",
    "offsides": "#fdb462",
    "goals": "#b3de69",
    "red_cards": "#fccde5",
    "fouls": "#d9d9d9",
    "yellow_cards": "#bc80bd",
    "penalties": "#ccebc5",
}

indicator_display_names = {
    "first_half_time": "First-half time",
    "second_half_time": "Second-half time",
    "total_time": "Total time",
    "offsides": "Offsides",
    "goals": "Goals",
    "red_cards": "Red cards",
    "fouls": "Fouls",
    "yellow_cards": "Yellow cards",
    "penalties": "Penalties",
}

edge_colors = {
    "positive": "#fb8072",
    "negative": "#80b1d3",
}

real_network_colors = {"node_fill": "#D9EBF2", "node_edge": "none"}
random_network_colors = {"node_fill": "#F2E0D9", "node_edge": "none"}
background_color = "#FAFAF8"
fallback_node_color = "#E7E7E7"

INDICATOR_COLORS = indicator_colors
INDICATOR_DISPLAY_NAMES = indicator_display_names
EDGE_COLORS = edge_colors
REAL_DATA_FAMILY = real_network_colors
RANDOM_DATA_FAMILY = random_network_colors
REAL_NETWORK_COLORS = real_network_colors
RANDOM_NETWORK_COLORS = random_network_colors
BACKGROUND_COLOR = background_color
FALLBACK_NODE_COLOR = fallback_node_color


def indicator_color(name: str) -> str:
    return indicator_colors.get(str(name), fallback_node_color)


def indicator_display_name(name: str) -> str:
    return indicator_display_names.get(str(name), str(name).replace("_", " ").title())

