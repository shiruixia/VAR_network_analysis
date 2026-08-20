# -*- coding: utf-8 -*-
"""Unified manuscript color configuration based on Scientific Colour Maps vik.

The HEX values below are parsed from:

- code/vik/vik10.txt
- code/vik/vik25.txt
- code/vik/vik50.txt
- code/vik/vik100.txt

No runtime colormap dependency is required.
"""

VIK10 = [
    "#001261",
    "#033E7D",
    "#1E6F9D",
    "#71A8C4",
    "#C9DDE7",
    "#EACEBD",
    "#D39774",
    "#BE6533",
    "#8B2706",
    "#590008",
]

VIK25 = [
    "#001261",
    "#02236C",
    "#023376",
    "#034481",
    "#06568C",
    "#156798",
    "#307DA6",
    "#4E92B4",
    "#71A8C4",
    "#94BED2",
    "#B3D1DF",
    "#D5E3E9",
    "#ECE5E0",
    "#EDD5C8",
    "#E4BFAA",
    "#DCAC90",
    "#D39774",
    "#CB835A",
    "#C37243",
    "#BA5E2A",
    "#A94512",
    "#942F06",
    "#7E1D06",
    "#6C0E07",
    "#590008",
]

VIK50 = [
    "#001261",
    "#011A66",
    "#02226B",
    "#022B71",
    "#023376",
    "#023A7B",
    "#034280",
    "#034A85",
    "#06548B",
    "#0B5D91",
    "#136697",
    "#1E6F9D",
    "#2B79A4",
    "#3C85AC",
    "#4B90B3",
    "#5A9ABA",
    "#6AA4C1",
    "#7AAEC8",
    "#8DBAD0",
    "#9DC4D6",
    "#ADCDDD",
    "#BDD6E3",
    "#CCDFE8",
    "#DEE6E9",
    "#E8E7E5",
    "#EEE3DC",
    "#EEDBD0",
    "#EBD0C0",
    "#E7C6B2",
    "#E3BCA5",
    "#DFB298",
    "#DBA88B",
    "#D69D7C",
    "#D29470",
    "#CE8B64",
    "#CA8258",
    "#C6794C",
    "#C26E3F",
    "#BE6533",
    "#B85C28",
    "#B2511D",
    "#A94512",
    "#9C3709",
    "#912D06",
    "#872406",
    "#7E1D06",
    "#741506",
    "#6A0D07",
    "#620607",
    "#590008",
]

VIK100 = [
    "#001261",
    "#011764",
    "#011A66",
    "#021F69",
    "#02226B",
    "#02276E",
    "#022A70",
    "#022E73",
    "#023376",
    "#023678",
    "#023A7B",
    "#033E7D",
    "#034280",
    "#034582",
    "#034A85",
    "#044F88",
    "#05528A",
    "#07578D",
    "#095B90",
    "#0E6093",
    "#136697",
    "#176999",
    "#1E6F9D",
    "#2373A0",
    "#2B79A4",
    "#307DA6",
    "#3983AB",
    "#4289AF",
    "#488DB2",
    "#5194B6",
    "#5798B9",
    "#619EBD",
    "#67A2C0",
    "#71A8C4",
    "#7AAEC8",
    "#80B2CA",
    "#8AB8CE",
    "#90BCD1",
    "#9AC2D5",
    "#A0C5D8",
    "#AACBDC",
    "#B3D1DF",
    "#BAD5E2",
    "#C3DAE5",
    "#C9DDE7",
    "#D2E1E9",
    "#D8E4E9",
    "#E0E6E9",
    "#E7E7E7",
    "#EAE6E4",
    "#EDE4DE",
    "#EEE1DA",
    "#EEDDD3",
    "#EDD7CB",
    "#ECD3C5",
    "#EACEBD",
    "#E9CAB8",
    "#E6C4B0",
    "#E4BFAA",
    "#E2BAA2",
    "#DFB49A",
    "#DEB095",
    "#DBAA8D",
    "#DAA688",
    "#D7A081",
    "#D69D7C",
    "#D39774",
    "#D1926D",
    "#CF8E68",
    "#CD8961",
    "#CC855D",
    "#C98056",
    "#C87C51",
    "#C6774A",
    "#C37243",
    "#C26E3F",
    "#BF6938",
    "#BE6533",
    "#BB602D",
    "#B85C28",
    "#B55521",
    "#B04F1B",
    "#AD4A16",
    "#A74310",
    "#A33E0D",
    "#9C3709",
    "#963107",
    "#912D06",
    "#8B2706",
    "#872406",
    "#811F06",
    "#7E1D06",
    "#781806",
    "#731406",
    "#6F1107",
    "#6A0D07",
    "#670A07",
    "#620607",
    "#5E0308",
    "#590008",
]

# Figure 1 node color palette.
#
# IMPORTANT:
# Figure 1 currently uses the ColorBrewer Set3 qualitative palette requested
# for the nine-node network display. This is the main edit location for Figure 1
# node colors.
#
# Colors are assigned in the same order used by
# make_paper_figure1_network_comparison.py:
#
# first_half_time -> second_half_time -> total_time -> total_offsides ->
# total_goals -> total_red_cards -> total_fouls -> total_yellow_cards ->
# total_penalties.
#
# Current requested Set3 color sequence:
# ["#8dd3c7", "#ffffb3", "#bebada", "#fdb462", "#b3de69",
#  "#fccde5", "#d9d9d9", "#bc80bd", "#ccebc5", "#ffed6f"]
indicator_colors = {
    "first_half_time": "#8dd3c7",
    "second_half_time": "#ffffb3",
    "total_time": "#bebada",
    "total_offsides": "#fdb462",
    "total_goals": "#b3de69",
    "total_red_cards": "#fccde5",
    "total_fouls": "#d9d9d9",
    "total_yellow_cards": "#bc80bd",
    "total_penalties": "#ccebc5",
}

indicator_display_names = {
    "first_half_time": "First-half time",
    "second_half_time": "Second-half time",
    "total_time": "Total time",
    "goals": "Goals",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "fouls": "Fouls",
    "offsides": "Offsides",
    "penalties": "Penalties",
    "total_penalties": "Penalties",
    "total_goals": "Goals",
    "total_yellow_cards": "Yellow cards",
    "total_red_cards": "Red cards",
    "total_fouls": "Fouls",
    "total_offsides": "Offsides",
}

figure1_node_display_labels = {
    "first_half_time": "First-half time",
    "second_half_time": "Second-half time",
    "total_time": "Total time",
    "goals": "Goals",
    "yellow_cards": "Yellow cards",
    "red_cards": "Red cards",
    "fouls": "Fouls",
    "offsides": "Offsides",
    "penalties": "Penalties",
    "total_penalties": "Penalties",
    "total_goals": "Goals",
    "total_yellow_cards": "Yellow cards",
    "total_red_cards": "Red cards",
    "total_fouls": "Fouls",
    "total_offsides": "Offsides",
}
edge_colors = {
    "positive": "#fb8072",
    "negative": "#80b1d3",
}

real_network_colors = {
    "node_fill": "#bebada",
    "node_edge": "#1F2937",
}

random_network_colors = {
    "node_fill": "#ffffb3",
    "node_edge": "#1F2937",
}

background_color = "#FAFAF8"
legend_text_color = "#222222"
node_edge_color = "#1F2937"
fallback_node_color = "#E7E7E7"
FIGURE1_NODE_COLOR = "#D9EBF2"

# Backward-compatible aliases for existing figure scripts.
INDICATOR_COLORS = indicator_colors
INDICATOR_DISPLAY_NAMES = indicator_display_names
EDGE_COLORS = edge_colors
REAL_DATA_FAMILY = real_network_colors
RANDOM_DATA_FAMILY = random_network_colors
BACKGROUND_COLOR = background_color
LEGEND_TEXT_COLOR = legend_text_color
NODE_EDGE_COLOR = node_edge_color
FALLBACK_NODE_COLOR = fallback_node_color


def indicator_color(name: str) -> str:
    """Return the stable indicator color, or a neutral fallback."""
    return indicator_colors.get(str(name), fallback_node_color)


def indicator_display_name(name: str) -> str:
    """Return the manuscript display name without changing raw variable names."""
    return indicator_display_names.get(str(name), str(name).replace("_", " ").title())

# Shared network drawing parameters for manuscript Figure1/Figure2.
# These values mirror the current Figure2 schematic network style and are
# intentionally centralized here so Figure1 and Figure2 do not drift apart.
NETWORK_STYLE = {
    "real_node_color": "#bebada",
    "random_node_color": "#ffffb3",
    "node_edge_color": "#1F2937",
    "node_size": 700,
    "figsize": (5.2, 5.2),
    "pair_figsize": (10.4, 5.2),
    "layout_scale": {
        "ligue1": 1.55,
        "euro": 1.00,
        "worldcup": 1.00,
    },
    "axis_padding": 0.18,
}


def figure1_node_display_label(name: str) -> str:
    """Return compact Figure1 node label for text inside nodes."""
    return figure1_node_display_labels.get(str(name), indicator_display_name(name))
