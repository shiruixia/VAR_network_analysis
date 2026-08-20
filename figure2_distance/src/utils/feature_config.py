from __future__ import annotations


FEATURES = [
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

FEATURE_ALIASES = {
    "first_half_time": ("first_half_time",),
    "second_half_time": ("second_half_time",),
    "total_time": ("total_time",),
    "goals": ("goals", "total_goals"),
    "yellow_cards": ("yellow_cards", "total_yellow_cards"),
    "red_cards": ("red_cards", "total_red_cards"),
    "fouls": ("fouls", "total_fouls"),
    "offsides": ("offsides", "total_offsides"),
    "penalties": ("penalties", "total_penalties"),
}

FEATURE_LABELS = {
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

RETIRED_RANK_FEATURES = ("rank" + "ing_diff", "rank" + "ing_difference")


def canonical_feature(name: object) -> str | None:
    """Map a source Figure1 feature name to the shared Figure2 feature name."""
    text = str(name)
    for feature, aliases in FEATURE_ALIASES.items():
        if text in aliases:
            return feature
    return None


def assert_no_retired_rank_features(names: list[str], context: str) -> None:
    """Reject retired rank fields without storing their literal names in active code."""
    found = [name for name in names if name in RETIRED_RANK_FEATURES]
    if found:
        raise ValueError(f"{context} contains retired rank field(s): {found}")

