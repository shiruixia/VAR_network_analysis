"""Shared indicator configuration for Figure1 data analysis.

The manuscript-level variable system is fixed to the nine match-level
indicators below. Legacy Ligue 1 and EURO result workbooks may still contain
the historical ``total_*`` column names; the alias helpers keep those old
result files readable without changing the analysis variables.
"""
from __future__ import annotations

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

LEGACY_TO_CANONICAL = {
    "total_goals": "goals",
    "total_yellow_cards": "yellow_cards",
    "total_red_cards": "red_cards",
    "total_fouls": "fouls",
    "total_offsides": "offsides",
    "total_penalties": "penalties",
}

CANONICAL_TO_LEGACY = {
    "goals": "total_goals",
    "yellow_cards": "total_yellow_cards",
    "red_cards": "total_red_cards",
    "fouls": "total_fouls",
    "offsides": "total_offsides",
    "penalties": "total_penalties",
}


def canonical_indicator(name: str) -> str:
    """Return the canonical manuscript indicator name."""
    return LEGACY_TO_CANONICAL.get(str(name), str(name))


def canonical_indicator_list(names: list[str]) -> list[str]:
    """Map a list of indicator names to the manuscript variable system."""
    return [canonical_indicator(name) for name in names]


def legacy_compatible_order(columns: list[str]) -> list[str]:
    """Return available columns ordered by MATCH_INDICATORS with legacy fallback."""
    column_set = {str(column) for column in columns}
    ordered: list[str] = []
    for indicator in MATCH_INDICATORS:
        if indicator in column_set:
            ordered.append(indicator)
            continue
        legacy = CANONICAL_TO_LEGACY.get(indicator)
        if legacy in column_set:
            ordered.append(legacy)
    return ordered
