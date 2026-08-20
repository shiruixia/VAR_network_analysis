from __future__ import annotations

from utils.feature_config import FEATURE_LABELS


def metric_label(name: object) -> str:
    """Return display label for a Figure2 metric."""
    key = str(name).strip()
    return FEATURE_LABELS.get(key, key.replace("_", " ").title())

