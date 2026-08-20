from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.feature_config import FEATURES
from utils.figure2_paths import match_data_root
from utils.match_data_loader import DATASET_FILES, clean_indicator_var_data, load_match_level_data
from utils.network_comparison import N_PERMUTATIONS, RANDOM_SEED


DATASETS = ("ligue1", "euro", "worldcup")
DISPLAY_NAMES = {
    "ligue1": "Ligue 1",
    "euro": "EURO",
    "worldcup": "World Cup",
}


@dataclass(frozen=True)
class DatasetAudit:
    dataset: str
    display_name: str
    path: Path
    sheet_name: str
    raw_total_n: int
    raw_var0_n: int
    raw_var1_n: int
    raw_var_invalid_n: int
    raw_var_sum_matches_total: bool
    missing_counts: dict[str, int]
    rows_with_any_missing: int
    complete_valid_n: int
    dropped_total_n: int
    dropped_var0_n: int
    dropped_var1_n: int
    analysis_before_n: int
    analysis_after_n: int
    analysis_total_n: int
    before_ratio: float
    after_ratio: float
    group_a_min: int
    group_a_max: int
    group_b_min: int
    group_b_max: int
    abnormal_permutation_count: int
    random_seed: int
    n_permutations: int


def _raw_path(dataset: str) -> Path:
    return match_data_root() / DATASET_FILES[dataset]


def _coerced_analysis_frame(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [*FEATURES, "VAR"]
    frame = raw[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def _verify_permutation_sizes(total_n: int, group0_size: int) -> tuple[int, int, int, int, int]:
    group1_size = total_n - group0_size
    assert group0_size + group1_size == total_n
    rng = np.random.default_rng(RANDOM_SEED)
    group_a_counts: list[int] = []
    group_b_counts: list[int] = []
    abnormal = 0
    for _ in range(N_PERMUTATIONS):
        labels = rng.permutation(np.r_[np.zeros(group0_size, dtype=int), np.ones(group1_size, dtype=int)])
        if len(labels) != total_n:
            abnormal += 1
        group_a_n = int(np.sum(labels == 0))
        group_b_n = int(np.sum(labels == 1))
        group_a_counts.append(group_a_n)
        group_b_counts.append(group_b_n)
        if group_a_n != group0_size or group_b_n != group1_size:
            abnormal += 1
    return (
        min(group_a_counts),
        max(group_a_counts),
        min(group_b_counts),
        max(group_b_counts),
        abnormal,
    )


def audit_dataset(dataset: str) -> DatasetAudit:
    path = _raw_path(dataset)
    if not path.exists():
        raise FileNotFoundError(path)

    raw = pd.read_excel(path)
    raw_total_n = int(len(raw))
    coerced = _coerced_analysis_frame(raw)
    var_numeric = coerced["VAR"]
    raw_var0_n = int(var_numeric.eq(0).sum())
    raw_var1_n = int(var_numeric.eq(1).sum())
    raw_var_invalid_n = int((~var_numeric.isin([0, 1])).sum())
    raw_var_sum_matches_total = (raw_var0_n + raw_var1_n) == raw_total_n

    missing_counts = {column: int(coerced[column].isna().sum()) for column in [*FEATURES, "VAR"]}
    missing_mask = coerced[[*FEATURES, "VAR"]].isna().any(axis=1)
    rows_with_any_missing = int(missing_mask.sum())
    complete = coerced.dropna(subset=[*FEATURES, "VAR"]).copy()
    complete["VAR"] = complete["VAR"].astype(int)
    complete_valid_n = int(len(complete))
    dropped_total_n = raw_total_n - complete_valid_n
    dropped_var0_n = int(var_numeric.loc[missing_mask].eq(0).sum())
    dropped_var1_n = int(var_numeric.loc[missing_mask].eq(1).sum())

    loaded = load_match_level_data(dataset)
    code_values, code_labels = clean_indicator_var_data(loaded, list(FEATURES))
    analysis_before_n = int(complete["VAR"].eq(0).sum())
    analysis_after_n = int(complete["VAR"].eq(1).sum())
    analysis_total_n = int(len(complete))

    assert list(FEATURES) == list(code_values.columns)
    assert len(code_values) == analysis_total_n
    assert len(code_labels) == analysis_total_n
    assert int(code_labels.eq(0).sum()) == analysis_before_n
    assert int(code_labels.eq(1).sum()) == analysis_after_n
    assert analysis_before_n + analysis_after_n == analysis_total_n
    assert complete_valid_n + dropped_total_n == raw_total_n
    assert raw_var0_n + raw_var1_n + raw_var_invalid_n == raw_total_n

    group_a_min, group_a_max, group_b_min, group_b_max, abnormal = _verify_permutation_sizes(
        analysis_total_n,
        analysis_before_n,
    )
    assert group_a_min == group_a_max == analysis_before_n
    assert group_b_min == group_b_max == analysis_after_n
    assert abnormal == 0

    return DatasetAudit(
        dataset=dataset,
        display_name=DISPLAY_NAMES[dataset],
        path=path,
        sheet_name="Sheet1",
        raw_total_n=raw_total_n,
        raw_var0_n=raw_var0_n,
        raw_var1_n=raw_var1_n,
        raw_var_invalid_n=raw_var_invalid_n,
        raw_var_sum_matches_total=raw_var_sum_matches_total,
        missing_counts=missing_counts,
        rows_with_any_missing=rows_with_any_missing,
        complete_valid_n=complete_valid_n,
        dropped_total_n=dropped_total_n,
        dropped_var0_n=dropped_var0_n,
        dropped_var1_n=dropped_var1_n,
        analysis_before_n=analysis_before_n,
        analysis_after_n=analysis_after_n,
        analysis_total_n=analysis_total_n,
        before_ratio=analysis_before_n / analysis_total_n if analysis_total_n else 0.0,
        after_ratio=analysis_after_n / analysis_total_n if analysis_total_n else 0.0,
        group_a_min=group_a_min,
        group_a_max=group_a_max,
        group_b_min=group_b_min,
        group_b_max=group_b_max,
        abnormal_permutation_count=abnormal,
        random_seed=RANDOM_SEED,
        n_permutations=N_PERMUTATIONS,
    )


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(frame.to_string(index=False))


def main() -> None:
    audits = [audit_dataset(dataset) for dataset in DATASETS]

    raw_rows = [
        {
            "dataset": audit.dataset,
            "raw_total_n": audit.raw_total_n,
            "raw_VAR0_n": audit.raw_var0_n,
            "raw_VAR1_n": audit.raw_var1_n,
            "raw_VAR_invalid_or_missing_n": audit.raw_var_invalid_n,
            "raw_VAR_sum_matches_total": audit.raw_var_sum_matches_total,
        }
        for audit in audits
    ]
    _print_table("Raw VAR counts", pd.DataFrame(raw_rows))

    cleaning_rows = [
        {
            "dataset": audit.dataset,
            "rows_with_any_analysis_missing": audit.rows_with_any_missing,
            "dropped_VAR0_n": audit.dropped_var0_n,
            "dropped_VAR1_n": audit.dropped_var1_n,
            "complete_valid_n": audit.complete_valid_n,
        }
        for audit in audits
    ]
    _print_table("Complete-case cleaning", pd.DataFrame(cleaning_rows))

    analysis_rows = [
        {
            "dataset": audit.dataset,
            "analysis_before_n": audit.analysis_before_n,
            "analysis_after_n": audit.analysis_after_n,
            "analysis_total_n": audit.analysis_total_n,
            "before_ratio": round(audit.before_ratio, 6),
            "after_ratio": round(audit.after_ratio, 6),
            "permutation_group_A_min": audit.group_a_min,
            "permutation_group_A_max": audit.group_a_max,
            "permutation_group_B_min": audit.group_b_min,
            "permutation_group_B_max": audit.group_b_max,
            "abnormal_permutation_count": audit.abnormal_permutation_count,
        }
        for audit in audits
    ]
    _print_table("Analysis and permutation group sizes", pd.DataFrame(analysis_rows))

    missing_rows = []
    for audit in audits:
        for column in [*FEATURES, "VAR"]:
            missing_rows.append(
                {
                    "dataset": audit.dataset,
                    "column": column,
                    "missing_or_non_numeric_n": audit.missing_counts[column],
                }
            )
    _print_table("Missing/non-numeric counts by analysis column", pd.DataFrame(missing_rows))

    print("\nAssertions passed for all datasets.")
    print(f"random_seed={RANDOM_SEED}; n_permutations={N_PERMUTATIONS}")


if __name__ == "__main__":
    main()
