# -*- coding: utf-8 -*-
"""
Pipeline entry point for manuscript Table 1.

This script reads the standardized match-level Excel files from
``paper_assets/data`` and generates Table 1 Excel workbooks.

Workflow:

data/*.xlsx
    ↓
read_match_data()
    ↓
generate_table1()
    ↓
write_table1_workbook()
    ↓
table1/results/*.xlsx
"""

from __future__ import annotations

from pathlib import Path
import importlib.util
import sys

import pandas as pd


# ============================================================
# Import statistics module
# ============================================================

# Current directory:
# paper_assets/table1/src
SRC_DIR = Path(__file__).resolve().parent

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


# Load statistics.py dynamically
STATISTICS_PATH = SRC_DIR / "statistics.py"

spec = importlib.util.spec_from_file_location(
    "table1_statistics",
    STATISTICS_PATH
)

if spec is None or spec.loader is None:
    raise ImportError(
        f"Cannot load Table1 statistics module: {STATISTICS_PATH}"
    )


table1_statistics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(table1_statistics)


# Import functions from statistics.py
FEATURES = table1_statistics.FEATURES

generate_table1 = table1_statistics.generate_table1

write_table1_workbook = (
    table1_statistics.write_table1_workbook
)


# ============================================================
# Path configuration
# ============================================================

# table1/
TABLE1_ROOT = Path(__file__).resolve().parents[1]

# paper_assets/
PAPER_ASSETS_ROOT = TABLE1_ROOT.parent


# Input data directory:
#
# paper_assets/data/
#   ├── ligue1_matches.xlsx
#   ├── euro_matches.xlsx
#   └── worldcup_matches.xlsx
#
DATA_DIR = PAPER_ASSETS_ROOT / "data"


# Output directory:
#
# paper_assets/table1/results/
#
RESULTS_DIR = TABLE1_ROOT / "results"



# ============================================================
# Input files
# ============================================================

INPUT_FILES = {

    "ligue1":
        DATA_DIR / "ligue1_matches.xlsx",

    "euro":
        DATA_DIR / "euro_matches.xlsx",

    "worldcup":
        DATA_DIR / "worldcup_matches.xlsx",
}



# ============================================================
# Output files
# ============================================================

OUTPUT_FILES = {

    "ligue1":
        RESULTS_DIR / "table1_ligue1.xlsx",

    "euro":
        RESULTS_DIR / "table1_euro.xlsx",

    "worldcup":
        RESULTS_DIR / "table1_worldcup.xlsx",

    "combined":
        RESULTS_DIR / "table1_combined.xlsx",
}



# ============================================================
# Required input columns
# ============================================================

# These columns must exist in every dataset.

REQUIRED_COLUMNS = [

    "home_team",

    "away_team",

    *FEATURES,

    "VAR",
]



# ============================================================
# Read standardized match data
# ============================================================

def read_match_data(path: Path) -> pd.DataFrame:
    """
    Read one standardized match-level Excel file.

    Input:
        ligue1_matches.xlsx
        euro_matches.xlsx
        worldcup_matches.xlsx

    Output:
        pandas DataFrame containing required variables only.
    """

    # Check file existence
    if not path.exists():
        raise FileNotFoundError(
            f"Table1 input not found: {path}"
        )


    # Read Excel
    frame = pd.read_excel(path)


    # Check required columns
    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in frame.columns
    ]


    if missing:
        raise ValueError(
            f"Missing required columns: {missing}; file={path}"
        )


    # Keep only analysis-related columns
    return frame.loc[:, REQUIRED_COLUMNS].copy()



# ============================================================
# Main pipeline
# ============================================================

def main() -> None:
    """
    Generate all Table 1 Excel workbooks.
    """

    # Create output directory
    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    # Store individual tables
    tables = {}


    # Store audit information from statistics.py
    audits = {}



    # Process each dataset
    for dataset, input_path in INPUT_FILES.items():


        print(
            f"[Table1] Reading {dataset}: {input_path}"
        )


        # Load data
        data = read_match_data(input_path)


        # Statistical analysis
        #
        # Includes:
        # - Mean
        # - Median
        # - IQR
        # - Mann-Whitney U
        # - Z
        # - p-value
        # - Effect size
        #
        table, audit = generate_table1(data)


        tables[dataset] = table

        audits[dataset] = audit



        # Save individual Table1
        write_table1_workbook(
            OUTPUT_FILES[dataset],
            table,
            audit
        )


        print(
            f"[Table1] Written: {OUTPUT_FILES[dataset]}"
        )



    # ========================================================
    # Combine three datasets into one workbook
    # ========================================================

    combined = pd.concat(
        [
            table.assign(
                Dataset=dataset
            ).loc[
                :,
                [
                    "Dataset",
                    *table.columns.tolist()
                ]
            ]

            for dataset, table in tables.items()
        ],

        ignore_index=True
    )



    combined_audit = pd.concat(
        [
            audit.assign(
                Dataset=dataset
            )

            for dataset, audit in audits.items()
        ],

        ignore_index=True
    )



    # Save combined workbook
    write_table1_workbook(
        OUTPUT_FILES["combined"],
        combined,
        combined_audit,

        # Each dataset as separate sheet
        extra_sheets={
            dataset: table
            for dataset, table in tables.items()
        }
    )


    print(
        f"[Table1] Written: {OUTPUT_FILES['combined']}"
    )



# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()