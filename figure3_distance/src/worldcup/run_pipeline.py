from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from run_pipeline import run_dataset


def run() -> list[Path]:
    """Run the Figure3 WD pipeline for the dataset named by this directory."""
    return run_dataset(Path(__file__).resolve().parent.name)


if __name__ == "__main__":
    for path in run():
        print(path)
