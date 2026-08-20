from __future__ import annotations

from pathlib import Path
import sys

SRC = Path(__file__).resolve().parents[1]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from utils.dataset_entrypoints import run_real_network


DATASET = Path(__file__).resolve().parent.name


def run() -> list[Path]:
    return run_real_network(DATASET)


if __name__ == "__main__":
    for path in run():
        print(path)
