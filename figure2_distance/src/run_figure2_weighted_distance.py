from __future__ import annotations

from utils.dataset_entrypoints import run_all_datasets


def main() -> None:
    for path in run_all_datasets():
        print(path)


if __name__ == "__main__":
    main()
