# -*- coding: utf-8 -*-
"""Compatibility wrapper for manuscript Figure 3 generation."""
from __future__ import annotations

import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from make_paper_figure3 import main


if __name__ == "__main__":
    for output in main():
        print(output)
