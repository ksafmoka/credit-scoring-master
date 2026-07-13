#!/usr/bin/env python3
"""Fail fast if Python version is unsupported for this project."""

from __future__ import annotations

import sys


def main() -> int:
    major, minor = sys.version_info[:2]
    ok = major == 3 and 10 <= minor <= 12
    print(f"Python {major}.{minor}.{sys.version_info.micro}  ({sys.executable})")
    if not ok:
        print(
            "\nERROR: this project needs Python 3.10, 3.11, or 3.12.\n"
            "Python 3.13/3.14 often has no binary wheels for pandas/lightgbm/catboost,\n"
            "so pip tries to compile from source and fails.\n\n"
            "Fix (WSL / Ubuntu example):\n"
            "  sudo apt update\n"
            "  sudo apt install -y python3.11 python3.11-venv\n"
            "  cd /path/to/credit-scoring-uplift-master\n"
            "  rm -rf .venv\n"
            "  python3.11 -m venv .venv\n"
            "  source .venv/bin/activate\n"
            "  python -m pip install -U pip setuptools wheel\n"
            "  pip install -r requirements/requirements-local.txt\n"
        )
        return 1
    print("Python version OK for local install.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
