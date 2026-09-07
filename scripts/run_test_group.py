"""Run complete core or ML test groups, including pytest and unittest cases."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TESTS_DIR = PROJECT_ROOT / "tests"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def selected_test_files(group: str) -> list[Path]:
    """Return deterministic test files for one CI group."""
    files = sorted(TESTS_DIR.glob("test_*.py"))
    if group == "ml":
        return [path for path in files if path.name.startswith("test_ml_")]
    if group == "core":
        return [path for path in files if not path.name.startswith("test_ml_")]
    raise ValueError(f"Unknown test group: {group}")


def main() -> int:
    import pytest

    parser = argparse.ArgumentParser(description="Run one repository test group.")
    parser.add_argument("group", choices=["core", "ml"])
    args = parser.parse_args()
    files = selected_test_files(args.group)
    if not files:
        parser.error(f"No tests found for group: {args.group}")
    return int(pytest.main(["-q", *(str(path) for path in files)]))


if __name__ == "__main__":
    sys.exit(main())
