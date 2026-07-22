"""Run explicit core or ML unittest groups without ambiguous glob exclusions."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys
import unittest


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


def build_suite(group: str) -> unittest.TestSuite:
    """Load each selected test file without requiring ``tests`` to be a package."""
    loader = unittest.defaultTestLoader
    suite = unittest.TestSuite()
    for path in selected_test_files(group):
        module_name = f"job_copilot_tests.{path.stem}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to load test file: {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        suite.addTests(loader.loadTestsFromModule(module))
    return suite


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one repository test group.")
    parser.add_argument("group", choices=["core", "ml"])
    args = parser.parse_args()
    result = unittest.TextTestRunner(verbosity=2).run(build_suite(args.group))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(main())
