"""Tests for deterministic Core/ML CI test selection."""

from __future__ import annotations

import unittest
import subprocess
import sys
import tempfile
from pathlib import Path

from scripts.run_test_group import selected_test_files


class TestGroupingTests(unittest.TestCase):
    def test_core_runner_executes_function_and_class_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            test_file = Path(directory) / "test_mixed.py"
            test_file.write_text(
                "import unittest\n"
                "class ExistingTest(unittest.TestCase):\n"
                "    def test_class(self): self.assertTrue(True)\n"
                "def test_function():\n"
                "    assert False, 'function-test-was-executed'\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                [sys.executable, "-c",
                 "from pathlib import Path; import sys; "
                 "from scripts import run_test_group as runner; "
                 "runner.TESTS_DIR = Path(sys.argv[1]); "
                 "sys.argv = ['run_test_group.py', 'core']; "
                 "raise SystemExit(runner.main())", directory],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
            )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
        self.assertIn("function-test-was-executed", result.stdout + result.stderr)
        self.assertIn("1 passed", result.stdout)

    def test_groups_are_disjoint_complete_and_keep_manual_tests(self) -> None:
        core = {path.name for path in selected_test_files("core")}
        ml = {path.name for path in selected_test_files("ml")}
        self.assertFalse(core & ml)
        self.assertTrue(all(name.startswith("test_ml_") for name in ml))
        self.assertTrue(all(not name.startswith("test_ml_") for name in core))
        self.assertIn("test_manual_jobs_lifecycle.py", core)
        self.assertIn("test_dashboard_manual_helpers.py", core)

    def test_unknown_group_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            selected_test_files("everything")


if __name__ == "__main__":
    unittest.main()
