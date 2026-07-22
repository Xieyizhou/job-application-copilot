"""Tests for deterministic Core/ML CI test selection."""

from __future__ import annotations

import unittest

from scripts.run_test_group import selected_test_files


class TestGroupingTests(unittest.TestCase):
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
