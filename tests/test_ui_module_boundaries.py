"""Guard the decomposed UI and cover-letter orchestration boundaries."""

from __future__ import annotations

import ast
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def top_level_function_size(relative_path: str, function_name: str) -> int:
    """Return source line count for one top-level function."""
    tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            return int(node.end_lineno or node.lineno) - node.lineno + 1
    raise AssertionError(f"Function not found: {relative_path}::{function_name}")


def file_size(relative_path: str) -> int:
    """Return the physical line count for a source file."""
    return len((PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines())


class UIOrchestrationBoundaryTests(unittest.TestCase):
    def test_review_jobs_entrypoint_stays_orchestration_only(self) -> None:
        self.assertLessEqual(
            top_level_function_size("src/dashboard_review_page.py", "job_descriptions_tab"),
            50,
        )

    def test_manual_entrypoint_stays_orchestration_only(self) -> None:
        self.assertLessEqual(
            top_level_function_size("src/dashboard_manual.py", "render_manual_add_extract_tab"),
            35,
        )

    def test_phase_two_ui_files_stay_bounded(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_manual.py"), 750)
        self.assertLessEqual(file_size("src/dashboard_manual_entry.py"), 140)
        self.assertLessEqual(file_size("src/dashboard_cover_letter_components.py"), 250)
        self.assertLessEqual(file_size("src/dashboard_cover_letter_selection.py"), 100)
        self.assertLessEqual(
            top_level_function_size("src/dashboard_cover_letter.py", "package_viewer_tab"),
            30,
        )

    def test_phase_three_ui_files_stay_bounded(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_home.py"), 140)
        self.assertLessEqual(file_size("src/dashboard_tracker.py"), 80)
        self.assertLessEqual(file_size("src/dashboard_tracker_components.py"), 200)
        self.assertLessEqual(file_size("src/dashboard_settings.py"), 120)
        self.assertLessEqual(file_size("src/dashboard_settings_sections.py"), 130)
        self.assertLessEqual(top_level_function_size("src/dashboard_home.py", "dashboard_tab"), 35)
        self.assertLessEqual(top_level_function_size("src/dashboard_tracker.py", "tracker_tab"), 45)
        self.assertLessEqual(top_level_function_size("src/dashboard_settings.py", "safety_notes_tab"), 35)

    def test_cover_letter_entrypoints_delegate_details(self) -> None:
        self.assertLessEqual(
            top_level_function_size("src/generate_cover_letter.py", "generate_cover_letter"),
            40,
        )
        self.assertLessEqual(
            top_level_function_size("src/generate_cover_letter.py", "build_internal_notes"),
            20,
        )


if __name__ == "__main__":
    unittest.main()
