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
    def test_dashboard_page_wrappers_only_delegate_built_services(self) -> None:
        for function_name in (
            "dashboard_tab",
            "fetch_jobs_tab",
            "manual_job_target_tab",
            "job_descriptions_tab",
            "tracker_tab",
            "package_viewer_tab",
            "safety_notes_tab",
        ):
            self.assertLessEqual(
                top_level_function_size("src/dashboard.py", function_name),
                4,
            )

    def test_dashboard_repository_owns_read_only_data_orchestration(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_repository.py"), 340)
        self.assertLessEqual(
            top_level_function_size("src/dashboard.py", "list_job_description_files"),
            12,
        )
        self.assertLessEqual(
            top_level_function_size("src/dashboard.py", "build_dashboard_job_record"),
            10,
        )
        self.assertLessEqual(
            top_level_function_size("src/dashboard.py", "load_screened_jobs"),
            15,
        )
        self.assertLessEqual(
            top_level_function_size("src/dashboard.py", "load_tracker_rows"),
            20,
        )

    def test_dashboard_analysis_service_owns_caching_and_fallbacks(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_analysis_service.py"), 140)
        self.assertLessEqual(
            top_level_function_size("src/dashboard.py", "analyze_job_for_dashboard"),
            35,
        )

    def test_company_verification_controls_are_extracted_from_dashboard(self) -> None:
        self.assertLessEqual(
            file_size("src/dashboard_company_verification.py"),
            190,
        )
        for function_name in (
            "company_generation_allowed",
            "render_manual_company_confirmation",
            "render_markdown_company_confirmation",
        ):
            with self.assertRaises(AssertionError):
                top_level_function_size("src/dashboard.py", function_name)

    def test_review_jobs_entrypoint_stays_orchestration_only(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_review_page.py"), 520)
        self.assertLessEqual(file_size("src/dashboard_review_filters.py"), 350)
        self.assertLessEqual(
            top_level_function_size("src/dashboard_review_page.py", "job_descriptions_tab"),
            50,
        )
        self.assertLessEqual(
            top_level_function_size(
                "src/dashboard_review_filters.py",
                "render_review_filter_controls",
            ),
            45,
        )

    def test_manual_entrypoint_stays_orchestration_only(self) -> None:
        self.assertLessEqual(
            top_level_function_size("src/dashboard_manual.py", "render_manual_add_extract_tab"),
            35,
        )

    def test_find_jobs_entrypoint_stays_orchestration_only(self) -> None:
        self.assertLessEqual(file_size("src/dashboard_fetch.py"), 480)
        self.assertLessEqual(file_size("src/dashboard_fetch_history.py"), 190)
        self.assertLessEqual(
            top_level_function_size("src/dashboard_fetch.py", "fetch_jobs_tab"),
            25,
        )
        self.assertLessEqual(
            top_level_function_size("src/dashboard_fetch.py", "render_fetch_options_form"),
            45,
        )
        self.assertLessEqual(
            top_level_function_size("src/dashboard_fetch.py", "render_fetch_results"),
            15,
        )
        for function_name in (
            "fetch_run_label",
            "render_fetch_history_section",
            "render_fetch_run_job_cards",
            "render_fetch_run_job_table",
        ):
            with self.assertRaises(AssertionError):
                top_level_function_size("src/dashboard.py", function_name)

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
