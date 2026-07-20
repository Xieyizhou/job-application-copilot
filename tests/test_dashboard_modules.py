"""Focused contract tests for the refactored dashboard modules."""

from __future__ import annotations

import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard_fit import build_fit_presentation
from dashboard_packages import build_application_package_zip
from dashboard_regions import build_region_options, dynamic_source_options
from dashboard_review import review_inbox_view_matches
from dashboard_shell import switch_workspace_mode
import dashboard


class DashboardModuleContractTests(unittest.TestCase):
    def test_page_wrappers_inject_services_into_extracted_modules(self) -> None:
        with patch.object(dashboard, "render_home_page") as render_home:
            dashboard.dashboard_tab()
        home_services = render_home.call_args.args[0]
        self.assertIs(home_services.load_screened_jobs, dashboard.load_screened_jobs)
        self.assertIs(home_services.render_page_header, dashboard.render_page_header)

        with patch.object(dashboard, "render_tracker_page") as render_tracker:
            dashboard.tracker_tab()
        tracker_services = render_tracker.call_args.args[0]
        self.assertIs(tracker_services.load_tracker_rows, dashboard.load_tracker_rows)
        self.assertIs(tracker_services.run_with_captured_output, dashboard.run_with_captured_output)

        with patch.object(dashboard, "render_fetch_jobs_page") as render_fetch:
            dashboard.fetch_jobs_tab()
        fetch_services = render_fetch.call_args.args[0]
        self.assertIs(fetch_services.go_to_page, dashboard.go_to_page)
        self.assertIs(
            fetch_services.relocate_fetched_jobs_to_workspace,
            dashboard.relocate_fetched_jobs_to_workspace,
        )

        with patch.object(dashboard, "render_review_jobs_page") as render_review:
            dashboard.job_descriptions_tab()
        review_services = render_review.call_args.args[0]
        self.assertIs(review_services.load_screened_jobs, dashboard.load_screened_jobs)
        self.assertIs(review_services.save_job_to_tracker, dashboard.save_job_to_tracker)

    def test_workspace_switch_clears_cross_workspace_selection(self) -> None:
        state = {
            "workspace_mode": "Personal",
            "selected_review_job_path": "private/job.md",
            "latest_generated_package_dir": "private/generated",
        }

        switch_workspace_mode(state, "Demo")

        self.assertEqual(state["workspace_mode"], "Demo")
        self.assertFalse(state["workspace_setup_open"])
        self.assertNotIn("selected_review_job_path", state)
        self.assertNotIn("latest_generated_package_dir", state)

    def test_region_and_source_models_are_derived_from_loaded_jobs(self) -> None:
        jobs = [
            {"location": "Remote", "high_level_region": "Remote", "source": "jsearch"},
            {"location": "Toronto, Canada", "high_level_region": "Canada", "source": "jooble"},
        ]

        options = build_region_options(jobs)

        self.assertEqual(options["all"]["count"], 2)
        self.assertIn("high_level:remote", options)
        self.assertEqual(dynamic_source_options(jobs), ["all", "Jooble", "JSearch · Full JD"])

    def test_fit_presentation_keeps_low_confidence_score_provisional(self) -> None:
        job = {
            "analysis_available": True,
            "analysis_result": {
                "analysis_available": True,
                "coverage_score": 100,
                "score_breakdown": [],
            },
            "score": 62,
            "recommendation": "Manual Review",
            "confidence": {"level": "low"},
            "eligibility": {"status": "passed", "reasons": []},
        }

        presentation = build_fit_presentation(job)

        self.assertEqual(presentation["role_fit"], "Provisional 62/100")
        self.assertEqual(presentation["coverage_score"], 100)

    def test_review_filter_uses_fit_and_operational_state(self) -> None:
        job = {
            "analysis_available": True,
            "score": 86,
            "recommendation": "Apply",
            "confidence": {"level": "high"},
            "eligibility": {"status": "passed"},
        }

        self.assertTrue(review_inbox_view_matches(job, "Recommended", "Not tracked", "No cover letter"))
        self.assertTrue(review_inbox_view_matches(job, "Needs Review", "Not tracked", "No cover letter"))
        self.assertFalse(review_inbox_view_matches(job, "Ignored", "Not tracked", "No cover letter"))

    def test_bundle_zip_uses_allowlist_and_never_includes_resume(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            (bundle / "cover_letter.md").write_text("cover", encoding="utf-8")
            (bundle / "analysis.md").write_text("analysis", encoding="utf-8")
            (bundle / "resume.md").write_text("private resume", encoding="utf-8")

            zip_bytes, included = build_application_package_zip(bundle)

            self.assertEqual([path.name for path in included], ["cover_letter.md", "analysis.md"])
            with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
                self.assertEqual(archive.namelist(), ["cover_letter.md", "analysis.md"])


if __name__ == "__main__":
    unittest.main()
