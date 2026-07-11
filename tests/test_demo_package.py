"""Focused checks for the tracked, read-only Demo application package."""

from __future__ import annotations

import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from docx import Document
from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard


DEMO_PACKAGE_DIR = PROJECT_ROOT / "data" / "demo" / "sample_package"
DOCX_NAMES = ["tailored_resume.docx", "cover_letter.docx"]
EXPECTED_ZIP_NAMES = {
    "tailored_resume.md",
    "tailored_resume.docx",
    "cover_letter.md",
    "cover_letter.docx",
    "analysis.md",
    "tailoring_notes.md",
    "cover_letter_notes.md",
}


class DemoPackageTests(unittest.TestCase):
    def test_demo_docx_files_are_valid_and_neutral(self) -> None:
        expected_text = {
            "tailored_resume.docx": [
                "Demo Candidate",
                "Bachelor's degree in Data Science, Fictional Technical College",
                "Data Analysis Intern, Fictional Community Research Lab",
                "Built a classification pipeline with Python and scikit-learn.",
                "Evaluated models with cross-validation",
            ],
            "cover_letter.docx": [
                "Demo Candidate",
                "Northstar Metrics Studio",
                "fictional Data Analyst role",
                "fictional community research internship",
                "This is fictional Demo output and must not be submitted to an employer.",
            ],
        }
        for filename in DOCX_NAMES:
            path = DEMO_PACKAGE_DIR / filename
            self.assertTrue(path.is_file())
            self.assertGreater(path.stat().st_size, 0)
            with zipfile.ZipFile(path) as archive:
                self.assertIsNone(archive.testzip())
                names = set(archive.namelist())
                self.assertIn("word/document.xml", names)
                self.assertFalse(any(name.startswith("word/media/") for name in names))
                self.assertFalse(any(name.startswith("word/embeddings/") for name in names))
                self.assertFalse(any("comments" in name for name in names))
                self.assertNotIn("docProps/thumbnail.jpeg", names)
                self.assertFalse(any(name.startswith("customXml/") for name in names))

            document = Document(path)
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
            for expected in expected_text[filename]:
                self.assertIn(expected, text)
            self.assertEqual(document.core_properties.author, "Job Application Copilot Demo")
            self.assertEqual(document.core_properties.last_modified_by, "Job Application Copilot Demo")

    def test_demo_zip_includes_all_sanitized_sample_materials(self) -> None:
        zip_bytes, package_files = dashboard.build_application_package_zip(DEMO_PACKAGE_DIR)
        self.assertEqual({path.name for path in package_files}, EXPECTED_ZIP_NAMES)
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            self.assertEqual(set(archive.namelist()), EXPECTED_ZIP_NAMES)
            self.assertIsNone(archive.testzip())

    def test_demo_package_runtime_is_ready_and_does_not_create_personal_workspace(self) -> None:
        local_workspace = PROJECT_ROOT / "data" / "local_workspace"
        existed_before = local_workspace.exists()
        app = AppTest.from_file(PROJECT_ROOT / "src" / "dashboard.py")
        app.run(timeout=30)
        app.radio[0].set_value("Application Package").run(timeout=30)

        self.assertEqual(list(app.exception), [])
        statuses = app.table[0].value.set_index("Material")["Status"].to_dict()
        self.assertEqual(
            statuses,
            {
                "Tailored Resume": "Ready",
                "Resume DOCX": "Ready",
                "Cover Letter": "Ready",
                "Cover Letter DOCX": "Ready",
                "Match Report": "Ready",
                "Internal Notes": "Ready",
            },
        )
        self.assertEqual(
            {button.label for button in app.get("download_button")},
            {
                "Download Resume DOCX",
                "Download Cover Letter DOCX",
                "Download Match Report",
                "Download Internal Notes",
                "Download Full Application Package ZIP",
            },
        )
        self.assertEqual(local_workspace.exists(), existed_before)

    @patch.object(dashboard, "demo_mode_enabled", return_value=True)
    def test_demo_review_filter_and_sort_options_are_total(self, _demo_mode: object) -> None:
        jobs = dashboard.load_screened_jobs()
        self.assertTrue(jobs)
        tracker_rows: list[dict[str, object]] = []

        for inbox_view in [
            "Recommended",
            "Needs Review",
            "Package Ready",
            "Not Tracked",
            "Ignored",
            "All Jobs",
        ]:
            matches = [
                job
                for job in jobs
                if dashboard.review_inbox_view_matches(
                    job,
                    inbox_view,
                    dashboard.tracker_status_for_job(job, tracker_rows),
                    dashboard.package_status_for_job(job, tracker_rows),
                )
            ]
            self.assertIsInstance(matches, list)

        for sort_by in [
            "Score high to low",
            "Newest first",
            "Recommendation",
            "Company A-Z",
            "Package status",
            "Tracker status",
        ]:
            self.assertEqual(len(dashboard.sorted_review_jobs(jobs, sort_by)), len(jobs))

        region_options = dashboard.build_region_options(jobs)
        for option in region_options.values():
            matches = [job for job in jobs if dashboard.job_matches_region_option(job, option)]
            self.assertIsInstance(matches, list)

        sources = dashboard.dynamic_source_options(jobs)
        recommendations = ["all", "Apply", "Maybe Apply", "Skip or Low Priority"]
        confidences = ["all", *dashboard.CONFIDENCE_RANK]
        for source in sources:
            for recommendation in recommendations:
                for confidence in confidences:
                    matches = [
                        job
                        for job in jobs
                        if (source == "all" or dashboard.source_display_name(str(job["source"])) == source)
                        and (recommendation == "all" or job["recommendation"] == recommendation)
                        and (confidence == "all" or job["confidence"] == confidence)
                    ]
                    self.assertIsInstance(matches, list)

        first_job, second_job = jobs[:2]
        self.assertIs(
            dashboard.resolve_review_job_selection(jobs, "stale label", "data/demo/jobs/missing.md"),
            first_job,
        )
        self.assertIs(
            dashboard.resolve_review_job_selection(jobs, second_job["label"], str(first_job["path"])),
            second_job,
        )

    def test_missing_demo_docx_is_unavailable_without_generation(self) -> None:
        self.assertEqual(
            dashboard.readiness_status(
                source_exists=True,
                docx_exists=False,
                read_only_sample=True,
            ),
            "Unavailable",
        )
        zip_bytes, package_files = dashboard.build_application_package_zip(
            PROJECT_ROOT / "tests" / "fixtures" / "missing_demo_package"
        )
        self.assertEqual(package_files, [])
        with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
            self.assertEqual(archive.namelist(), [])


if __name__ == "__main__":
    unittest.main()
