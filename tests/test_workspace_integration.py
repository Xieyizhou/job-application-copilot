"""Sanitized end-to-end coverage for a configured Personal workspace."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apply_package import create_application_package
from ml.jd_quality import JDQualityError
from workspace import initialize_personal_workspace


class PersonalWorkspaceIntegrationTests(unittest.TestCase):
    def test_sanitized_personal_workspace_generates_package_and_tracker(self) -> None:
        candidate_text = """# Fictional Candidate

## Skills
- Python, SQL, pandas, and data visualization

## Experience
- Built a fictional reporting dashboard for sample datasets.
"""
        job_text = """# Fictional Analytics Labs — Data Analyst

Company: Fictional Analytics Labs
Role: Data Analyst
Location: Remote
Job URL: https://example.invalid/jobs/data-analyst
Source: manual
Company Confirmed By User: yes
Company Confidence: High
Company Evidence: Confirmed during sanitized test setup.

Description Source: full_jd_manual
JD Fetch Status: complete

## Job Description
Responsibilities
You will analyze operational and customer data to explain measurable business outcomes.
You will build and maintain SQL transformations, recurring dashboards, and data-quality checks.
You will collaborate with product and operations partners on clearly documented questions.
You will communicate findings, assumptions, and limitations to technical and business audiences.

Requirements
Required experience with Python and SQL for reproducible analysis and reporting workflows.
Must have experience building dashboards and validating metrics against source datasets.
Knowledge of statistics, data visualization, and practical quality-control methods is required.
Ability to translate ambiguous questions into documented analytical steps is required.
Experience communicating findings to cross-functional stakeholders is preferred.

About the team
This fictional analytics team maintains shared data products for several business functions.
Team members review queries, document metric definitions, test recurring workflows, and improve
how decisions are supported. The role includes peer review, established engineering practices,
privacy-aware handling of sample data, professional development, and regular feedback. The analyst
balances independent investigation with collaboration and helps the organization use reliable,
clearly explained evidence in routine planning and operational reviews.
"""

        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir) / "local_workspace"
            workspace = initialize_personal_workspace(
                "fictional_candidate.md", candidate_text.encode("utf-8"), root=root
            )
            job_path = workspace.jobs_dir / "fictional_data_analyst.md"
            job_path.write_text(job_text, encoding="utf-8")

            manifest_text = (root / "workspace.json").read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            self.assertNotIn(candidate_text, manifest_text)
            self.assertEqual(manifest["resume_source"], "candidate/candidate_source.md")

            summary = create_application_package(
                job_path,
                workspace,
                company="Fictional Analytics Labs",
                role="Data Analyst",
                location="Remote",
                job_url="https://example.invalid/jobs/data-analyst",
            )

            for key in ["analysis_path", "cover_letter_path", "cover_letter_docx_path", "cover_letter_notes_path"]:
                path = summary[key]
                self.assertIsInstance(path, Path)
                self.assertTrue(path.is_file())
                self.assertTrue(path.is_relative_to(workspace.generated_dir))
            self.assertNotIn("resume_path", summary)
            self.assertNotIn("resume_docx_path", summary)
            self.assertFalse((summary["package_dir"] / "tailored_resume.md").exists())
            self.assertFalse((summary["package_dir"] / "tailored_resume.docx").exists())
            cover_letter_text = summary["cover_letter_path"].read_text(encoding="utf-8")
            notes_text = summary["cover_letter_notes_path"].read_text(encoding="utf-8")
            self.assertIn("Python, SQL, pandas, and data visualization", cover_letter_text)
            self.assertIn("Requirement-to-Resume Evidence Map", notes_text)
            self.assertIn("Direct support", notes_text)
            self.assertTrue(workspace.tracker_database_path and workspace.tracker_database_path.is_file())
            self.assertIsInstance(summary["tracker_id"], int)

    @patch("jd_enrichment.jsearch_configured", return_value=False)
    def test_incomplete_jd_cannot_generate_employer_facing_files(self, _configured) -> None:
        candidate_text = "# Fictional Candidate\n\n- Python and SQL reporting experience."
        incomplete_job = """# Data Analyst
Company: Fictional Analytics Labs
Role: Data Analyst
Location: Remote
Job URL: https://example.invalid/jobs/incomplete
Source: Jooble
Description Source: api_snippet
JD Fetch Status: snippet_only
Company Confirmed By User: yes
Company Confidence: High

## Job Description
Use Python and SQL for reporting...
"""
        with tempfile.TemporaryDirectory() as temporary_dir:
            workspace = initialize_personal_workspace(
                "fictional_candidate.md",
                candidate_text.encode("utf-8"),
                root=Path(temporary_dir) / "local_workspace",
            )
            job_path = workspace.jobs_dir / "incomplete.md"
            job_path.write_text(incomplete_job, encoding="utf-8")

            with self.assertRaisesRegex(JDQualityError, "scoring-ready full job description"):
                create_application_package(
                    job_path,
                    workspace,
                    company="Fictional Analytics Labs",
                    role="Data Analyst",
                    location="Remote",
                    job_url="https://example.invalid/jobs/incomplete",
                )

            self.assertFalse(any(workspace.generated_dir.rglob("cover_letter.md")))


if __name__ == "__main__":
    unittest.main()
