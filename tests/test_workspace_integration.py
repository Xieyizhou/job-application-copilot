"""Sanitized end-to-end coverage for a configured Personal workspace."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from apply_package import create_application_package
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

## Requirements
- Python and SQL
- Analyze data and communicate findings
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
            self.assertTrue(workspace.tracker_database_path and workspace.tracker_database_path.is_file())
            self.assertIsInstance(summary["tracker_id"], int)


if __name__ == "__main__":
    unittest.main()
