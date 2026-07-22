"""Regression tests for safe full-JD lookup and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from jd_enrichment import enrich_saved_job_description
import manual_jobs
from ml.jd_quality import classify_jd_quality


TARGET_TEXT = """# Data Analyst
Company: Example Analytics
Company Normalized: Example Analytics
Role: Data Analyst
Location: Remote
Job URL: https://jobs.example.com/data-analyst
Source: Jooble
Description Source: api_snippet
JD Fetch Status: snippet_only
Company Confirmed By User: yes

## Job Description
Use Python and SQL to support reporting...
"""


FULL_DESCRIPTION = """Responsibilities
You will analyze customer and product data to explain measurable business outcomes.
You will build and maintain SQL transformations, recurring dashboards, and data-quality checks.
You will collaborate with engineering, product, and operations partners on documented questions.
You will communicate findings, assumptions, and limitations to technical and nontechnical stakeholders.

Requirements
Required experience with Python and SQL for reproducible data analysis and reporting.
Must have experience building dashboards and validating metrics against source systems.
Knowledge of statistics, experiment design, and practical data-quality methods is required.
Ability to translate ambiguous business questions into documented analytical steps is required.
Two years of experience delivering analysis projects or equivalent applied work is preferred.

About the team
The analytics group maintains shared data products used across several business functions.
Team members plan work together, review queries and definitions, document decisions, and improve
recurring workflows. The role has access to established engineering standards, peer review,
privacy-aware data handling practices, professional development support, and regular feedback.
The successful analyst will balance independent investigation with collaboration and will help
the organization make decisions using reliable, clearly explained evidence."""


def candidate(*, company: str = "Example Analytics", role: str = "Data Analyst", source_id: str = "js-1") -> dict[str, object]:
    return {
        "source_job_id": source_id,
        "company": company,
        "role": role,
        "location": "Remote",
        "job_url": "https://jobs.example.com/data-analyst",
        "description": FULL_DESCRIPTION,
        "requirements": "",
        "salary": "",
        "source": "jsearch",
    }


class FullJDEnrichmentTests(unittest.TestCase):
    def test_strict_match_replaces_snippet_and_preserves_metadata(self) -> None:
        calls: list[tuple[str, str, str, int]] = []

        def search(country: str, query: str, location: str, limit: int):
            calls.append((country, query, location, limit))
            return [candidate()]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(path, search_jobs=search, configured=lambda: True)
            updated_text = path.read_text(encoding="utf-8")

        self.assertTrue(result["updated"])
        self.assertEqual(result["status"], "updated")
        self.assertEqual(calls[0][0], "us")
        self.assertIn("Data Analyst at Example Analytics", calls[0][1])
        self.assertIn("Company Confirmed By User: yes", updated_text)
        self.assertIn("Description Source: full_jd_api", updated_text)
        self.assertIn("JD Enriched By: JSearch", updated_text)
        self.assertNotIn("support reporting...", updated_text)
        self.assertTrue(classify_jd_quality(updated_text)["reliable_scoring_ready"])

    def test_company_mismatch_does_not_modify_saved_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=lambda *_args: [candidate(company="Different Corporation")],
                configured=lambda: True,
            )
            final_text = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "no_safe_match")
        self.assertFalse(result["updated"])
        self.assertEqual(final_text, TARGET_TEXT)

    def test_ambiguous_matches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=lambda *_args: [candidate(source_id="js-1"), candidate(source_id="js-2")],
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "ambiguous_match")
        self.assertFalse(result["updated"])

    def test_ambiguous_matches_without_source_ids_fail_closed(self) -> None:
        first = candidate(source_id="")
        second = candidate(source_id="")
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=lambda *_args: [first, second],
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "ambiguous_match")
        self.assertFalse(result["updated"])

    def test_missing_api_configuration_does_not_search_or_write(self) -> None:
        search_called = False

        def search(*_args):
            nonlocal search_called
            search_called = True
            return [candidate()]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(path, search_jobs=search, configured=lambda: False)

        self.assertEqual(result["status"], "not_configured")
        self.assertFalse(search_called)

    def test_manual_record_sync_preserves_enriched_description(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            jobs_dir = root / "manual_jobs"
            index_path = jobs_dir / "manual_jobs.jsonl"
            saved_dir = jobs_dir / "saved_jobs"
            uploads_dir = jobs_dir / "uploads"
            saved_dir.mkdir(parents=True)
            index_path.write_text(
                json.dumps({"id": "manual-1", "job_description": "Old snippet", "updated_at": "old"}) + "\n",
                encoding="utf-8",
            )
            markdown_path = saved_dir / "job.md"
            markdown_path.write_text(
                TARGET_TEXT.replace("api_snippet", "full_jd_api")
                .replace("snippet_only", "complete")
                .replace(
                    "Company Confirmed By User: yes",
                    "Company Confirmed By User: yes\nJD Enriched By: JSearch\nJD Enrichment Match: 96%",
                )
                .replace("Use Python and SQL to support reporting...", FULL_DESCRIPTION)
                + "\n",
                encoding="utf-8",
            )

            with (
                patch.object(manual_jobs, "MANUAL_JOBS_DIR", jobs_dir),
                patch.object(manual_jobs, "MANUAL_JOBS_JSONL", index_path),
                patch.object(manual_jobs, "MANUAL_SAVED_JOBS_DIR", saved_dir),
                patch.object(manual_jobs, "MANUAL_UPLOADS_DIR", uploads_dir),
            ):
                updated = manual_jobs.sync_manual_job_from_markdown("manual-1", markdown_path)
                stored = json.loads(index_path.read_text(encoding="utf-8").strip())

        self.assertIsNotNone(updated)
        self.assertIn("analyze customer and product data", stored["job_description"])
        self.assertEqual(stored["description_source"], "full_jd_api")
        self.assertEqual(stored["jd_fetch_status"], "complete")
        self.assertEqual(stored["jd_enriched_by"], "JSearch")


if __name__ == "__main__":
    unittest.main()
