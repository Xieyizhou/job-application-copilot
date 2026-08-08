"""Regression tests for safe full-JD lookup and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fetch_jobs import JSearchNoResultsError
from jd_enrichment import (
    enrich_saved_job_description,
    enrich_saved_job_description_from_url,
    replace_saved_job_description,
)
from job_page_fetch import FetchedJobPage
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
    def test_original_url_fetch_replaces_snippet_and_records_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description_from_url(
                path,
                fetch_page=lambda url: FetchedJobPage(
                    description=FULL_DESCRIPTION,
                    source_url=url,
                    extractor="jobposting_json_ld",
                    title="Data Analyst",
                    company="Example Analytics",
                ),
            )
            updated_text = path.read_text(encoding="utf-8")

        self.assertTrue(result["updated"])
        self.assertIn("Description Source: company_site", updated_text)
        self.assertIn("JD Enriched By: Original job URL", updated_text)
        self.assertIn("JD Enrichment Extractor: jobposting_json_ld", updated_text)

    def test_original_url_identity_mismatch_leaves_file_unchanged(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description_from_url(
                path,
                fetch_page=lambda url: FetchedJobPage(
                    description=FULL_DESCRIPTION,
                    source_url=url,
                    extractor="jobposting_json_ld",
                    title="Nurse Practitioner",
                    company="Different Health",
                ),
            )
            final_text = path.read_text(encoding="utf-8")

        self.assertEqual(result["status"], "page_identity_mismatch")
        self.assertFalse(result["updated"])
        self.assertEqual(final_text, TARGET_TEXT)

    def test_manual_full_jd_is_validated_before_replacement(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            rejected = replace_saved_job_description(path, "A short job summary.")
            self.assertFalse(rejected["updated"])
            self.assertEqual(path.read_text(encoding="utf-8"), TARGET_TEXT)

            accepted = replace_saved_job_description(path, FULL_DESCRIPTION)
            updated_text = path.read_text(encoding="utf-8")

        self.assertTrue(accepted["updated"])
        self.assertIn("Description Source: manual_full_jd", updated_text)
        self.assertIn("JD Enriched By: User verified paste", updated_text)

    def test_reuses_matching_local_jsearch_full_jd_before_calling_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_root = Path(temp_dir) / "jobs"
            target_path = jobs_root / "adzuna" / "target.md"
            candidate_path = jobs_root / "jsearch" / "full.md"
            target_path.parent.mkdir(parents=True)
            candidate_path.parent.mkdir(parents=True)
            target_path.write_text(
                TARGET_TEXT.replace("Example Analytics", "Hewlett Packard Enterprise")
                .replace("Data Analyst", "Machine Learning Graduate"),
                encoding="utf-8",
            )
            candidate_path.write_text(
                TARGET_TEXT.replace("Example Analytics", "HPE")
                .replace("Data Analyst", "Machine Learning Graduate")
                .replace("Source: Jooble", "Source: JSearch")
                .replace("Description Source: api_snippet", "Description Source: full_jd_api")
                .replace("JD Fetch Status: snippet_only", "JD Fetch Status: complete")
                .replace("Use Python and SQL to support reporting...", FULL_DESCRIPTION)
                .replace(
                    "Company Confirmed By User: yes",
                    "Company Confirmed By User: yes\nSource Job ID: local-jsearch-1",
                ),
                encoding="utf-8",
            )

            result = enrich_saved_job_description(
                target_path,
                search_jobs=lambda *_args: self.fail("provider should not be called"),
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "updated")
        self.assertTrue(result["updated"])

    def test_broadens_query_after_exact_search_has_no_results(self) -> None:
        calls: list[tuple[str, str, str, int]] = []

        def search(country: str, query: str, location: str, limit: int):
            calls.append((country, query, location, limit))
            if len(calls) == 1:
                raise JSearchNoResultsError("no exact results")
            return [candidate()]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=search,
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "updated")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[1][1], "Data Analyst")

    def test_search_attempts_include_company_jobs_fallback(self) -> None:
        calls: list[tuple[str, str, str, int]] = []

        def search(country: str, query: str, location: str, limit: int):
            calls.append((country, query, location, limit))
            if query != "jobs at Example Analytics":
                raise JSearchNoResultsError("not found")
            return [candidate()]

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=search,
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "updated")
        self.assertIn("jobs at Example Analytics", [call[1] for call in calls])

    def test_snippet_overlap_recovers_a_rewritten_title_at_same_company(self) -> None:
        target = (
            TARGET_TEXT.replace("Example Analytics", "Motional")
            .replace("Data Analyst", "Machine Learning Internship, Behaviors Research")
            .replace(
                "Use Python and SQL to support reporting...",
                " ".join(FULL_DESCRIPTION.split()[:55]),
            )
        )
        rewritten = candidate(
            company="Motional",
            role="2026 Summer Intern - PhD - Behaviors",
        )
        rewritten["job_url"] = "https://careers.example.com/rewritten-title"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(target, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=lambda *_args: [rewritten],
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "updated")
        self.assertGreaterEqual(result["match_components"]["description"], 0.7)

    def test_unrelated_titles_do_not_create_a_false_related_status(self) -> None:
        unrelated = candidate(
            company="Unrelated Company",
            role="Senior Machine Learning Scientist",
        )
        unrelated["job_url"] = "https://jobs.example.com/unrelated"
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(TARGET_TEXT, encoding="utf-8")
            result = enrich_saved_job_description(
                path,
                search_jobs=lambda *_args: [unrelated],
                configured=lambda: True,
            )

        self.assertEqual(result["status"], "posting_not_found")

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
        self.assertEqual(calls[0][1], "Data Analyst Example Analytics")
        self.assertEqual(calls[0][2], "Remote")
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
