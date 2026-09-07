"""Regression tests for full-job-description search and evidence preference."""

from __future__ import annotations

import dashboard_repository

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import fetch_jobs


class JSearchNormalizationTests(unittest.TestCase):
    def test_normalizes_full_description_and_structured_requirements(self) -> None:
        job = fetch_jobs.normalize_jsearch_job(
            {
                "job_id": "job-123",
                "employer_name": "Example Analytics",
                "job_title": "Data Analyst",
                "job_city": "Toronto",
                "job_country": "Canada",
                "job_apply_link": "https://careers.example.com/jobs/123?utm_source=test",
                "job_description": "<p>Build reliable analytics products with Python and SQL.</p>",
                "job_highlights": {
                    "Qualifications": ["Python and SQL", "Clear communication"],
                    "Responsibilities": ["Build dashboards"],
                },
            },
            "Canada",
        )

        self.assertEqual(job["source"], "jsearch")
        self.assertEqual(job["description_source"], "full_jd_api")
        self.assertEqual(job["jd_fetch_status"], "complete")
        self.assertEqual(job["description"], "Build reliable analytics products with Python and SQL.")
        self.assertIn("Clear communication", job["requirements"])
        self.assertNotIn("utm_source", job["job_url"])
        self.assertEqual(job["discovery_url"], "")

    @patch("fetch_jobs.load_jsearch_api_key", return_value="test-key")
    @patch("requests.get")
    def test_fetch_uses_full_description_endpoint(self, get: Mock, _load_key: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": [
                {
                    "job_id": "job-123",
                    "employer_name": "Example Analytics",
                    "job_title": "Data Analyst",
                    "job_country": "United States",
                    "job_apply_link": "https://careers.example.com/jobs/123",
                    "job_description": " ".join(["Python SQL analytics communication"] * 40),
                }
            ]
        }
        get.return_value = response

        jobs = fetch_jobs.fetch_jsearch_jobs("us", "data analyst", "Remote", 5)

        self.assertEqual(len(jobs), 1)
        self.assertGreater(len(jobs[0]["description"].split()), 100)
        _, kwargs = get.call_args
        self.assertEqual(kwargs["headers"], {"x-api-key": "test-key"})
        self.assertEqual(kwargs["params"]["country"], "us")

    @patch("fetch_jobs.load_jsearch_api_key", return_value="test-key")
    @patch("requests.get")
    def test_fetch_accepts_current_v2_jobs_envelope(self, get: Mock, _load_key: Mock) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "data": {
                "jobs": [
                    {
                        "job_id": "job-v2",
                        "employer_name": "Example Robotics",
                        "job_title": "Machine Learning Engineer",
                        "job_country": "Singapore",
                        "job_apply_link": "https://careers.example.com/jobs/v2",
                        "job_description": " ".join(["Python machine learning systems"] * 40),
                    }
                ],
                "cursor": "next-page-token",
            }
        }
        get.return_value = response

        jobs = fetch_jobs.fetch_jsearch_jobs("sg", "machine learning engineer", "Singapore", 5)

        self.assertEqual(len(jobs), 1)
        self.assertEqual(jobs[0]["role"], "Machine Learning Engineer")
        self.assertEqual(jobs[0]["location"], "Singapore")

    @patch("fetch_jobs.load_jsearch_api_key", return_value="test-key")
    @patch("requests.get")
    def test_fetch_distinguishes_no_results_from_missing_descriptions(
        self, get: Mock, _load_key: Mock
    ) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        get.return_value = response
        response.json.return_value = {"data": []}
        with self.assertRaises(fetch_jobs.JSearchNoResultsError):
            fetch_jobs.fetch_jsearch_jobs("sg", "rare role", "Singapore", 5)

        response.json.return_value = {
            "data": [
                {
                    "job_id": "summary-only",
                    "employer_name": "Example",
                    "job_title": "Rare role",
                }
            ]
        }
        with self.assertRaises(fetch_jobs.JSearchNoFullDescriptionsError):
            fetch_jobs.fetch_jsearch_jobs("sg", "rare role", "Singapore", 5)

    def test_markdown_records_description_provenance(self) -> None:
        markdown = fetch_jobs.build_job_markdown(
            fetch_jobs.normalize_jsearch_job(
                {
                    "job_id": "job-123",
                    "employer_name": "Example Analytics",
                    "job_title": "Data Analyst",
                    "job_country": "Canada",
                    "job_apply_link": "https://careers.example.com/jobs/123",
                    "job_description": "Full description with Python, SQL, and reporting responsibilities.",
                },
                "Canada",
            )
        )
        self.assertIn("Description Source: full_jd_api", markdown)
        self.assertIn("JD Fetch Status: complete", markdown)
        self.assertIn("Discovery URL: https://careers.example.com/jobs/123", markdown)


class DashboardEvidencePreferenceTests(unittest.TestCase):
    def test_duplicate_resolution_prefers_full_jd_over_snippet(self) -> None:
        shared = {
            "company": "Example Analytics",
            "role": "Data Analyst",
            "location": "Remote",
            "job_url": "",
        }
        snippet = {
            **shared,
            "source": "jooble",
            "description_source": "api_snippet",
            "jd_fetch_status": "snippet_only",
            "description_word_count": 35,
        }
        full = {
            **shared,
            "source": "jsearch",
            "description_source": "full_jd_api",
            "jd_fetch_status": "complete",
            "description_word_count": 480,
        }

        unique = dashboard_repository.deduplicate_dashboard_jobs([snippet, full])

        self.assertEqual(len(unique), 1)
        self.assertEqual(unique[0]["source"], "jsearch")

    def test_identical_snippet_syndication_collapses_across_locations(self) -> None:
        shared = {
            "company": "Example Analytics",
            "role": "Data Analyst",
            "job_url": "",
            "source": "jooble",
            "description_source": "api_snippet",
            "jd_fetch_status": "snippet_only",
            "description_fingerprint": "same-preview",
        }

        unique = dashboard_repository.deduplicate_dashboard_jobs(
            [
                {**shared, "location": "Singapore"},
                {**shared, "location": "Remote"},
            ]
        )

        self.assertEqual(len(unique), 1)

    def test_complete_jobs_at_distinct_locations_remain_distinct(self) -> None:
        shared = {
            "company": "Example Analytics",
            "role": "Data Analyst",
            "job_url": "",
            "source": "jsearch",
            "description_source": "full_jd_api",
            "jd_fetch_status": "complete",
            "description_fingerprint": "same-description",
        }

        unique = dashboard_repository.deduplicate_dashboard_jobs(
            [
                {**shared, "location": "Singapore"},
                {**shared, "location": "Canada"},
            ]
        )

        self.assertEqual(len(unique), 2)


class FetchQualityGateTests(unittest.TestCase):
    def test_full_description_is_saved_without_original_url(self) -> None:
        decision = fetch_jobs.should_save_fetched_job(
            {"jd_fetch_status": "complete", "job_url": ""}
        )

        self.assertEqual(decision, (True, "full_description"))

    def test_aggregator_preview_without_original_is_skipped(self) -> None:
        decision = fetch_jobs.should_save_fetched_job(
            {
                "jd_fetch_status": "snippet_only",
                "job_url": "https://www.jooble.org/desc/123",
            }
        )

        self.assertEqual(decision, (False, "preview_without_original"))

    def test_preview_with_employer_url_can_be_recovered(self) -> None:
        decision = fetch_jobs.should_save_fetched_job(
            {
                "jd_fetch_status": "snippet_only",
                "job_url": "https://careers.example.com/jobs/123",
            }
        )

        self.assertEqual(decision, (True, "recoverable_original_url"))


if __name__ == "__main__":
    unittest.main()
