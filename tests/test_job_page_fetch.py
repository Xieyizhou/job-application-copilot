"""Tests for direct original-job-page extraction."""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from job_page_fetch import JobPageFetchError, extract_job_page, fetch_job_page


DESCRIPTION = """Responsibilities
You will build production data pipelines, maintain reliable analytics systems, and collaborate
with product and engineering teams. You will document decisions and communicate results.

Requirements
Required experience with Python and SQL for data analysis and reporting. Must have experience
building dashboards and data-quality checks. Knowledge of statistics and experiment design is
required. Ability to translate business questions into documented analytical work is required.
Candidates need two years of relevant experience or equivalent applied project work.
"""


class JobPageExtractionTests(unittest.TestCase):
    def test_prefers_jobposting_json_ld(self) -> None:
        posting = {
            "@context": "https://schema.org",
            "@type": "JobPosting",
            "title": "Data Analyst",
            "hiringOrganization": {"@type": "Organization", "name": "Example Analytics"},
            "description": DESCRIPTION.replace("\n", "<br>"),
        }
        page = extract_job_page(
            f'<html><script type="application/ld+json">{json.dumps(posting)}</script></html>',
            "https://jobs.example.com/123",
        )

        self.assertEqual(page.extractor, "jobposting_json_ld")
        self.assertEqual(page.title, "Data Analyst")
        self.assertEqual(page.company, "Example Analytics")
        self.assertIn("Required experience with Python", page.description)

    def test_uses_visible_job_description_container_when_json_ld_is_absent(self) -> None:
        body = " ".join(DESCRIPTION.split())
        page = extract_job_page(
            f'<main><div class="job-description">{body}</div></main>',
            "https://jobs.example.com/123",
        )

        self.assertEqual(page.extractor, "job_page_container")
        self.assertIn("data-quality checks", page.description)

    @patch("job_page_fetch._public_http_url", return_value=True)
    @patch("job_page_fetch.requests.get")
    def test_fetch_rejects_non_html_response(self, get: Mock, _public: Mock) -> None:
        response = Mock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.headers = {"Content-Type": "application/pdf"}
        response.raise_for_status.return_value = None
        get.return_value = response

        with self.assertRaises(JobPageFetchError):
            fetch_job_page("https://jobs.example.com/123")

    @patch("job_page_fetch._public_http_url", return_value=True)
    @patch("job_page_fetch.requests.get")
    def test_fetch_explains_when_a_site_blocks_automation(self, get: Mock, _public: Mock) -> None:
        response = Mock()
        response.is_redirect = False
        response.is_permanent_redirect = False
        response.status_code = 403
        get.return_value = response

        with self.assertRaisesRegex(JobPageFetchError, "blocked the server fetch"):
            fetch_job_page("https://jobs.example.com/123")


if __name__ == "__main__":
    unittest.main()
