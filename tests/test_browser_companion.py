"""Tests for the authenticated loopback browser companion."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from browser_companion import (
    CompanionConfig,
    import_browser_capture,
    load_or_create_token,
    resolve_saved_job,
    start_browser_companion,
)


FULL_DESCRIPTION = """Responsibilities
Build and maintain reliable machine learning training and evaluation pipelines.
Deploy versioned models and monitor their behavior under real production traffic.
Collaborate with product, data, and infrastructure teams on measurable outcomes.
Document experiments, limitations, operational decisions, and model-quality results.

Requirements
Required experience with Python and machine learning model development.
Must have experience evaluating models with reproducible metrics and test datasets.
Knowledge of data pipelines, production inference, and observability is required.
Ability to communicate technical trade-offs to cross-functional partners is required.
Two years of applied machine learning experience or equivalent project work is preferred.

About the team
The applied machine learning group owns shared training, evaluation, and deployment systems.
Engineers work with product partners to define success metrics, review experiments, diagnose
production failures, and improve recurring model-delivery workflows. The role combines
independent technical ownership with peer review, privacy-aware data handling, operational
documentation, and clear communication about uncertainty. Team members also maintain common
libraries, incident playbooks, and quality dashboards used across multiple production services.
"""


def saved_job_text(url: str = "https://jobs.example.com/ml-engineer") -> str:
    return f"""# Machine Learning Engineer
Company: Example AI
Company Normalized: Example AI
Role: Machine Learning Engineer
Location: Remote
Job URL: {url}
Source: Jooble
Description Source: api_snippet
JD Fetch Status: snippet_only

## Job Description
Build machine learning systems...
"""


class BrowserCompanionTests(unittest.TestCase):
    def test_load_or_create_token_is_stable_and_private(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "token"
            first = load_or_create_token(path)
            second = load_or_create_token(path)
            mode = os.stat(path).st_mode & 0o777
        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 24)
        self.assertEqual(mode, 0o600)

    def test_resolve_saved_job_prefers_exact_canonical_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            path = jobs_dir / "source" / "job.md"
            path.parent.mkdir(parents=True)
            path.write_text(saved_job_text("https://jobs.example.com/ml-engineer?utm_source=x"), encoding="utf-8")
            resolved = resolve_saved_job(
                jobs_dir,
                page_url="https://jobs.example.com/ml-engineer",
                title="Different title",
                company="Different company",
            )
        self.assertEqual(resolved, path)

    def test_import_replaces_only_a_matching_saved_job(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            path = jobs_dir / "job.md"
            path.write_text(saved_job_text(), encoding="utf-8")
            result = import_browser_capture(
                CompanionConfig(jobs_dir=jobs_dir, token="local-token"),
                {
                    "url": "https://jobs.example.com/ml-engineer",
                    "title": "Machine Learning Engineer",
                    "company": "Example AI",
                    "description": FULL_DESCRIPTION,
                    "extractor": "browser_jobposting_json_ld",
                },
            )
            updated = path.read_text(encoding="utf-8")
        self.assertTrue(result["ok"])
        self.assertIn("Description Source: browser_companion", updated)
        self.assertIn("JD Enriched By: Local browser companion", updated)
        self.assertIn(FULL_DESCRIPTION, updated)

    def test_import_rejects_an_unsaved_page(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            result = import_browser_capture(
                CompanionConfig(jobs_dir=Path(temp_dir), token="local-token"),
                {
                    "url": "https://jobs.example.com/unknown",
                    "title": "Machine Learning Engineer",
                    "company": "Example AI",
                    "description": FULL_DESCRIPTION,
                },
            )
        self.assertEqual(result["status"], "saved_job_not_found")
        self.assertFalse(result["ok"])

    def test_http_import_requires_token_and_returns_verified_result(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_dir = Path(temp_dir)
            path = jobs_dir / "job.md"
            path.write_text(saved_job_text(), encoding="utf-8")
            companion = start_browser_companion(jobs_dir, port=0, token="test-token-value")
            endpoint = f"http://127.0.0.1:{companion.port}/v1/import"
            body = json.dumps(
                {
                    "url": "https://jobs.example.com/ml-engineer",
                    "title": "Machine Learning Engineer",
                    "company": "Example AI",
                    "description": FULL_DESCRIPTION,
                }
            ).encode("utf-8")
            try:
                with self.assertRaises(HTTPError) as error_context:
                    urlopen(Request(endpoint, data=body, method="POST"), timeout=2)
                self.assertEqual(error_context.exception.code, 401)
                request = Request(
                    endpoint,
                    data=body,
                    method="POST",
                    headers={
                        "Content-Type": "application/json",
                        "X-JobCopilot-Token": "test-token-value",
                    },
                )
                with urlopen(request, timeout=2) as response:
                    result = json.loads(response.read().decode("utf-8"))
            finally:
                companion.stop()
        self.assertTrue(result["ok"])


if __name__ == "__main__":
    unittest.main()
