"""Tests for one-click public ATS JD completion."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ats_jd_completion import (
    ats_completion_progress,
    complete_public_ats_jobs,
    public_ats_completion_candidates,
)


INCOMPLETE = """# Data Engineer
Company: Example Analytics
Role: Data Engineer
Job URL: {url}
Description Source: api_snippet
JD Fetch Status: snippet_only

## Job Description
Build data products...
"""


class PublicATSCompletionTests(unittest.TestCase):
    def test_candidates_include_only_incomplete_supported_urls(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            supported = root / "supported.md"
            unsupported = root / "unsupported.md"
            supported.write_text(
                INCOMPLETE.format(url="https://jobs.ashbyhq.com/acme/job-1"),
                encoding="utf-8",
            )
            unsupported.write_text(
                INCOMPLETE.format(url="https://www.linkedin.com/jobs/view/1"),
                encoding="utf-8",
            )

            candidates = public_ats_completion_candidates([supported, unsupported])

        self.assertEqual(candidates, [supported])

    def test_batch_reports_updates_and_unsupported_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            supported = root / "supported.md"
            unsupported = root / "unsupported.md"
            supported.write_text(
                INCOMPLETE.format(url="https://jobs.lever.co/acme/job-1"),
                encoding="utf-8",
            )
            unsupported.write_text(
                INCOMPLETE.format(url="https://example.com/job-2"), encoding="utf-8"
            )

            result = complete_public_ats_jobs(
                [supported, unsupported],
                enrich=lambda path: {"updated": path == supported, "status": "updated"},
            )

        self.assertEqual(result["eligible"], 1)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["unsupported"], 1)
        self.assertEqual(result["providers"]["Lever"]["updated"], 1)

    def test_batch_can_resolve_an_aggregator_record(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "aggregator.md"
            path.write_text(
                INCOMPLETE.format(url="https://jooble.org/job/123"), encoding="utf-8"
            )
            result = complete_public_ats_jobs(
                [path],
                resolve=lambda _path: {"updated": True, "status": "updated"},
            )

        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["resolved_links"], 1)
        self.assertEqual(result["unsupported"], 0)

    def test_resolvable_candidates_include_incomplete_aggregator_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "job.md"
            path.write_text(
                INCOMPLETE.format(url="https://jooble.org/job/123"), encoding="utf-8"
            )
            candidates = public_ats_completion_candidates(
                [path], include_resolvable=True
            )
        self.assertEqual(candidates, [path])

    def test_batch_limit_rotates_to_unattempted_jobs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs = Path(temp_dir) / "jobs" / "jooble"
            jobs.mkdir(parents=True)
            paths = []
            for index in range(12):
                path = jobs / f"{index:02d}.md"
                path.write_text(
                    INCOMPLETE.format(url=f"https://jooble.org/job/{index}"),
                    encoding="utf-8",
                )
                paths.append(path)
            calls: list[Path] = []

            def unresolved(path: Path) -> dict[str, object]:
                calls.append(path)
                return {"updated": False, "status": "not_found", "message": "No match"}

            first = complete_public_ats_jobs(paths, resolve=unresolved, max_jobs=10)
            progress = ats_completion_progress(paths)
            second = complete_public_ats_jobs(paths, resolve=unresolved, max_jobs=10)

        self.assertEqual(first["attempted"], 10)
        self.assertEqual(first["deferred"], 2)
        self.assertEqual(progress["pending"], 2)
        self.assertEqual(progress["failed"], 10)
        self.assertEqual(second["attempted"], 2)
        self.assertEqual(calls[-2:], paths[-2:])


if __name__ == "__main__":
    unittest.main()
