"""Focused tests for read-only dashboard repository operations."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from dashboard_repository import (
    build_dashboard_job_record,
    list_job_description_files,
    load_tracker_rows,
)


class DashboardRepositoryTests(unittest.TestCase):
    def test_job_file_listing_is_workspace_scoped_and_searchable(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            demo_dir = root / "demo"
            jobs_dir = root / "personal"
            demo_dir.mkdir()
            (jobs_dir / "nested").mkdir(parents=True)
            (demo_dir / "analyst.md").write_text("demo", encoding="utf-8")
            (jobs_dir / "nested" / "scientist.md").write_text(
                "personal",
                encoding="utf-8",
            )
            def predicate(path: Path) -> bool:
                return path.is_file() and path.suffix == ".md"

            demo_paths = list_job_description_files(
                demo_mode=True,
                demo_job_dir=demo_dir,
                jobs_dir=jobs_dir,
                is_job_description=predicate,
            )
            personal_paths = list_job_description_files(
                demo_mode=False,
                demo_job_dir=demo_dir,
                jobs_dir=jobs_dir,
                is_job_description=predicate,
                search_text="scientist",
            )

        self.assertEqual([path.name for path in demo_paths], ["analyst.md"])
        self.assertEqual([path.name for path in personal_paths], ["scientist.md"])

    def test_record_builder_keeps_independent_quality_and_risk_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            job_path = Path(temp_dir) / "role.md"
            job_path.write_text(
                "\n".join(
                    [
                        "# Data Analyst",
                        "Company: Example Labs",
                        "Role: Data Analyst",
                        "Location: Remote",
                        "Source: JSearch",
                        "Job URL: https://example.com/jobs/1",
                        "## Job Description",
                        "Analyze data with Python and SQL.",
                    ]
                ),
                encoding="utf-8",
            )
            record = build_dashboard_job_record(
                job_path,
                read_text=lambda path: path.read_text(encoding="utf-8"),
                detect_red_flags=lambda _text: ["Manual constraint review"],
                collect_warnings=lambda _text: ["Partial posting"],
            )

        self.assertEqual(record["company"], "Example Labs")
        self.assertEqual(record["role"], "Data Analyst")
        self.assertEqual(record["source"], "jsearch")
        self.assertEqual(record["red_flags"], ["Manual constraint review"])
        self.assertEqual(record["warnings"], ["Partial posting"])
        self.assertIsNone(record["score"])
        self.assertEqual(record["recommendation"], "Manual Review")

    def test_tracker_query_filters_without_dashboard_session_state(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            database_path = Path(temp_dir) / "applications.db"
            with sqlite3.connect(database_path) as connection:
                connection.execute(
                    """
                    CREATE TABLE applications (
                        id INTEGER PRIMARY KEY,
                        company TEXT,
                        role TEXT,
                        location TEXT,
                        job_url TEXT,
                        match_score INTEGER,
                        recommendation TEXT,
                        status TEXT,
                        resume_file TEXT,
                        cover_letter_file TEXT,
                        notes TEXT,
                        created_at TEXT,
                        applied_date TEXT
                    )
                    """
                )
                connection.executemany(
                    """
                    INSERT INTO applications VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    [
                        (
                            1,
                            "Example Labs",
                            "Analyst",
                            "Remote",
                            "https://example.com/1",
                            80,
                            "Apply",
                            "ready",
                            "",
                            "",
                            "",
                            "2026-01-01",
                            None,
                        ),
                        (
                            2,
                            "Other Labs",
                            "Scientist",
                            "Remote",
                            "https://example.com/2",
                            40,
                            "Manual Review",
                            "saved",
                            "",
                            "",
                            "",
                            "2026-01-02",
                            None,
                        ),
                    ],
                )
            rows = load_tracker_rows(
                database_path,
                statuses=["ready"],
                minimum_score=70,
                company_search="example",
            )

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["company"], "Example Labs")
        self.assertEqual(rows[0]["match_score"], 80)


if __name__ == "__main__":
    unittest.main()
