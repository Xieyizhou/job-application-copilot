"""Lifecycle tests for the local SQLite application tracker."""

from __future__ import annotations

import io
import sqlite3
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

from tracker import (
    add_application,
    build_parser,
    delete_application,
    list_applications,
    sanitize_job_url,
    show_application,
    update_status,
    validate_status,
)


def application_args(**overrides: object) -> SimpleNamespace:
    values: dict[str, object] = {
        "company": "Example Analytics",
        "role": "Data Analyst",
        "location": "Remote",
        "job_url": "https://jobs.example.test/roles/7?utm_source=mail&ref=direct",
        "match_score": 82,
        "recommendation": "Apply",
        "status": "saved",
        "resume_file": "resume.pdf",
        "cover_letter_file": "cover-letter.docx",
        "notes": "Review manually.",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class TrackerLifecycleTests(unittest.TestCase):
    def test_add_deduplicate_update_show_list_and_delete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "applications.db"
            output = io.StringIO()
            with redirect_stdout(output):
                application_id = add_application(application_args(), database_path)
                duplicate_id = add_application(application_args(), database_path)
                update_status(application_id, "applied", database_path)
                show_application(application_id, database_path)
                list_applications(database_path)
            self.assertEqual(application_id, duplicate_id)
            text = output.getvalue()
            self.assertIn("Application already exists", text)
            self.assertIn("Status: applied", text)
            self.assertIn("Example Analytics", text)

            with sqlite3.connect(database_path) as connection:
                row = connection.execute(
                    "SELECT job_url, status, applied_date FROM applications WHERE id = ?",
                    (application_id,),
                ).fetchone()
            assert row is not None
            self.assertEqual(row[0], "https://jobs.example.test/roles/7?ref=direct")
            self.assertEqual(row[1], "applied")
            self.assertTrue(row[2])

            with redirect_stdout(io.StringIO()):
                delete_application(application_id, database_path)
            with sqlite3.connect(database_path) as connection:
                count = connection.execute("SELECT COUNT(*) FROM applications").fetchone()[0]
            self.assertEqual(count, 0)

    def test_missing_records_and_invalid_status_fail_safely(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database_path = Path(directory) / "applications.db"
            output = io.StringIO()
            with redirect_stdout(output):
                list_applications(database_path)
                show_application(999, database_path)
                update_status(999, "ready", database_path)
                delete_application(999, database_path)
            text = output.getvalue()
            self.assertIn("No applications found", text)
            self.assertIn("No application found with id 999", text)
        with self.assertRaises(ValueError):
            validate_status("submitted-automatically")

    def test_url_sanitization_and_parser_contract(self) -> None:
        self.assertEqual(
            sanitize_job_url("https://www.adzuna.com/land/ad/12345?aztt=secret#details"),
            "https://www.adzuna.com/details/12345",
        )
        self.assertEqual(sanitize_job_url(""), "")
        parser = build_parser()
        parsed = parser.parse_args(
            ["add", "--company", "Example", "--role", "Analyst", "--status", "ready"]
        )
        self.assertEqual(parsed.command, "add")
        self.assertEqual(parsed.status, "ready")


if __name__ == "__main__":
    unittest.main()
