"""Tests for the extracted fetch-history presentation."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard_fetch_history as fetch_history_ui


class DashboardFetchHistoryTests(unittest.TestCase):
    def test_labels_and_rows_preserve_fetch_run_counts(self) -> None:
        run = {
            "created_at": "sample-run",
            "source": "jsearch",
            "region": "Remote",
            "query": "Data Analyst",
            "total_jobs_returned": 12,
            "new_jobs_count": 4,
            "duplicate_jobs_count": 8,
            "fetch_status": "success",
        }

        self.assertEqual(
            fetch_history_ui.fetch_run_label(run),
            "sample-run | JSearch · Full JD | Remote | Data Analyst | 4 new",
        )
        self.assertEqual(
            fetch_history_ui.fetch_history_rows([run]),
            [
                {
                    "Date/time": "sample-run",
                    "Source": "JSearch · Full JD",
                    "Region": "Remote",
                    "Query": "Data Analyst",
                    "Total returned": 12,
                    "New jobs": 4,
                    "Already seen": 8,
                    "Status": "success",
                }
            ],
        )

    def test_job_rows_normalize_title_location_and_saved_state(self) -> None:
        jobs = [
            {
                "company": "Example Labs",
                "role": "Data Analyst",
                "location": "remote",
                "source": "jsearch",
                "path": "data/jobs/example.md",
            }
        ]

        self.assertEqual(
            fetch_history_ui.fetch_run_job_rows(jobs),
            [
                {
                    "Company": "Example Labs",
                    "Role": "Data Analyst",
                    "Location": "remote",
                    "Source": "JSearch · Full JD",
                    "Saved": "Yes",
                }
            ],
        )

    def test_history_selection_renders_the_selected_run(self) -> None:
        runs = [
            {
                "created_at": "newest-run",
                "source": "jsearch",
                "region": "Remote",
                "query": "Data Analyst",
                "new_jobs_count": 2,
            },
            {
                "created_at": "older-run",
                "source": "jooble",
                "region": "Canada",
                "query": "ML Engineer",
                "new_jobs_count": 1,
            },
        ]
        ui = MagicMock()
        ui.selectbox.return_value = fetch_history_ui.fetch_run_label(runs[1])

        with (
            patch.object(fetch_history_ui, "st", ui),
            patch.object(fetch_history_ui, "load_fetch_runs", return_value=runs),
            patch.object(fetch_history_ui, "render_fetch_run_details") as render_details,
        ):
            fetch_history_ui.render_fetch_history_section()

        render_details.assert_called_once_with(runs[1])
        ui.dataframe.assert_called_once()

    def test_empty_history_and_empty_job_table_are_explicit(self) -> None:
        ui = MagicMock()
        with (
            patch.object(fetch_history_ui, "st", ui),
            patch.object(fetch_history_ui, "load_fetch_runs", return_value=[]),
        ):
            fetch_history_ui.render_fetch_history_section()
            fetch_history_ui.render_fetch_run_job_table([], "No jobs.")

        self.assertEqual(
            [call.args[0] for call in ui.info.call_args_list],
            ["No fetch history yet.", "No jobs."],
        )

    def test_job_cards_render_saved_preview_and_original_link(self) -> None:
        ui = MagicMock()
        jobs = [
            {
                "company": "Example Labs",
                "role": "Data Analyst",
                "location": "Remote",
                "source": "jsearch",
                "job_url": "https://example.test/jobs/sample",
                "path": "saved-job.md",
            }
        ]

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "saved-job.md").write_text("Saved job preview", encoding="utf-8")
            with (
                patch.object(fetch_history_ui, "st", ui),
                patch.object(fetch_history_ui, "PROJECT_ROOT", root),
            ):
                fetch_history_ui.render_fetch_run_job_cards(jobs, "No jobs.")

        ui.link_button.assert_called_once_with(
            "Open original posting",
            "https://example.test/jobs/sample",
        )
        ui.markdown.assert_any_call("Saved job preview")

    def test_run_details_keeps_new_and_seen_jobs_separate(self) -> None:
        ui = MagicMock()
        run = {
            "source": "jsearch",
            "region": "Remote",
            "query": "Data Analyst",
            "total_jobs_returned": 3,
            "new_jobs_count": 1,
            "duplicate_jobs_count": 2,
            "fetch_status": "partial",
            "notes": "One provider warning.",
            "new_jobs": [{"company": "New Company"}],
            "previously_seen_jobs": [{"company": "Seen Company"}],
        }

        with (
            patch.object(fetch_history_ui, "st", ui),
            patch.object(fetch_history_ui, "render_fetch_run_job_cards") as cards,
            patch.object(fetch_history_ui, "render_fetch_run_job_table") as table,
        ):
            fetch_history_ui.render_fetch_run_details(run)
            fetch_history_ui.render_fetch_run_details({})

        self.assertEqual(cards.call_count, 2)
        cards.assert_any_call(
            run["new_jobs"],
            "No new jobs were discovered in this search.",
        )
        cards.assert_any_call(
            run["previously_seen_jobs"],
            "No already seen jobs were returned.",
        )
        table.assert_called_once_with(
            run["new_jobs"],
            "No new jobs were discovered in this search.",
        )
        ui.warning.assert_called_once_with("One provider warning.")


if __name__ == "__main__":
    unittest.main()
