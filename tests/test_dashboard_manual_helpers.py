"""Behavior tests for the extracted manual-job Dashboard helpers."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, Mock, patch

import dashboard_manual
from manual_jobs import ExtractionResult


class UploadedFile:
    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self.payload = payload
        self.size = len(payload)

    def getvalue(self) -> bytes:
        return self.payload


class DashboardManualHelperTests(unittest.TestCase):
    def test_suggestions_populate_only_empty_reviewable_fields(self) -> None:
        state = {"manual_company": "Accepted Company", "manual_title": ""}
        suggestions = {
            "company": "Parser Company",
            "company_confidence": "high",
            "title": "Data Analyst",
            "job_title_confidence": "medium",
            "source": "Unknown source",
            "source_confidence": "high",
        }
        with patch.object(dashboard_manual.st, "session_state", state):
            dashboard_manual.apply_suggestions_to_empty_fields(suggestions)
            self.assertEqual(state["manual_company"], "Accepted Company")
            self.assertEqual(state["manual_title"], "Data Analyst")
            self.assertNotIn("manual_source", state)
            self.assertTrue(dashboard_manual.form_field_needs_suggestion("manual_title", "Other"))
            dashboard_manual.apply_suggestion_to_form_field("manual_location", " Remote ")
            self.assertEqual(state["manual_location"], "Remote")

    def test_readiness_and_red_flags_reflect_accepted_state(self) -> None:
        long_jd = " ".join(["Python SQL analytics requirement"] * 25)
        suggestions = {
            "title": "Data Analyst",
            "job_title_confidence": "high",
            "company": "Example",
            "location": "Remote",
            "location_options": ["Remote"],
            "visa_note": "",
        }
        state = {
            "manual_company": "Example",
            "manual_title": "Data Analyst",
            "manual_location": "Remote",
            "manual_url": "https://jobs.example.test/7",
            "manual_visa_note": "",
        }
        with patch.object(dashboard_manual.st, "session_state", state):
            warnings = dashboard_manual.build_manual_red_flags(
                suggestions,
                url=state["manual_url"],
                job_description=long_jd,
                reports=[],
            )
            self.assertEqual(warnings, [])
            self.assertEqual(
                dashboard_manual.match_readiness_for(suggestions, warnings, long_jd),
                ("Ready to save", ""),
            )
            state["manual_url"] = ""
            self.assertEqual(
                dashboard_manual.match_readiness_for(suggestions, [], long_jd)[0],
                "Needs review",
            )

    def test_metadata_labels_and_upload_combination(self) -> None:
        reports = [
            {"metadata_title": " Example Role "},
            {"metadata_title": "Example Role"},
            {"metadata_title": "Second Role"},
        ]
        self.assertEqual(
            dashboard_manual.manual_source_metadata_from_reports(reports),
            {"metadata_titles": ["Example Role", "Second Role"]},
        )
        record = {"company": "Example", "title": "Analyst", "created_at": "2026", "id": "7"}
        self.assertEqual(dashboard_manual.manual_record_label(record), "Example | Analyst | 2026 | 7")
        self.assertEqual(dashboard_manual.compact_location_value("Multiple offices: A; B"), "Multiple offices")
        self.assertEqual(dashboard_manual.format_confidence("MEDIUM"), "medium")
        self.assertEqual(dashboard_manual.split_suggestion_lines("A\n\nB"), ["A", "B"])

        uploads = [UploadedFile("one.txt", b"Easy Apply\nRole: Analyst"), UploadedFile("two.txt", b"SQL required")]
        results = [
            ExtractionResult("Easy Apply\nRole: Analyst", warning="Review", report={"method": "text"}),
            ExtractionResult("SQL required"),
        ]
        with patch.object(dashboard_manual, "extract_text_from_upload", side_effect=results):
            raw, cleaned, messages, filenames, extraction_reports = (
                dashboard_manual.combine_upload_extraction_results(uploads)
            )
        self.assertIn("Role: Analyst", raw)
        self.assertNotIn("Easy Apply", cleaned)
        self.assertIn("one.txt: Review", messages)
        self.assertEqual(filenames, ["one.txt", "two.txt"])
        self.assertEqual(extraction_reports[0]["file_name"], "one.txt")

    def test_clear_state_preserves_unrelated_values(self) -> None:
        state = {
            "manual_company": "Example",
            "manual_parser_suggestions": {"title": "Analyst"},
            "manual_generate_7": True,
            "manual_upload_key_suffix": 2,
            "unrelated": "keep",
        }
        with patch.object(dashboard_manual.st, "session_state", state):
            dashboard_manual.clear_manual_job_session_state()
        self.assertEqual(state["unrelated"], "keep")
        self.assertEqual(state["manual_upload_key_suffix"], 3)
        self.assertEqual(state["manual_status"], "Saved")
        self.assertNotIn("manual_company", state)
        self.assertNotIn("manual_generate_7", state)

    def test_submission_validation_and_successful_save(self) -> None:
        payload = {
            "submitted": True,
            "company": "Example",
            "title": "Data Analyst",
            "location": "Remote",
            "source": "Company careers page",
            "url": "https://jobs.example.test/7",
            "salary_range": "",
            "visa_note": "",
            "status": "Saved",
            "notes": "",
            "job_description": "A complete job description",
            "suggestions": {"title": "Data Analyst"},
        }
        ui = MagicMock()
        ui.session_state = {}
        with patch.object(dashboard_manual, "st", ui):
            missing_title = {**payload, "title": ""}
            dashboard_manual.save_manual_form_submission(missing_title, [])
            ui.error.assert_called_with("Job title is required.")

            ui.reset_mock()
            missing_description = {**payload, "job_description": ""}
            dashboard_manual.save_manual_form_submission(missing_description, [])
            ui.error.assert_called_with(
                "Job description is required. Paste text manually or extract it from an upload."
            )

            ui.reset_mock()
            with patch.object(dashboard_manual, "is_valid_url", return_value=False):
                dashboard_manual.save_manual_form_submission(payload, [])
            ui.error.assert_called_with("Enter a valid http(s) Job URL, or leave it blank.")

            ui.reset_mock()
            saved = {**payload, "created_at": "2026-07-22", "id": "manual-7"}
            upload = UploadedFile("job.txt", b"Job description")
            with patch.object(dashboard_manual, "is_valid_url", return_value=True), patch.object(
                dashboard_manual, "duplicate_manual_job_exists", return_value=False
            ), patch.object(dashboard_manual, "save_manual_job", return_value=saved) as save:
                dashboard_manual.save_manual_form_submission(payload, [upload])
            save.assert_called_once()
            self.assertEqual(
                ui.session_state["manual_generate_selected"],
                dashboard_manual.manual_record_label(saved),
            )
            ui.success.assert_called_once()

    def test_package_generation_guards_and_success(self) -> None:
        ui = MagicMock()
        ui.session_state = {}
        render_confirmation = Mock(return_value={"company": "Example"})
        generation_allowed = Mock(return_value=True)
        summary = {
            "match_score": 78,
            "recommendation": "Review",
            "analysis_path": Path("analysis.md"),
            "cover_letter_path": Path("cover-letter.md"),
            "cover_letter_docx_path": Path("cover-letter.docx"),
            "tracker_id": 9,
        }
        services = dashboard_manual.ManualPageServices(
            company_generation_allowed=generation_allowed,
            current_workspace=Mock(return_value=object()),
            demo_mode_enabled=Mock(return_value=False),
            relative_path=lambda path: str(path),
            render_manual_company_confirmation=render_confirmation,
            render_page_header=Mock(),
            run_with_captured_output=Mock(return_value=(summary, "backend output")),
        )
        record = {
            "id": "manual-7",
            "company": "Example",
            "title": "Data Analyst",
            "location": "Remote",
            "url": "https://jobs.example.test/7",
            "notes": "",
            "markdown_path": "job.md",
        }

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "job.md").write_text("full description", encoding="utf-8")
            with patch.object(dashboard_manual, "PROJECT_ROOT", root), patch.object(
                dashboard_manual, "st", ui
            ), patch.object(
                dashboard_manual,
                "classify_jd_quality",
                return_value={
                    "reliable_scoring_ready": False,
                    "display_label": "Incomplete",
                    "next_action": "Paste the full JD.",
                },
            ), patch.object(dashboard_manual, "jsearch_configured", return_value=False):
                dashboard_manual.generate_package_for_manual_record(record, "generate", services)
            ui.warning.assert_called_once()
            services.run_with_captured_output.assert_not_called()

            ui.reset_mock()
            ui.session_state = {}
            ui.button.return_value = True
            with patch.object(dashboard_manual, "PROJECT_ROOT", root), patch.object(
                dashboard_manual, "st", ui
            ), patch.object(
                dashboard_manual,
                "classify_jd_quality",
                return_value={"reliable_scoring_ready": True},
            ), patch.object(dashboard_manual, "sync_manual_job_from_markdown"), patch.object(
                dashboard_manual, "update_manual_job"
            ):
                dashboard_manual.generate_package_for_manual_record(record, "generate", services)
            services.run_with_captured_output.assert_called_once()
            self.assertEqual(ui.session_state["manual_generated_summary"]["tracker_id"], 9)
            ui.success.assert_called_once()

    def test_prepare_state_and_demo_page_short_circuit(self) -> None:
        state = {
            "manual_pending_suggestions": {
                "title": "Analyst",
                "job_title_confidence": "high",
            },
            "manual_pending_clean_text": "Clean JD",
        }
        with patch.object(dashboard_manual.st, "session_state", state):
            dashboard_manual.prepare_manual_job_session_state()
        self.assertEqual(state["manual_title"], "Analyst")
        self.assertEqual(state["manual_job_description"], "Clean JD")

        ui = MagicMock()
        services = dashboard_manual.ManualPageServices(
            company_generation_allowed=Mock(),
            current_workspace=Mock(),
            demo_mode_enabled=Mock(return_value=True),
            relative_path=Mock(),
            render_manual_company_confirmation=Mock(),
            render_page_header=Mock(),
            run_with_captured_output=Mock(),
        )
        with patch.object(dashboard_manual, "st", ui):
            dashboard_manual.manual_job_target_tab(services)
        services.render_page_header.assert_called_once()
        ui.info.assert_called_once()

    def test_summary_and_saved_record_render_paths(self) -> None:
        ui = MagicMock()
        ui.session_state = {
            "manual_company": "Example",
            "manual_title": "Data Analyst",
            "manual_location": "Remote",
            "manual_url": "https://jobs.example.test/7",
            "manual_visa_note": "Sponsorship available",
        }
        ui.columns.side_effect = lambda value, **kwargs: [MagicMock() for _ in range(
            value if isinstance(value, int) else len(value)
        )]
        ui.expander.return_value = MagicMock()
        ui.button.return_value = False
        suggestions = {
            "company": "Parser Company",
            "company_confidence": "medium",
            "company_evidence": "Parser Company careers",
            "title": "Data Analyst",
            "job_title_confidence": "high",
            "job_title_evidence": "Data Analyst",
            "location": "Remote",
            "location_confidence": "high",
            "location_evidence": "Remote",
            "visa_note": "Sponsorship available",
            "visa_confidence": "medium",
            "visa_evidence": "Sponsorship may be available",
            "responsibilities": ["Build reporting pipelines"],
            "requirements": ["Python", "SQL"],
            "keywords": ["Python", "SQL"],
        }
        full_jd = " ".join(["Python SQL analytics requirement"] * 25)
        report = {
            "file_name": "job.pdf",
            "pages_processed": 2,
            "characters_extracted": 1500,
            "sections_detected": 3,
            "method": "pdf-text",
        }
        record = {
            "company": "Example",
            "title": "Data Analyst",
            "created_at": "2026-07-22",
            "id": "7",
            "url": "https://jobs.example.test/7",
            "job_description": full_jd,
        }
        with patch.object(dashboard_manual, "st", ui):
            dashboard_manual.render_compact_at_a_glance(
                suggestions,
                job_description=full_jd,
                reports=[],
            )
            dashboard_manual.render_extraction_reports([report])
            ui.selectbox.return_value = dashboard_manual.manual_record_label(record)
            self.assertEqual(dashboard_manual.select_manual_record([record], "selected"), record)
            dashboard_manual.render_manual_record_long_details(record)
        ui.success.assert_called_once()
        ui.link_button.assert_called_once()


if __name__ == "__main__":
    unittest.main()
