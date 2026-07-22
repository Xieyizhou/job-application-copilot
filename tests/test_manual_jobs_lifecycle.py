"""Persistence, extraction, and parsing tests for manually added jobs."""

from __future__ import annotations

import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import manual_jobs


FULL_JD = """Data Analyst
Example Analytics
Remote · Full-time

## Responsibilities
- Build recurring dashboards and automated data pipelines.
- Partner with stakeholders to define useful business metrics.
- Validate datasets and document analytical decisions.

## Requirements
- Two years of professional experience using Python and SQL.
- Experience with Tableau, Excel, statistics, and data visualization.
- Strong written communication and cross-functional collaboration.

## Logistics
This role is remote in the United States. No visa sponsorship is available.
The salary range is $70,000 to $90,000.

Example Analytics builds reporting products for operations teams. The successful
candidate will investigate data quality, explain findings, maintain documentation,
and improve repeatable workflows across several business functions.
"""


class ManualJobLifecycleTests(unittest.TestCase):
    def patched_storage(self, root: Path) -> ExitStack:
        stack = ExitStack()
        jobs_dir = root / "manual_jobs"
        stack.enter_context(patch.object(manual_jobs, "PROJECT_ROOT", root))
        stack.enter_context(patch.object(manual_jobs, "MANUAL_JOBS_DIR", jobs_dir))
        stack.enter_context(patch.object(manual_jobs, "MANUAL_JOBS_JSONL", jobs_dir / "manual_jobs.jsonl"))
        stack.enter_context(patch.object(manual_jobs, "MANUAL_SAVED_JOBS_DIR", jobs_dir / "saved_jobs"))
        stack.enter_context(patch.object(manual_jobs, "MANUAL_UPLOADS_DIR", jobs_dir / "uploads"))
        return stack

    def test_save_update_confirm_and_reload_manual_job(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.patched_storage(Path(directory)):
            record = manual_jobs.save_manual_job(
                company="Example Analytics",
                title="Data Analyst",
                location="Remote, United States",
                source="Company website",
                url="https://jobs.example.test/data-analyst",
                salary_range="$70,000-$90,000",
                visa_note="No visa sponsorship",
                status="Saved",
                notes="Initial review",
                job_description=FULL_JD,
                upload_files=[("job.md", FULL_JD.encode())],
            )
            records = manual_jobs.load_manual_jobs()
            self.assertEqual(len(records), 1)
            self.assertTrue(manual_jobs.duplicate_manual_job_exists(
                "Example Analytics", "Data Analyst", "https://jobs.example.test/data-analyst"
            ))
            self.assertFalse(manual_jobs.duplicate_manual_job_exists(
                "Example Analytics",
                "Data Analyst",
                "https://jobs.example.test/data-analyst",
                exclude_id=record["id"],
            ))
            markdown_path = Path(directory) / record["markdown_path"]
            self.assertTrue(markdown_path.is_file())
            self.assertIn("## Job Description", markdown_path.read_text())
            self.assertTrue(record["source_upload_filenames"])

            updated = manual_jobs.update_manual_job(record["id"], status="Analyzed", notes="Checked")
            assert updated is not None
            self.assertEqual(updated["status"], "Analyzed")
            confirmed = manual_jobs.confirm_manual_job_company(record["id"], "Example Analytics Inc.")
            assert confirmed is not None
            self.assertTrue(confirmed["company_confirmed_by_user"])

    def test_jsonl_loader_ignores_malformed_lines(self) -> None:
        with tempfile.TemporaryDirectory() as directory, self.patched_storage(Path(directory)):
            manual_jobs.ensure_manual_job_dirs()
            manual_jobs.MANUAL_JOBS_JSONL.write_text(
                '{"id":"valid","company":"Example"}\nnot-json\n[]\n', encoding="utf-8"
            )
            self.assertEqual(manual_jobs.load_manual_jobs(), [{"id": "valid", "company": "Example"}])

    def test_text_upload_and_quality_helpers_fail_closed(self) -> None:
        text_result = manual_jobs.extract_text_from_upload("job.txt", b"Role: Analyst")
        self.assertEqual(text_result.text, "Role: Analyst")
        self.assertTrue(manual_jobs.extract_text_from_upload("job.exe", b"x").error)
        self.assertTrue(manual_jobs.pdf_extraction_is_low_quality(manual_jobs.ExtractionResult("short")))
        self.assertEqual(
            manual_jobs.join_unique_warnings(["First\nSecond", "First", ""]),
            "First\nSecond",
        )

    def test_structured_parser_extracts_reviewable_fields(self) -> None:
        suggestions = manual_jobs.parse_job_description_suggestions(FULL_JD)
        self.assertEqual(suggestions["title"], "Data Analyst")
        self.assertIn("Example Analytics", suggestions["company"])
        self.assertIn("Remote", suggestions["location"])
        self.assertIn("Python", suggestions["keywords"])
        self.assertIn("No visa sponsorship", suggestions["visa_note"])
        self.assertTrue(suggestions["requirements"])
        self.assertTrue(suggestions["responsibilities"])

    def test_cleanup_sections_and_quality_warnings(self) -> None:
        noisy = """LinkedIn\nEasy Apply\nData Analyst\nExample Analytics\nRemote\n## Requirements\nPython and SQL required.\n"""
        cleaned = manual_jobs.clean_extracted_job_text(noisy)
        self.assertNotIn("Easy Apply", cleaned)
        self.assertEqual(manual_jobs.canonical_section_key("## Requirements"), "requirements")
        sections = manual_jobs.parse_structured_sections(FULL_JD)
        self.assertTrue(sections["requirements"])
        warnings = manual_jobs.job_description_quality_warnings(
            company="", title="", location="", url="", job_description="short"
        )
        self.assertGreaterEqual(len(warnings), 4)

    def test_linkedin_header_and_location_variants(self) -> None:
        header = manual_jobs.parse_linkedin_pdf_header(
            [
                "Machine Learning Engineer",
                "Example AI · Singapore · Full-time",
                "Remote matches your job preferences",
            ]
        )
        self.assertEqual(header["company"], "Example AI")
        self.assertEqual(header["job_title"], "Machine Learning Engineer")
        self.assertEqual(header["job_type"], "Full-time")
        self.assertEqual(manual_jobs.parse_linkedin_workplace_type("Workplace type is Hybrid"), "Hybrid")
        self.assertEqual(manual_jobs.parse_linkedin_job_type("Job type is Part time"), "Part-time")
        self.assertEqual(
            manual_jobs.infer_title_company_location_from_header(
                ["Data Analyst", "Example Analytics · London, United Kingdom"]
            ),
            ("Data Analyst", "Example Analytics", "London, United Kingdom"),
        )
        self.assertEqual(
            manual_jobs.split_location_options("Remote, Singapore, or London, United Kingdom"),
            ["Singapore", "London", "United Kingdom"],
        )
        self.assertEqual(manual_jobs.detected_known_locations("Singapore or London"), ["Singapore", "London"])

    def test_authorization_employment_and_company_evidence(self) -> None:
        authorization = (
            "Candidates must already have the right to work. "
            "No visa sponsorship is available."
        )
        note, confidence, evidence = manual_jobs.visa_evidence_for(authorization)
        self.assertEqual(note, "Existing work authorization required; no visa sponsorship.")
        self.assertEqual(confidence, "high")
        self.assertIn("visa sponsorship", evidence.lower())
        self.assertTrue(manual_jobs.has_strict_work_authorization_phrase(authorization))
        self.assertIn("right to work", manual_jobs.find_authorization_lines(authorization).lower())
        self.assertEqual(
            manual_jobs.infer_employment_type("This is a full-time internship position."),
            ("Internship", "Internship language detected."),
        )
        company, company_confidence, company_evidence = manual_jobs.infer_company_from_body(
            "Example Analytics Labs builds reporting software. "
            "Example Analytics Labs provides data tools."
        )
        self.assertEqual(company, "Example Analytics Labs")
        self.assertEqual(company_confidence, "high")
        self.assertIn("repeated", company_evidence)

    def test_section_heading_is_never_inferred_as_company(self) -> None:
        job_text = """About the role
Responsibilities
You will build production data pipelines and maintain dashboards.
Requirements
Experience with Python and SQL is required.
"""
        suggestions = manual_jobs.parse_job_description_suggestions(job_text)
        self.assertEqual(suggestions["company"], "")
        self.assertEqual(suggestions["company_confidence"], "low")
        self.assertFalse(manual_jobs.looks_like_company_candidate("About the role"))

    def test_pdf_formatting_and_fallback_selection(self) -> None:
        formatted = manual_jobs.format_pdf_pages(
            ["1 of 2\n## Responsibilities\nBuild dash-\nboards", "## Requirements\nPython"]
        )
        self.assertIn("--- Page 1 ---", formatted)
        self.assertIn("Build dashboards", formatted)
        self.assertEqual(
            manual_jobs.detect_section_headings(formatted),
            ["## Responsibilities", "## Requirements"],
        )
        report = manual_jobs.build_pdf_extraction_report(
            "test", [formatted], formatted, {"title": " Example Role "}
        )
        self.assertEqual(report["metadata_title"], "Example Role")
        self.assertEqual(report["sections_detected"], 2)

        weak = manual_jobs.ExtractionResult("short", warning="weak", report={"warnings": ["weak"]})
        strong = manual_jobs.ExtractionResult("x" * 600, report={"warnings": []})
        with patch.object(manual_jobs, "extract_pdf_with_pdfplumber", return_value=weak), patch.object(
            manual_jobs, "extract_pdf_with_pymupdf", return_value=strong
        ):
            selected = manual_jobs.extract_text_from_pdf(b"pdf")
        self.assertEqual(selected.text, strong.text)
        self.assertIn("weak", selected.warning)

        empty = manual_jobs.ExtractionResult("", warning="unavailable", report={"method": "none"})
        with patch.object(manual_jobs, "extract_pdf_with_pdfplumber", return_value=empty), patch.object(
            manual_jobs, "extract_pdf_with_pymupdf", return_value=empty
        ), patch.object(manual_jobs, "extract_pdf_with_pymupdf_ocr", return_value=empty):
            failed = manual_jobs.extract_text_from_pdf(b"pdf")
        self.assertFalse(failed.text)
        self.assertIn("unavailable", failed.warning)

    def test_role_title_and_summary_helpers(self) -> None:
        self.assertEqual(
            manual_jobs.extract_role_title_phrases("Example_AI_Machine_Learning_Engineer.pdf"),
            ["AI Machine Learning Engineer"],
        )
        self.assertTrue(manual_jobs.is_plausible_job_title_line("Data Analyst"))
        self.assertFalse(manual_jobs.is_plausible_job_title_line("This sentence builds products for customers."))
        self.assertEqual(manual_jobs.title_confidence_for("Data Analyst", "Found in header"), "medium")
        self.assertEqual(manual_jobs.deduplicate_preserving_order(["a", "b", "a"]), ["a", "b"])
        self.assertEqual(
            manual_jobs.extract_section(
                "Responsibilities\n- Build models\nRequirements\n- Python",
                ["responsibil"],
            ),
            "Build models",
        )
        summary = manual_jobs.build_role_summary(
            title="Data Analyst",
            company="Example",
            employment_type="Full-time",
            keywords=["Python", "SQL"],
            sections={"responsibilities": ["Build dashboards"]},
        )
        self.assertIn("Data Analyst at Example", summary)


if __name__ == "__main__":
    unittest.main()
