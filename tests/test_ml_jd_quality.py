"""Tests for local job-description quality classification."""

from __future__ import annotations

import unittest

from ml.jd_quality import classify_jd_quality, extract_description_body


class JobDescriptionQualityTests(unittest.TestCase):
    def test_discovery_provider_short_text_is_a_likely_snippet(self) -> None:
        job = """# Data Analyst
Source: Jooble
Description Source: search_snippet
JD Fetch Status: snippet_only

## Job Description
The analyst will use Python and SQL to support reporting, work with stakeholders,
and build dashboards. Required experience with data analysis and communication...
"""
        result = classify_jd_quality(job)
        self.assertEqual(result["label"], "likely_snippet")
        self.assertFalse(result["reliable_scoring_ready"])
        self.assertTrue(result["appears_incomplete"])
        self.assertIn("original full posting", result["next_action"])

    def test_complete_source_with_substantive_sections_is_scoring_ready(self) -> None:
        job = """# Data Analyst
Source: JSearch
Description Source: full_jd_api
JD Fetch Status: complete

## Job Description
Responsibilities
You will analyze product and customer data to identify trends and explain business outcomes.
You will build and maintain SQL transformations, recurring reports, and quality checks.
You will collaborate with product, engineering, and operations partners on measurable questions.
You will document metric definitions and support reliable dashboard delivery for stakeholders.

Requirements
Required experience with Python and SQL for practical data analysis and reporting work.
Must have experience with data visualization, analytical communication, and stakeholder reviews.
Knowledge of statistics, experiment design, and data quality methods is required.
Ability to translate ambiguous business questions into documented analytical steps is required.
Two years of experience delivering reproducible analysis or equivalent project work is preferred.

About us
The team maintains shared data products used across several business functions. The role works
with established engineering standards, peer review, documented definitions, and privacy-aware
data handling. Team members plan work together, review results, and improve recurring workflows.

Benefits
The employer provides health coverage, paid leave, learning support, and flexible work options.
"""
        result = classify_jd_quality(job)
        self.assertEqual(result["label"], "scoring_ready")
        self.assertTrue(result["reliable_scoring_ready"])
        self.assertGreaterEqual(result["requirement_statement_count"], 3)
        self.assertGreaterEqual(result["responsibility_statement_count"], 2)
        self.assertGreaterEqual(result["quality_score"], 70)

    def test_markdown_subheadings_remain_part_of_saved_job_description(self) -> None:
        job = """# Data Engineer
Source: Manual
Description Source: full_jd_manual
JD Fetch Status: complete

## Job Description
## Responsibilities
- You will build and maintain production data pipelines for analytics teams.
- You will monitor data quality and collaborate with engineering stakeholders.

## Requirements
- Required experience with Python and SQL in production environments.
- Must have experience with ETL orchestration and automated testing.
- Knowledge of cloud data platforms and observability is required.
- Ability to document systems and communicate tradeoffs is required.

## Benefits
The employer provides health coverage, paid leave, professional development,
and flexible work arrangements. The team uses documented reviews and shared
engineering standards to maintain reliable data products across the company.
"""
        body = extract_description_body(job)
        result = classify_jd_quality(job)

        self.assertIn("## Responsibilities", body)
        self.assertIn("## Requirements", body)
        self.assertIn("ETL orchestration", body)
        self.assertNotEqual(result["label"], "empty_or_unreadable")
        self.assertGreaterEqual(result["requirement_statement_count"], 3)
        self.assertGreaterEqual(result["responsibility_statement_count"], 2)

    def test_empty_manual_record_has_zero_document_quality(self) -> None:
        job = """# Data Analyst
Source: Manual
Description Source: full_jd_manual
JD Fetch Status: complete

## Job Description
"""
        result = classify_jd_quality(job)

        self.assertEqual(result["label"], "empty_or_unreadable")
        self.assertEqual(result["quality_score"], 0)

    def test_boilerplate_does_not_create_scoring_readiness(self) -> None:
        boilerplate = " ".join(
            [
                "Equal opportunity employer and affirmative action statement.",
                "Reasonable accommodation and disability status information.",
                "Privacy policy and employment eligibility verification details.",
            ]
            * 12
        )
        result = classify_jd_quality(boilerplate)
        self.assertEqual(result["label"], "boilerplate_heavy")
        self.assertFalse(result["provisional_scoring_ready"])
        self.assertGreaterEqual(result["boilerplate_share"], 0.45)

    def test_empty_text_fails_closed(self) -> None:
        result = classify_jd_quality("")
        self.assertEqual(result["label"], "empty_or_unreadable")
        self.assertEqual(result["quality_score"], 0)
        self.assertFalse(result["provisional_scoring_ready"])


if __name__ == "__main__":
    unittest.main()
