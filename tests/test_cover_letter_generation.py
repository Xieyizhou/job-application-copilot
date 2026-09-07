"""Cover Letter v2 stays concise, provenance-backed, and fail-closed."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_cover_letter import (  # noqa: E402
    DEFAULT_THEME_KEYWORDS,
    TARGET_WORD_COUNT_MAX,
    TARGET_WORD_COUNT_MIN,
    CoverLetterClaim,
    CoverLetterGenerationError,
    assert_cover_letter_claim_gate,
    build_cover_letter,
    build_cover_letter_plan,
    build_internal_notes,
    render_claim_paragraph,
    validate_manual_cover_letter_draft,
)


RESUME_TEXT = """# Ada Example

## Analytics Internship
- Built a Python and SQL reporting dashboard for operational datasets.
- Created recurring dashboards and communicated findings to product and operations partners.
- Reduced recurring data-quality review time by 30% through automated checks.

## Machine Learning Project
- Evaluated classification models with macro F1 and confusion matrices.
"""

JOB_TEXT = """# Data Analyst

Company: Fictional Analytics Labs
Role: Data Analyst
Location: Remote
Company Confirmed By User: yes
Company Confidence: High
Company Evidence: Confirmed in test input.

## Requirements
- Python and SQL for data analysis
- Build dashboards and communicate findings
- Must hold citizenship for an unsupported restricted assignment
"""


class CoverLetterGenerationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bank = {
            "theme_keywords": DEFAULT_THEME_KEYWORDS,
            "experiences": [
                {
                    "name": "Unsafe bank-only example",
                    "tags": ["python"],
                    "evidence": ["Invented a production system that is not in the resume."],
                    "safe_phrases": ["Invented a production system that is not in the resume."],
                }
            ],
        }

    def test_plan_and_letter_use_only_selected_resume_claims(self) -> None:
        plan = build_cover_letter_plan(RESUME_TEXT, JOB_TEXT, self.bank)
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)

        self.assertEqual(plan.schema_version, 1)
        self.assertEqual(plan.validation_status, "passed")
        self.assertEqual(len(plan.claims), 2)
        self.assertTrue(all(claim.evidence in RESUME_TEXT for claim in plan.claims))
        self.assertTrue(all(paragraph.claim_ids for paragraph in plan.paragraphs))
        self.assertIn("Ada Example", letter)
        self.assertIn("Python and SQL reporting dashboard", letter)
        self.assertIn("communicated findings", letter)
        self.assertNotIn("Invented a production system", letter)
        self.assertNotIn("citizenship", letter.casefold())

    def test_letter_has_hard_word_range_and_no_generic_lead(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        word_count = len(letter.split())

        self.assertIn("The Data Analyst position at Fictional Analytics Labs centers on", letter)
        self.assertNotIn("I am writing to apply", letter)
        self.assertNotIn("I am excited to apply", letter)
        self.assertGreaterEqual(word_count, TARGET_WORD_COUNT_MIN)
        self.assertLessEqual(word_count, TARGET_WORD_COUNT_MAX)

    def test_internal_notes_trace_exact_resume_evidence(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        notes = build_internal_notes(RESUME_TEXT, JOB_TEXT, self.bank, [], letter)
        self.assertIn("Claim Trace — Exact Resume Evidence", notes)
        self.assertIn("Built a Python and SQL reporting dashboard", notes)
        self.assertIn("Requirement-to-Resume Evidence Map", notes)
        self.assertIn("Direct support", notes)

    def test_one_requirement_or_low_similarity_blocks_generation(self) -> None:
        resume = """# Ada Example

## History Project
- Presented archival research findings to a student seminar.
"""
        job = """# Platform Engineer

Company: Fictional Platform Lab
Role: Platform Engineer
Company Confirmed By User: yes
Company Confidence: High
Company Evidence: Confirmed in test input.

## Requirements
- Manage Kubernetes infrastructure
"""
        with self.assertRaises(CoverLetterGenerationError) as raised:
            build_cover_letter(resume, job, self.bank)

        self.assertIn("two distinct", " ".join(raised.exception.reasons))

    def test_missing_confirmed_name_blocks_generation(self) -> None:
        with self.assertRaises(CoverLetterGenerationError) as raised:
            build_cover_letter_plan(RESUME_TEXT, JOB_TEXT, self.bank, candidate_name="")
        self.assertIn("Confirm your name", " ".join(raised.exception.reasons))

    def test_partial_only_claims_fail_direct_gate(self) -> None:
        claims = [
            CoverLetterClaim("claim_1", "Python", "required", "Partial", "Used Python.", "Skills", 0.6),
            CoverLetterClaim("claim_2", "SQL", "required", "Partial", "Used SQL.", "Project", 0.6),
        ]
        with self.assertRaises(CoverLetterGenerationError) as raised:
            assert_cover_letter_claim_gate(
                claims,
                {"unmatched_requirements": []},
                candidate_name="Ada Example",
            )
        self.assertIn("Direct", " ".join(raised.exception.reasons))

    def test_partial_claim_uses_transfer_language(self) -> None:
        paragraph = render_claim_paragraph(
            CoverLetterClaim(
                "claim_2",
                "production orchestration",
                "required",
                "Partial",
                "Built a local research workflow.",
                "Research Project",
                0.55,
            )
        )
        self.assertIn("not the same setting", paragraph)
        self.assertIn("could transfer", paragraph)
        self.assertNotIn("I have production orchestration", paragraph)

    def test_manual_draft_validation_requires_safe_format_and_review(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        errors, warnings = validate_manual_cover_letter_draft(letter)
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

        errors, warnings = validate_manual_cover_letter_draft(
            letter.replace("Thank you", "My work authorization is confirmed. Thank you")
        )
        self.assertEqual(errors, [])
        self.assertTrue(any("work authorization" in warning for warning in warnings))

        errors, _ = validate_manual_cover_letter_draft("A short claim plan draft.")
        self.assertTrue(any("internal workflow" in error for error in errors))
        self.assertTrue(any("150" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
