"""Cover-letter generation stays concise, role-specific, and resume-grounded."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from generate_cover_letter import (  # noqa: E402
    DEFAULT_THEME_KEYWORDS,
    TARGET_WORD_COUNT_MAX,
    build_cover_letter,
    build_internal_notes,
)


RESUME_TEXT = """# Ada Example

## Analytics Internship
- Built a Python and SQL reporting dashboard for operational datasets.
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

    def test_letter_uses_uploaded_resume_not_bank_only_claims(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        self.assertIn("Ada Example", letter)
        self.assertIn("Python and SQL reporting dashboard", letter)
        self.assertNotIn("Invented a production system", letter)

    def test_letter_has_direct_lead_and_hard_word_cap(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        self.assertIn("The Data Analyst role at Fictional Analytics Labs emphasizes", letter)
        self.assertNotIn("I am writing to apply", letter)
        self.assertNotIn("I am excited to apply", letter)
        self.assertLessEqual(len(letter.split()), TARGET_WORD_COUNT_MAX)

    def test_internal_notes_trace_exact_resume_evidence(self) -> None:
        letter = build_cover_letter(RESUME_TEXT, JOB_TEXT, self.bank)
        notes = build_internal_notes(RESUME_TEXT, JOB_TEXT, self.bank, [], letter)
        self.assertIn("Claim Trace — Exact Resume Evidence", notes)
        self.assertIn("Built a Python and SQL reporting dashboard", notes)
        self.assertIn("uploaded resume is not rewritten or regenerated", notes)


if __name__ == "__main__":
    unittest.main()
