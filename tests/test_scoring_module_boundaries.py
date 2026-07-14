"""Compatibility checks for the split scoring modules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import analyze_job  # noqa: E402
import scoring_eligibility  # noqa: E402
import scoring_engine  # noqa: E402
import scoring_extraction  # noqa: E402
import scoring_matching  # noqa: E402
import scoring_report  # noqa: E402


class ScoringModuleBoundaryTests(unittest.TestCase):
    def test_public_facade_reexports_production_functions(self) -> None:
        self.assertIs(analyze_job.parse_job_description, scoring_extraction.parse_job_description)
        self.assertIs(analyze_job.match_strength_for_keyword, scoring_matching.match_strength_for_keyword)
        self.assertIs(analyze_job.evaluate_eligibility, scoring_eligibility.evaluate_eligibility)
        self.assertIs(analyze_job.score_job_texts, scoring_engine.score_job_texts)
        self.assertIs(analyze_job.build_markdown_report, scoring_report.build_markdown_report)

    def test_legacy_constant_imports_remain_available(self) -> None:
        self.assertEqual(analyze_job.DIRECT_MATCH_STRENGTH, 1.0)
        self.assertEqual(analyze_job.PARTIAL_MATCH_STRENGTH, 0.6)
        self.assertIn("Core technical skills", analyze_job.SCORE_CATEGORIES)


if __name__ == "__main__":
    unittest.main()
