"""Unit tests for UI-independent dashboard analysis orchestration."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dashboard_analysis_service import (
    analyze_dashboard_job,
    dashboard_analysis_cache_key,
    unavailable_dashboard_analysis,
)


class DashboardAnalysisServiceTests(unittest.TestCase):
    def test_unavailable_result_keeps_decision_signals_independent(self) -> None:
        result = unavailable_dashboard_analysis("No reliable requirements.")

        self.assertIsNone(result["score"])
        self.assertEqual(result["recommendation"], "Manual Review")
        self.assertEqual(result["eligibility"]["status"], "manual_review")
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertFalse(result["analysis_available"])

    def test_content_cache_reuses_result_and_invalidates_changed_job(self) -> None:
        analyzer = Mock(
            return_value={
                "score": 75,
                "confidence": {"active_requirement_count": 2},
            }
        )
        cache: dict[str, dict[str, object]] = {}
        job = {"canonical_job_key": "sample-role"}

        first = analyze_dashboard_job(
            job,
            "Python and SQL",
            "Python evidence",
            analyzer=analyzer,
            scoring_version="test-v1",
            cache=cache,
            workspace_mode="demo",
            workspace_root="data/demo",
        )
        second = analyze_dashboard_job(
            job,
            "Python and SQL",
            "Python evidence",
            analyzer=analyzer,
            scoring_version="test-v1",
            cache=cache,
            workspace_mode="demo",
            workspace_root="data/demo",
        )
        changed = analyze_dashboard_job(
            job,
            "Python, SQL, and statistics",
            "Python evidence",
            analyzer=analyzer,
            scoring_version="test-v1",
            cache=cache,
            workspace_mode="demo",
            workspace_root="data/demo",
        )

        self.assertTrue(first["analysis_available"])
        self.assertEqual(second, first)
        self.assertTrue(changed["analysis_available"])
        self.assertEqual(analyzer.call_count, 2)

    def test_analyzer_failure_returns_manual_review_without_caching(self) -> None:
        def failing_analyzer(job_text: str, candidate_text: str) -> dict[str, object]:
            _ = job_text, candidate_text
            raise ValueError("invalid structured input")

        cache: dict[str, dict[str, object]] = {}
        result = analyze_dashboard_job(
            {"path": "sample.md"},
            "Python required",
            "Python evidence",
            analyzer=failing_analyzer,
            scoring_version="test-v1",
            cache=cache,
        )

        self.assertFalse(result["analysis_available"])
        self.assertEqual(result["recommendation"], "Manual Review")
        self.assertEqual(cache, {})

    def test_empty_inputs_short_circuit_before_analyzer(self) -> None:
        analyzer = Mock()
        cases = [
            ("", "Python evidence", "Job description is empty or unreadable."),
            ("Python required", "", "Candidate source is missing or empty."),
        ]

        for job_text, candidate_text, reason in cases:
            with self.subTest(reason=reason):
                result = analyze_dashboard_job(
                    {"path": "sample.md"},
                    job_text,
                    candidate_text,
                    analyzer=analyzer,
                    scoring_version="test-v1",
                )
                self.assertEqual(result["main_reason"], reason)

        analyzer.assert_not_called()

    def test_cache_key_is_scoped_to_workspace_and_scoring_version(self) -> None:
        job = {"canonical_job_key": "sample-role"}
        base = dashboard_analysis_cache_key(
            job,
            "Python",
            "Python evidence",
            scoring_version="v1",
            workspace_mode="demo",
            workspace_root="data/demo",
        )
        version_changed = dashboard_analysis_cache_key(
            job,
            "Python",
            "Python evidence",
            scoring_version="v2",
            workspace_mode="demo",
            workspace_root="data/demo",
        )
        workspace_changed = dashboard_analysis_cache_key(
            job,
            "Python",
            "Python evidence",
            scoring_version="v1",
            workspace_mode="personal",
            workspace_root="data/personal",
        )

        self.assertNotEqual(base, version_changed)
        self.assertNotEqual(base, workspace_changed)


if __name__ == "__main__":
    unittest.main()
