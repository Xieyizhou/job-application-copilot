"""Regression tests for the simplified Review Jobs information hierarchy."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import dashboard_review_page
from dashboard_fit_sections import accepted_semantic_matches
from dashboard_review import REVIEW_INBOX_OPTIONS, review_inbox_view_matches
from dashboard_review_components import (
    hard_constraint,
    jd_quality_label,
    main_gap,
    strongest_evidence,
    visible_role_fit,
)
from dashboard_review_styles import badge_html, decision_field_html
from dashboard_review_selector import job_picker_label, picked_review_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def source_line_count(relative_path: str) -> int:
    return len((PROJECT_ROOT / relative_path).read_text(encoding="utf-8").splitlines())


def largest_top_level_function(relative_path: str) -> int:
    tree = ast.parse((PROJECT_ROOT / relative_path).read_text(encoding="utf-8"))
    sizes = [
        int(node.end_lineno or node.lineno) - node.lineno + 1
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    ]
    return max(sizes, default=0)


def sample_job(*, confidence: str = "high") -> dict[str, object]:
    return {
        "analysis_available": True,
        "score": 82,
        "recommendation": "Apply",
        "confidence": {"level": confidence, "active_requirement_count": 5},
        "eligibility": {"status": "passed", "reasons": []},
        "jd_quality": {"display_label": "Complete", "reliable_scoring_ready": True},
        "analysis_result": {
            "analysis_available": True,
            "coverage_score": 80,
            "score_breakdown": [],
            "matched_strengths": ["Built production data pipelines"],
            "weak_areas": ["Limited cloud deployment evidence"],
            "main_risk": "Cloud deployment evidence is limited.",
            "semantic_evidence": {
                "matches": [
                    {
                        "accepted": True,
                        "requirement": f"Requirement {index}",
                        "evidence": f"Evidence {index}",
                    }
                    for index in range(5)
                ]
            },
        },
    }


class ReviewHierarchyTests(unittest.TestCase):
    def test_default_inbox_has_four_decision_views(self) -> None:
        self.assertEqual(REVIEW_INBOX_OPTIONS, ["Recommended", "Needs attention", "Ready", "All"])
        job = sample_job()
        self.assertTrue(review_inbox_view_matches(job, "Recommended", "Not tracked", "No cover letter"))
        self.assertTrue(review_inbox_view_matches(job, "Needs attention", "Not tracked", "No cover letter"))
        self.assertTrue(review_inbox_view_matches(job, "Ready", "ready", "Cover letter ready"))
        self.assertTrue(review_inbox_view_matches(job, "All", "Not tracked", "No cover letter"))

    def test_show_all_reset_clears_hidden_operational_filters(self) -> None:
        state: dict[str, object] = {"review_tracker_filter": "Ignored"}
        with patch.object(dashboard_review_page.st, "session_state", state):
            dashboard_review_page.reset_review_filters("Recommended", False, show_all=True)
        self.assertEqual(state["review_inbox_view"], "All")
        self.assertEqual(state["review_tracker_filter"], "all")
        self.assertEqual(state["review_minimum_score"], 0)
        self.assertFalse(state["review_hide_hard_red_flags"])

    def test_low_confidence_hides_numeric_role_fit(self) -> None:
        self.assertEqual(visible_role_fit(sample_job(confidence="low")), "Not reliable")
        self.assertEqual(visible_role_fit(sample_job(confidence="high")), "82/100")

    def test_default_decision_content_is_concise_and_grounded(self) -> None:
        job = sample_job()
        self.assertEqual(strongest_evidence(job), "Built production data pipelines")
        self.assertEqual(main_gap(job), "Cloud deployment evidence is limited.")
        self.assertEqual(hard_constraint(job), "No hard constraint detected.")
        self.assertEqual(jd_quality_label(job), "Complete")
        self.assertEqual(len(accepted_semantic_matches(job["analysis_result"])), 3)

    def test_badges_escape_untrusted_values(self) -> None:
        rendered = badge_html("JD", "<script>alert(1)</script>", "warning")
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        decision = decision_field_html("Recommendation", "Manual Review <unsafe>")
        self.assertIn("Manual Review &lt;unsafe&gt;", decision)

    def test_job_picker_uses_paths_and_compact_labels(self) -> None:
        first = {"path": "/jobs/one.md", "company": "Example", "role": "Data Analyst", "recommendation": "Apply"}
        second = {"path": "/jobs/two.md", "company": "Example", "role": "ML Engineer", "recommendation": "Manual Review"}
        self.assertEqual(picked_review_job([first, second], "/jobs/two.md"), second)
        self.assertEqual(job_picker_label(second), "Example · ML Engineer — Manual Review")

    def test_review_modules_stay_bounded(self) -> None:
        self.assertLessEqual(source_line_count("src/dashboard_review_page.py"), 700)
        self.assertLessEqual(source_line_count("src/dashboard_review_components.py"), 250)
        self.assertLessEqual(source_line_count("src/dashboard_fit_sections.py"), 180)
        self.assertLessEqual(source_line_count("src/dashboard_review_styles.py"), 80)
        self.assertLessEqual(largest_top_level_function("src/dashboard_review_components.py"), 60)
        self.assertLessEqual(largest_top_level_function("src/dashboard_fit_sections.py"), 60)


if __name__ == "__main__":
    unittest.main()
