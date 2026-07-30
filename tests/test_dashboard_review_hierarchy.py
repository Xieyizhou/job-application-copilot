"""Regression tests for the simplified Review Jobs information hierarchy."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import dashboard_review_page
from dashboard_review_filters import reset_review_filters
from dashboard_fit_sections import accepted_semantic_matches
from dashboard_review import REVIEW_INBOX_OPTIONS, review_inbox_view_matches
from dashboard_review import primary_review_action
from dashboard_review_components import (
    hard_constraint,
    jd_quality_label,
    main_gap,
    strongest_evidence,
    visible_role_fit,
)
from dashboard_review_styles import action_note_html, decision_field_html
from dashboard_review_selector import (
    compact_review_table_row,
    review_table_row,
    selected_review_table_job,
)


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
            reset_review_filters("Recommended", False, show_all=True)
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

    def test_decision_fields_escape_untrusted_values(self) -> None:
        decision = decision_field_html("Recommendation", "Manual Review <unsafe>")
        self.assertIn("Manual Review &lt;unsafe&gt;", decision)
        action = action_note_html("Review <unsafe>", caution=True)
        self.assertIn("Review &lt;unsafe&gt;", action)

    def test_default_header_separates_four_product_signals(self) -> None:
        source = (
            PROJECT_ROOT / "src" / "dashboard_review_components.py"
        ).read_text(encoding="utf-8")
        fields = source[source.index("decision_fields = [") : source.index(
            "st.markdown(",
            source.index("decision_fields = ["),
        )]
        self.assertIn('"Role Fit"', fields)
        self.assertIn('"Eligibility"', fields)
        self.assertIn('"Confidence"', fields)
        self.assertIn('"JD Quality"', fields)
        self.assertNotIn('"Recommendation"', fields)

    def test_review_table_combines_selection_and_four_signals(self) -> None:
        first = {
            **sample_job(),
            "path": "/jobs/one.md",
            "company": "Example",
            "role": "Data Analyst",
        }
        second = {
            **sample_job(confidence="low"),
            "path": "/jobs/two.md",
            "company": "Example",
            "role": "ML Engineer",
        }
        self.assertIs(selected_review_table_job([first, second], [1], "/jobs/one.md"), second)
        self.assertIs(selected_review_table_job([first, second], [], "/jobs/two.md"), second)
        self.assertEqual(
            review_table_row(second),
            {
                "Job": "Example · ML Engineer",
                "Role Fit": "Not reliable",
                "Eligibility": "Passed",
                "Confidence": "Low",
                "JD Quality": "Complete",
            },
        )
        self.assertEqual(
            compact_review_table_row(second),
            {
                "Job": "Example · ML Engineer",
                "Signals": "Not reliable · Pass · Low · Complete",
            },
        )

    def test_incomplete_jd_controls_the_single_action(self) -> None:
        job = sample_job(confidence="medium")
        job["jd_quality"] = {
            "display_label": "Partial JD",
            "reliable_scoring_ready": False,
        }
        action = primary_review_action(
            job,
            "ready",
            "Cover letter ready",
        )
        self.assertEqual(action.label, "Get Full JD")
        self.assertEqual(action.target_section, "JD")
        self.assertIn("full job description", action.message)
        self.assertTrue(action.caution)

        demo_action = primary_review_action(
            job,
            "Demo only",
            "Demo cover letter",
            demo=True,
        )
        self.assertEqual(demo_action.target_section, "JD")
        self.assertNotIn("tracker", demo_action.message.lower())

    def test_default_review_ui_has_one_master_list_without_duplicate_selectors(self) -> None:
        source = (
            PROJECT_ROOT / "src" / "dashboard_review_page.py"
        ).read_text(encoding="utf-8")
        selector_source = (
            PROJECT_ROOT / "src" / "dashboard_review_selector.py"
        ).read_text(encoding="utf-8")
        style_source = (
            PROJECT_ROOT / "src" / "dashboard_desktop_styles.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("Number of recommendations", source)
        self.assertNotIn("Choose a job", source)
        self.assertNotIn("Compare {len(jobs)} jobs", source)
        self.assertIn("render_review_job_table(", source)
        self.assertIn('st.container(height=420, border=False, key="review_detail_panel")', source)
        self.assertIn("vertical_alignment=\"top\"", source)
        self.assertNotIn("st.dataframe(", selector_source)
        self.assertIn('key="review_job_list"', selector_source)
        self.assertNotIn("● ", selector_source)
        self.assertNotIn("review-job-selected-marker", selector_source)
        self.assertNotIn("review-job-selected-marker", style_source)
        self.assertIn('type="secondary" if selected else "tertiary"', selector_source)
        self.assertIn("review-job-row-copy", selector_source)
        self.assertNotIn("st.caption(row[\"Signals\"])", selector_source)
        self.assertIn("position:absolute;top:0;right:-6px;bottom:-1px;left:-3px", style_source)
        self.assertIn("currentColor", style_source)
        self.assertIn("--primary-color", style_source)
        self.assertIn("stBaseButton-secondary", style_source)
        left_panel_start = source.index(
            'with left_panel, st.container(key="review_job_list_panel")'
        )
        detail_panel_start = source.index(
            'with detail_panel, st.container(key="review_detail_shell")'
        )
        header_start = source.index('services.render_page_header(', left_panel_start)
        self.assertLess(left_panel_start, header_start)
        self.assertLess(header_start, detail_panel_start)

    def test_review_modules_stay_bounded(self) -> None:
        self.assertLessEqual(source_line_count("src/dashboard_review_page.py"), 520)
        self.assertLessEqual(source_line_count("src/dashboard_review_filters.py"), 350)
        self.assertLessEqual(source_line_count("src/dashboard_review_components.py"), 250)
        self.assertLessEqual(source_line_count("src/dashboard_fit_sections.py"), 180)
        self.assertLessEqual(source_line_count("src/dashboard_review_styles.py"), 80)
        self.assertLessEqual(largest_top_level_function("src/dashboard_review_components.py"), 60)
        self.assertLessEqual(largest_top_level_function("src/dashboard_fit_sections.py"), 60)
        self.assertLessEqual(largest_top_level_function("src/dashboard_review_filters.py"), 65)


if __name__ == "__main__":
    unittest.main()
