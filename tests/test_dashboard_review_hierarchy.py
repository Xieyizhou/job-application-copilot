"""Regression tests for the simplified Review Jobs information hierarchy."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

import dashboard_review_page
from dashboard_jd_document import format_job_description_body
from dashboard_review_filters import reset_review_filters
from dashboard_fit_sections import accepted_semantic_matches
from dashboard_review import (
    REVIEW_INBOX_OPTIONS,
    review_fit_band,
    review_fit_band_counts,
    review_inbox_view_matches,
)
from dashboard_review import primary_review_action
from dashboard_review_components import (
    hard_constraint,
    jd_quality_label,
    main_gap,
    strongest_evidence,
    visible_role_fit,
)
from dashboard_review_chrome import saved_job_context
from dashboard_review_styles import action_note_html, decision_field_html
from dashboard_review_selector import (
    compact_review_table_row,
    mobile_job_picker_key,
    mobile_job_picker_options,
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
    def test_programmatic_review_navigation_syncs_the_visible_tab(self) -> None:
        state: dict[str, object] = {}
        job = {"path": Path("/jobs/example.md"), "label": "Example"}
        with patch.object(dashboard_review_page.st, "session_state", state):
            dashboard_review_page.set_review_job_selection(job, "JD")
        self.assertEqual(state["selected_review_tab"], "JD")
        self.assertEqual(state["review_detail_tabs"], "Job description")

    def test_job_description_body_omits_local_record_metadata(self) -> None:
        text = "# Role\n\nCompany: Example\nSource: Test\n\n## Job Description\n\nBuild systems."
        self.assertEqual(
            dashboard_review_page.extract_job_description_body(text),
            "Build systems.",
        )

    def test_plain_api_job_description_becomes_readable_markdown(self) -> None:
        body = (
            "Data Analyst We Are: A growing team. Job Description: "
            "• Build dashboards. • Explain results. Education and Experience Required: "
            "• Bachelor's degree."
        )
        formatted = format_job_description_body(body, "Data Analyst")
        self.assertNotIn("Data Analyst We Are", formatted)
        self.assertIn("### We Are", formatted)
        self.assertIn("### Job Description", formatted)
        self.assertIn("- Build dashboards.", formatted)
        self.assertIn("### Education and Experience Required", formatted)

    def test_repeated_provider_requirement_suffix_is_removed(self) -> None:
        repeated = (
            "Core description. This is a sufficiently long repeated provider passage that "
            "appears earlier in the complete job description and should only render once.\n\n"
            "## Requirements\n\n"
            "This is a sufficiently long repeated provider passage that appears earlier in "
            "the complete job description and should only render once."
        )
        formatted = format_job_description_body(repeated)
        self.assertNotIn("## Requirements", formatted)
        self.assertEqual(formatted.count("sufficiently long repeated provider passage"), 1)

    def test_default_inbox_has_four_decision_views(self) -> None:
        self.assertEqual(REVIEW_INBOX_OPTIONS, ["Recommended", "Needs attention", "Ready", "All"])
        job = sample_job()
        self.assertTrue(
            review_inbox_view_matches(job, "Recommended", "Not tracked", "No cover letter")
        )
        self.assertTrue(
            review_inbox_view_matches(job, "Needs attention", "Not tracked", "No cover letter")
        )
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
        source = (PROJECT_ROOT / "src" / "dashboard_review_components.py").read_text(
            encoding="utf-8"
        )
        fields = source[
            source.index("decision_fields = [") : source.index(
                "st.markdown(",
                source.index("decision_fields = ["),
            )
        ]
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

    def test_saved_job_fit_bands_follow_natural_model_outputs(self) -> None:
        strong = sample_job()
        review = {**sample_job(confidence="low"), "score": 77}
        weak = {**sample_job(), "score": 51}
        self.assertEqual(review_fit_band(strong), "Strong")
        self.assertEqual(review_fit_band(review), "Review")
        self.assertEqual(review_fit_band(weak), "Weak")
        self.assertEqual(
            review_fit_band_counts([strong, review, weak]),
            {"All": 3, "Strong": 1, "Review": 1, "Weak": 1},
        )

    def test_responsive_picker_state_is_isolated_by_selected_job(self) -> None:
        first = mobile_job_picker_key("/jobs/one.md")
        second = mobile_job_picker_key("/jobs/two.md")
        self.assertNotEqual(first, second)
        self.assertTrue(first.startswith("review_mobile_job_choice_"))

    def test_duplicate_labels_do_not_override_unique_selected_path(self) -> None:
        first = {
            **sample_job(),
            "label": "Bespoke Labs · Machine Learning Engineer",
            "path": "/jobs/a.md",
        }
        second = {
            **sample_job(),
            "label": "Bespoke Labs · Machine Learning Engineer",
            "path": "/jobs/b.md",
        }
        selected = dashboard_review_page.resolve_review_job_selection(
            [first, second], first["label"], second["path"]
        )
        self.assertIs(selected, second)

    def test_responsive_picker_uses_unique_values_for_duplicate_titles(self) -> None:
        first = {
            **sample_job(),
            "path": "/jobs/a.md",
            "normalized_location": "Austin, Texas",
        }
        second = {
            **sample_job(),
            "path": "/jobs/b.md",
            "normalized_location": "Boston, Massachusetts",
        }
        paths, labels = mobile_job_picker_options([first, second])
        self.assertEqual(paths, ["/jobs/a.md", "/jobs/b.md"])
        self.assertIn("Austin, Texas", labels[paths[0]])
        self.assertIn("Boston, Massachusetts", labels[paths[1]])

    def test_saved_job_context_distinguishes_same_role_in_different_locations(self) -> None:
        job = {
            "normalized_location": "Austin, Travis County",
            "last_seen_at": "2026-07-11T09:30:00Z",
        }
        self.assertEqual(saved_job_context(job), "Austin, Travis County · Jul 11, 2026")

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
        source = (PROJECT_ROOT / "src" / "dashboard_review_page.py").read_text(encoding="utf-8")
        selector_source = (PROJECT_ROOT / "src" / "dashboard_review_selector.py").read_text(
            encoding="utf-8"
        )
        style_source = (PROJECT_ROOT / "src" / "dashboard_desktop_styles.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("Number of recommendations", source)
        self.assertNotIn("Choose a job", source)
        self.assertNotIn("Compare {len(jobs)} jobs", source)
        self.assertIn("render_review_job_table(", source)
        self.assertIn("Cover letter needs a complete job description", source)
        self.assertIn('"Go to Job description"', source)
        self.assertNotIn("key_prefix=f\"cover_letter_", source)
        self.assertIn('st.container(border=False, key="review_detail_panel")', source)
        self.assertNotIn('st.container(height=420, border=False, key="review_detail_panel")', source)
        recovery_source = (PROJECT_ROOT / "src" / "dashboard_jd_recovery.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("Open job page", recovery_source)
        self.assertIn("Fetch page", recovery_source)
        self.assertIn("Paste full JD", recovery_source)
        self.assertIn("Search provider", recovery_source)
        self.assertNotIn('st.expander("Paste full JD"', recovery_source)
        self.assertNotIn("height:calc(100vh - 16.8rem)", style_source)
        self.assertIn('vertical_alignment="top"', source)
        self.assertNotIn("st.dataframe(", selector_source)
        self.assertIn('key="review_job_list"', selector_source)
        self.assertNotIn("height=285 if demo else 500", selector_source)
        self.assertNotIn("● ", selector_source)
        self.assertNotIn("review-job-selected-marker", selector_source)
        self.assertNotIn("review-job-selected-marker", style_source)
        self.assertIn('type="secondary" if selected else "tertiary"', selector_source)
        self.assertIn("review-job-row-copy", selector_source)
        self.assertNotIn('st.caption(row["Signals"])', selector_source)
        self.assertIn("position:absolute;top:0;right:.55rem;bottom:-1px;left:-3px", style_source)
        self.assertIn("overscroll-behavior:contain;scrollbar-gutter:stable", style_source)
        self.assertNotIn("filtered_jobs[: services.max_recommendation_limit]", source)
        self.assertIn("currentColor", style_source)
        self.assertIn("--primary-color", style_source)
        self.assertIn("stBaseButton-secondary", style_source)
        self.assertIn("@media (max-width:1299px)", style_source)
        self.assertIn('@media (min-width:1100px) and (max-width:1299px)', style_source)
        self.assertIn("overflow-y:auto !important;overflow-x:hidden !important", style_source)
        self.assertIn("-webkit-line-clamp:2", style_source)
        self.assertIn("[0.28, 0.72]", source)
        left_panel_start = source.index(
            'with left_panel, st.container(key="review_job_list_panel")'
        )
        detail_panel_start = source.index(
            'with detail_panel, st.container(key="review_detail_shell")'
        )
        header_start = source.index("render_saved_jobs_header(", left_panel_start)
        self.assertLess(left_panel_start, header_start)
        self.assertLess(header_start, detail_panel_start)

        evidence_source = (PROJECT_ROOT / "src" / "dashboard_fit_sections.py").read_text(
            encoding="utf-8"
        )
        evidence_styles = (PROJECT_ROOT / "src" / "dashboard_evidence_styles.py").read_text(
            encoding="utf-8"
        )
        self.assertIn('key="evidence_main_scroll"', evidence_source)
        self.assertIn(".st-key-evidence_main_scroll", evidence_styles)
        self.assertIn("overflow-y:clip !important", evidence_styles)
        self.assertIn("flex:0 0 max", evidence_styles)
        self.assertIn("calc(100vh - 19rem)", evidence_styles)

    def test_mobile_is_a_responsive_companion_not_a_remote_product(self) -> None:
        selector_source = (PROJECT_ROOT / "src" / "dashboard_review_selector.py").read_text(
            encoding="utf-8"
        )
        style_source = (PROJECT_ROOT / "src" / "dashboard_desktop_styles.py").read_text(
            encoding="utf-8"
        )
        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn('key="review_mobile_job_picker"', selector_source)
        self.assertIn(".st-key-review_mobile_job_picker {display:block", style_source)
        self.assertIn(".st-key-review_job_list {display:none", style_source)
        self.assertIn("same locally run Streamlit app", readme)
        self.assertIn("not a\n  hosted service", readme)

    def test_review_modules_stay_bounded(self) -> None:
        self.assertLessEqual(source_line_count("src/dashboard_review_page.py"), 525)
        self.assertLessEqual(source_line_count("src/dashboard_review_filters.py"), 350)
        self.assertLessEqual(source_line_count("src/dashboard_review_components.py"), 250)
        self.assertLessEqual(source_line_count("src/dashboard_fit_sections.py"), 180)
        self.assertLessEqual(source_line_count("src/dashboard_review_styles.py"), 105)
        self.assertLessEqual(largest_top_level_function("src/dashboard_review_components.py"), 60)
        self.assertLessEqual(largest_top_level_function("src/dashboard_fit_sections.py"), 60)
        self.assertLessEqual(largest_top_level_function("src/dashboard_review_filters.py"), 65)


if __name__ == "__main__":
    unittest.main()
