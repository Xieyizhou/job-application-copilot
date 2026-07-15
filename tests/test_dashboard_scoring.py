"""Focused integration tests for canonical scoring across dashboard surfaces."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard
from apply_package import parse_analysis_summary


MATCHING_CANDIDATE = (
    "Recent graduate with a bachelor's degree. Evidence includes Python, SQL, pandas, "
    "machine learning, model evaluation, data analysis, data visualization, communication, "
    "documentation, teamwork, classification, and PCA."
)


def analyzed_job(job_text: str, candidate_text: str = MATCHING_CANDIDATE) -> dict[str, object]:
    job = {
        "company": "Fictional Signal Works",
        "role": "Data Analyst",
        "location": "Remote",
        "normalized_location": "Remote",
        "job_url": "https://example.com/fictional-role",
        "path": Path("fictional-role.md"),
        "legacy_score": 57,
        "legacy_recommendation": "Maybe Apply",
    }
    analysis = dashboard.analyze_job_for_dashboard(
        job,
        job_text,
        candidate_text,
        use_cache=False,
    )
    return dashboard.apply_canonical_analysis(job, analysis)


class CanonicalDashboardAnalysisTests(unittest.TestCase):
    def test_dashboard_uses_full_analyzer_not_lightweight_score(self) -> None:
        job = analyzed_job("Machine learning")
        self.assertEqual(dashboard.score_job_for_dashboard("Machine learning"), 57)
        self.assertEqual(job["score"], 62)
        self.assertEqual(job["analysis_result"]["coverage_score"], 100)
        self.assertEqual(job["recommendation"], "Manual Review")
        self.assertEqual(job["confidence"]["level"], "low")

    def test_low_confidence_labels_numeric_fit_as_provisional(self) -> None:
        presentation = dashboard.build_fit_presentation(analyzed_job("Machine learning"))
        self.assertEqual(presentation["role_fit"], "Provisional 62/100")
        self.assertIn("Low confidence", presentation["card_status"])
        self.assertIn("Provisional 62/100", presentation["card_status"])
        self.assertEqual(presentation["recommendation"], "Manual Review")
        self.assertEqual(presentation["coverage_score"], 100)
        self.assertEqual(presentation["terms"]["active_requirement_count"], 1)
        self.assertEqual(presentation["terms"]["matched_requirement_count"], 1)

    def test_medium_and_high_confidence_show_numeric_role_fit(self) -> None:
        medium = analyzed_job(
            "This fictional team is hiring an analyst to prepare carefully reviewed local reports for stakeholders. "
            "Requirements include Python, SQL, data analysis, and communication across routine project work."
        )
        high = analyzed_job(
            "This fictional team is hiring an analyst to prepare carefully reviewed local reports for stakeholders. "
            "Requirements include Python, SQL, pandas, machine learning, model evaluation, data analysis, "
            "data visualization, communication, documentation, and teamwork across project work."
        )
        self.assertEqual(medium["confidence"]["level"], "medium")
        self.assertRegex(dashboard.build_fit_presentation(medium)["role_fit"], r"^\d+/100$")
        self.assertEqual(high["confidence"]["level"], "high")
        self.assertRegex(dashboard.build_fit_presentation(high)["role_fit"], r"^\d+/100$")

    def test_failed_and_manual_eligibility_override_apply(self) -> None:
        failed = analyzed_job(
            "Senior Machine Learning Engineer requires Python, SQL, pandas, machine learning, "
            "model evaluation, data analysis, communication, documentation, and 5+ years experience required."
        )
        manual = analyzed_job(
            "This role requires Python, SQL, pandas, machine learning, model evaluation, data analysis, "
            "communication, documentation, and work authorization review for visa sponsorship."
        )
        self.assertEqual(failed["eligibility"]["status"], "failed")
        self.assertEqual(failed["recommendation"], "Skip / Not Eligible")
        self.assertEqual(manual["eligibility"]["status"], "manual_review")
        self.assertEqual(manual["recommendation"], "Manual Review")

    def test_safe_failure_never_promotes_legacy_score(self) -> None:
        job = analyzed_job("", "")
        self.assertFalse(job["analysis_available"])
        self.assertIsNone(job["score"])
        self.assertEqual(job["recommendation"], "Manual Review")
        presentation = dashboard.build_fit_presentation(job)
        self.assertEqual(presentation["role_fit"], "Not available")
        self.assertIn("Stored legacy score: 57/100", presentation["card_status"])


class RequirementSummaryTests(unittest.TestCase):
    def test_required_and_preferred_terms_are_grouped(self) -> None:
        job = analyzed_job(
            "Fictional role supporting carefully reviewed analytical projects and reports for stakeholders.\n"
            "Required: Python, SQL, and model evaluation.\n"
            "Nice to have: PCA and data visualization for additional project work.",
            "Recent graduate with Python and data visualization evidence.",
        )
        terms = dashboard.summarize_analysis_requirements(job["analysis_result"])
        self.assertEqual(terms["matched_required"], ["Python"])
        self.assertEqual(terms["missing_required"], ["SQL", "model evaluation"])
        self.assertEqual(terms["matched_preferred"], ["data visualization"])
        self.assertEqual(terms["missing_preferred"], ["PCA"])

    def test_partial_required_and_preferred_remain_labeled(self) -> None:
        required = analyzed_job(
            "Python and robotics are required for this fictional junior role.",
            "Junior engineer with Python, UAV, and route planning evidence.",
        )
        preferred = analyzed_job(
            "Python is required. Robotics is preferred for this fictional junior role.",
            "Junior engineer with Python, UAV, and route planning evidence.",
        )
        required_terms = dashboard.summarize_analysis_requirements(required["analysis_result"])
        preferred_terms = dashboard.summarize_analysis_requirements(preferred["analysis_result"])
        self.assertTrue(required_terms["partial_required"])
        self.assertIn("Partial match", required_terms["partial_required"][0])
        self.assertTrue(preferred_terms["partial_preferred"])
        self.assertIn("Partial match", preferred_terms["partial_preferred"][0])

    def test_duplicate_terms_are_removed_and_order_is_stable(self) -> None:
        job = analyzed_job(
            "Fictional role supporting carefully reviewed analytical projects and reports for stakeholders.\n"
            "Python and SQL are required. Python is required again. SQL is repeated.\n"
            "PCA is preferred for additional project work.",
            "Recent graduate with Python evidence.",
        )
        terms = dashboard.summarize_analysis_requirements(job["analysis_result"])
        combined = terms["matched_required"] + terms["missing_required"] + terms["missing_preferred"]
        self.assertEqual(len(combined), len(set(combined)))
        self.assertEqual(terms["matched_required"], ["Python"])
        self.assertEqual(terms["missing_required"], ["SQL"])

    def test_no_recognized_requirements_is_unavailable(self) -> None:
        job = analyzed_job("Fictional employer offers a pleasant office and reviewed applications.")
        self.assertFalse(job["analysis_available"])
        self.assertEqual(
            dashboard.build_fit_presentation(job)["terms"]["active_requirement_count"],
            0,
        )


class DashboardFilteringAndPersistenceTests(unittest.TestCase):
    def test_sorting_uses_canonical_score(self) -> None:
        lower = analyzed_job("Python SQL pandas data analysis model evaluation communication documentation teamwork required.")
        higher = analyzed_job("Python SQL pandas machine learning model evaluation data analysis visualization communication documentation teamwork required.")
        lower["score"] = 40
        higher["score"] = 80
        lower["legacy_score"] = 99
        higher["legacy_score"] = 1
        sorted_jobs = dashboard.sorted_review_jobs([lower, higher], "Role Fit high to low")
        self.assertGreaterEqual(sorted_jobs[0]["score"], sorted_jobs[1]["score"])
        self.assertIs(sorted_jobs[0], higher)

    def test_recommended_filter_excludes_failed_and_low_confidence(self) -> None:
        failed = analyzed_job(
            "Senior Data Engineer requires Python SQL pandas data analysis communication documentation and 5+ years experience required."
        )
        low = analyzed_job("Machine learning")
        for job in [failed, low]:
            self.assertFalse(
                dashboard.review_inbox_view_matches(job, "Recommended", "Not tracked", "No package")
            )
            self.assertTrue(
                dashboard.review_inbox_view_matches(job, "Needs Review", "Not tracked", "No package")
            )

    def test_strong_match_requires_passed_confident_apply(self) -> None:
        strong = analyzed_job(
            "This fictional team needs an analyst for carefully reviewed local project work and stakeholder reports. "
            "Requirements include Python, SQL, pandas, machine learning, model evaluation, data analysis, "
            "data visualization, communication, documentation, and teamwork."
        )
        low = analyzed_job("Machine learning")
        failed = analyzed_job(
            "Senior Machine Learning Engineer requires Python SQL pandas machine learning model evaluation "
            "data analysis visualization communication and 5+ years experience required."
        )
        self.assertTrue(dashboard.is_strong_match(strong))
        self.assertFalse(dashboard.is_strong_match(low))
        self.assertFalse(dashboard.is_strong_match(failed))

    def test_tracker_arguments_use_canonical_values_and_title(self) -> None:
        job = analyzed_job(
            "Python SQL pandas data analysis visualization communication documentation teamwork required."
        )
        job.update(
            {
                "company": "Axle",
                "role": "Sample Job",
                "preview": "Role: Data Science Fellow - AI/NLP",
            }
        )
        args = dashboard.tracker_args_for_job(job)
        self.assertEqual(args.match_score, job["score"])
        self.assertEqual(args.recommendation, job["recommendation"])
        self.assertEqual(args.role, "Data Science Fellow - AI/NLP")
        self.assertIn("Eligibility:", args.notes)
        self.assertIn("Scoring confidence:", args.notes)

    def test_package_summary_parsing_accepts_current_and_legacy_reports(self) -> None:
        job = analyzed_job(
            "Python SQL pandas data analysis visualization communication documentation teamwork required."
        )
        current = (
            f"- Role Fit Score: **{job['score']}/100**\n"
            f"- Recommendation: **{job['recommendation']}**"
        )
        legacy = "- Match score: **74/100**\n- Recommendation: **Maybe Apply**"
        self.assertEqual(parse_analysis_summary(current), (job["score"], job["recommendation"]))
        self.assertEqual(parse_analysis_summary(legacy), (74, "Maybe Apply"))

    def test_cache_reuses_and_invalidates_on_content_change(self) -> None:
        job = {"path": "fictional.md", "canonical_job_key": "fictional-key"}
        fake_workspace = SimpleNamespace(mode="demo", root=Path("data/demo"), resume_source_path=None)
        fake_result = {
            "score": 50,
            "recommendation": "Maybe Apply",
            "score_breakdown": [{"active_terms": ["Python"]}],
            "eligibility": {"status": "passed", "reasons": []},
            "confidence": {"level": "low", "active_requirement_count": 1, "candidate_evidence_count": 1, "reasons": []},
        }
        session_state: dict[str, object] = {}
        with (
            patch.object(dashboard, "current_workspace", return_value=fake_workspace),
            patch.object(dashboard.st, "session_state", session_state),
            patch.object(dashboard, "analyze_job_structured", return_value=fake_result) as analyzer,
        ):
            dashboard.analyze_job_for_dashboard(job, "Python", "Python", use_cache=True)
            dashboard.analyze_job_for_dashboard(job, "Python", "Python", use_cache=True)
            dashboard.analyze_job_for_dashboard(job, "Python and SQL", "Python", use_cache=True)
        self.assertEqual(analyzer.call_count, 2)


class DashboardActionGuidanceTests(unittest.TestCase):
    def test_tracker_actions_change_with_pipeline_stage(self) -> None:
        self.assertIn("Review fit", dashboard.tracker_next_action({"status": "saved"}))
        self.assertIn("apply manually", dashboard.tracker_next_action({"status": "ready"}))
        self.assertIn("Prepare role-specific", dashboard.tracker_next_action({"status": "interview"}))

    def test_old_applied_role_is_flagged_for_follow_up(self) -> None:
        row = {"status": "applied", "applied_date": "2020-01-01"}
        self.assertTrue(dashboard.tracker_follow_up_due(row))
        self.assertIn("Follow up", dashboard.tracker_next_action(row))

    def test_review_action_prioritizes_evidence_and_eligibility(self) -> None:
        missing_analysis = {"analysis_available": False}
        self.assertIn("complete job description", dashboard.review_job_next_action(missing_analysis))

        failed = analyzed_job(
            "Senior Data Engineer requires Python SQL pandas data analysis communication "
            "documentation and 5+ years experience required."
        )
        self.assertIn("hard constraint", dashboard.review_job_next_action(failed))

        low_confidence = analyzed_job("Machine learning")
        self.assertIn("full job description", dashboard.review_job_next_action(low_confidence))


if __name__ == "__main__":
    unittest.main()
