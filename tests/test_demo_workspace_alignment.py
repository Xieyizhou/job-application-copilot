"""Regression coverage for the fictional, read-only Demo scoring workspace."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from docx import Document


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import dashboard
from apply_package import parse_analysis_summary, parse_job_metadata
from workspace import WorkspaceError, demo_workspace


DEMO_JOB_DIR = PROJECT_ROOT / "data" / "demo" / "jobs"
DEMO_CANDIDATE = PROJECT_ROOT / "data" / "resume" / "resume_source.example.md"
DEMO_PACKAGE = PROJECT_ROOT / "data" / "demo" / "sample_package"
EXPECTED_JOBS = {"ai_analyst.md", "data_analyst.md", "machine_learning_intern.md"}


def load_demo_jobs() -> list[dict[str, object]]:
    with patch.object(dashboard, "demo_mode_enabled", return_value=True):
        return dashboard.load_screened_jobs()


class DemoWorkspaceAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.jobs = load_demo_jobs()
        cls.jobs_by_role = {str(job["role"]): job for job in cls.jobs}

    def test_demo_defaults_to_all_jobs_and_personal_keeps_recommended(self) -> None:
        self.assertEqual(dashboard.default_review_inbox_view(self.jobs, [], demo=True), "All Jobs")
        with (
            patch.object(dashboard, "tracker_status_for_job", return_value="Not tracked"),
            patch.object(dashboard, "package_status_for_job", return_value="No package"),
        ):
            self.assertEqual(
                dashboard.default_review_inbox_view([self.jobs_by_role["Data Analyst"]], [], demo=False),
                "Recommended",
            )

    def test_exactly_three_intended_demo_jobs_load_with_real_titles(self) -> None:
        self.assertEqual({path.name for path in DEMO_JOB_DIR.glob("*.md")}, EXPECTED_JOBS)
        self.assertEqual(len(self.jobs), 3)
        self.assertEqual(set(self.jobs_by_role), {"Data Analyst", "Machine Learning Intern", "Senior Data Scientist"})
        self.assertNotIn("Sample Job", set(self.jobs_by_role))

    def test_fictional_candidate_is_complete_and_has_no_personal_identifiers(self) -> None:
        candidate = DEMO_CANDIDATE.read_text(encoding="utf-8")
        for evidence in ["Bachelor's degree in Data Science", "Python", "SQL", "scikit-learn", "cross-validation"]:
            self.assertIn(evidence, candidate)
        normalized = candidate.lower()
        for forbidden in ["ucsd", "xieyizhou", "github.com", "linkedin.com", "@"]:
            self.assertNotIn(forbidden, normalized)

    def test_three_jobs_have_distinct_canonical_outcomes(self) -> None:
        analyst = self.jobs_by_role["Data Analyst"]
        intern = self.jobs_by_role["Machine Learning Intern"]
        senior = self.jobs_by_role["Senior Data Scientist"]

        self.assertTrue(analyst["analysis_available"])
        self.assertEqual(analyst["eligibility"]["status"], "passed")
        self.assertIn(analyst["recommendation"], {"Apply", "Apply / Maybe Apply", "Maybe Apply"})
        self.assertIn(analyst["confidence"]["level"], {"medium", "high"})

        self.assertTrue(intern["analysis_available"])
        self.assertEqual(intern["recommendation"], "Manual Review")
        self.assertEqual(intern["eligibility"]["status"], "manual_review")
        self.assertIn("robotics", intern["analysis_result"]["missing_skills"])
        self.assertTrue(intern["analysis_result"]["parsed_job"]["preferred_skills"])

        self.assertEqual(senior["recommendation"], "Skip / Not Eligible")
        self.assertEqual(senior["eligibility"]["status"], "failed")
        reason_codes = {reason["code"] for reason in senior["eligibility"]["reasons"]}
        self.assertIn("minimum_experience", reason_codes)
        self.assertIn("seniority_requirement", reason_codes)

    def test_demo_inbox_membership_and_dashboard_counts_follow_canonical_rules(self) -> None:
        recommended = [
            job for job in self.jobs
            if dashboard.review_inbox_view_matches(job, "Recommended", "Demo only", "Demo package")
        ]
        needs_review = [
            job for job in self.jobs
            if dashboard.review_inbox_view_matches(job, "Needs Review", "Demo only", "Demo package")
        ]
        all_jobs = [
            job for job in self.jobs
            if dashboard.review_inbox_view_matches(job, "All Jobs", "Demo only", "Demo package")
        ]
        self.assertEqual([job["role"] for job in recommended], ["Data Analyst"])
        self.assertEqual([job["role"] for job in needs_review], ["Machine Learning Intern"])
        self.assertEqual(len(all_jobs), 3)
        self.assertEqual(len(self.jobs), len(EXPECTED_JOBS))
        self.assertEqual(sum(dashboard.is_strong_match(job) for job in self.jobs), 1)
        with patch.object(dashboard, "demo_mode_enabled", return_value=True):
            package_statuses = {job["role"]: dashboard.package_status_for_job(job, []) for job in self.jobs}
        self.assertEqual(package_statuses["Data Analyst"], "Demo package")
        self.assertEqual(package_statuses["Machine Learning Intern"], "Demo only")
        self.assertEqual(package_statuses["Senior Data Scientist"], "Demo only")

    def test_demo_package_matches_live_data_analyst_analysis(self) -> None:
        analyst = self.jobs_by_role["Data Analyst"]
        report = (DEMO_PACKAGE / "analysis.md").read_text(encoding="utf-8")
        score, recommendation = parse_analysis_summary(report)
        metadata = parse_job_metadata(DEMO_JOB_DIR / "data_analyst.md")
        self.assertEqual((metadata["company"], metadata["role"]), (analyst["company"], analyst["role"]))
        self.assertEqual(score, analyst["score"])
        self.assertEqual(recommendation, analyst["recommendation"])
        self.assertIn(f"Eligibility: **{str(analyst['eligibility']['status']).title()}**", report)
        self.assertIn(f"Scoring Confidence: **{str(analyst['confidence']['level']).title()}**", report)
        terms = dashboard.summarize_analysis_requirements(analyst["analysis_result"])
        for term in terms["matched_required"] + terms["missing_required"]:
            self.assertRegex(report, rf"(?i)\b{re.escape(term)}\b")

    def test_demo_package_is_fictional_and_workspace_remains_read_only(self) -> None:
        package_text = "\n".join(path.read_text(encoding="utf-8") for path in DEMO_PACKAGE.glob("*.md"))
        for path in DEMO_PACKAGE.glob("*.docx"):
            package_text += "\n" + "\n".join(p.text for p in Document(path).paragraphs)
        normalized = package_text.lower()
        for forbidden in ["ucsd", "xieyizhou", "/users/", "github.com"]:
            self.assertNotIn(forbidden, normalized)
        self.assertIn("demo", normalized)
        workspace = demo_workspace()
        self.assertTrue(workspace.read_only)
        self.assertIsNone(workspace.tracker_database_path)
        with self.assertRaises(WorkspaceError):
            workspace.require_writable()


if __name__ == "__main__":
    unittest.main()
