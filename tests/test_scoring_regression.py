"""Deterministic golden, ranking, and invariant scoring regression tests."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_job import (  # noqa: E402
    DIRECT_MATCH_STRENGTH,
    calculate_match_score,
    calculate_score_breakdown,
    evaluate_eligibility,
    find_keywords,
    infer_candidate_experience_profile,
    parse_job_description,
    score_job_texts,
)


CASES_PATH = PROJECT_ROOT / "tests" / "fixtures" / "scoring_cases.json"


def recommendation_group(value: str) -> str:
    return {
        "Apply": "apply",
        "Apply / Maybe Apply": "apply",
        "Maybe Apply": "maybe",
        "Skip or Low Priority": "skip",
        "Skip / Not Eligible": "not_eligible",
        "Manual Review": "manual_review",
    }[value]


class ScoringGoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    def test_fixture_has_expected_case_count_and_unique_ids(self) -> None:
        self.assertGreaterEqual(len(self.cases), 15)
        self.assertLessEqual(len(self.cases), 25)
        case_ids = [case["case_id"] for case in self.cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_all_golden_cases(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case["case_id"]):
                result = score_job_texts(case["job_text"], case["candidate_text"])
                expected = case["expectations"]
                self.assertEqual(result["eligibility"]["status"], expected["eligibility"])
                self.assertEqual(result["confidence"]["level"], expected["confidence"])
                self.assertGreaterEqual(result["score"], expected["score_min"])
                self.assertLessEqual(result["score"], expected["score_max"])
                self.assertEqual(recommendation_group(result["recommendation"]), expected["recommendation_group"])
                reason_codes = {reason["code"] for reason in result["eligibility"]["reasons"]}
                self.assertTrue(set(expected["required_reason_codes"]).issubset(reason_codes))
                self.assertTrue(set(expected["forbidden_reason_codes"]).isdisjoint(reason_codes))

    def test_candidate_profile_inference_is_conservative(self) -> None:
        recent = infer_candidate_experience_profile(
            "Recent graduate with a bachelor's degree and internship experience."
        )
        self.assertEqual(recent["career_level"], "new_grad")
        self.assertEqual(recent["highest_degree"], "bachelor")
        self.assertIsNone(recent["years_experience"])

        explicit = infer_candidate_experience_profile(
            "Junior analyst with 2.5 years of professional experience and a master's degree."
        )
        self.assertEqual(explicit["career_level"], "junior")
        self.assertEqual(explicit["years_experience"], 2.5)
        self.assertEqual(explicit["highest_degree"], "master")

        unknown = infer_candidate_experience_profile("Python and SQL project portfolio.")
        self.assertEqual(unknown["career_level"], "unknown")
        self.assertEqual(unknown["highest_degree"], "unknown")

    def test_preferred_only_category_can_receive_full_credit(self) -> None:
        candidate = "Python pandas SQL data visualization"
        job = "These analytical skills are nice to have: Python, pandas, SQL, and data visualization."
        parsed = parse_job_description(job, candidate)
        breakdown = calculate_score_breakdown(
            parsed,
            ["Python", "pandas", "SQL", "data visualization"],
            infer_candidate_experience_profile(candidate),
        )
        technical = next(item for item in breakdown if item["category"] == "Core technical skills")
        self.assertEqual(technical["earned"], technical["possible"])
        self.assertEqual(technical["matched"], ["Python", "pandas", "SQL", "data visualization"])


class ScoringRankingTests(unittest.TestCase):
    candidate = (
        "Recent graduate with a bachelor's degree. Python pandas scikit-learn SQL machine learning "
        "model evaluation data visualization data analysis UAV route planning thermal data communication teamwork documentation."
    )

    def score(self, job: str, candidate: str | None = None) -> int:
        return int(score_job_texts(job, candidate or self.candidate)["score"])

    def test_new_grad_ml_ranks_above_senior_ml_manager(self) -> None:
        entry = "Entry-level new grad role requiring Python pandas scikit-learn machine learning model evaluation communication teamwork documentation."
        senior = "Senior Machine Learning Manager requiring Python pandas scikit-learn machine learning model evaluation communication teamwork documentation and 5+ years professional experience required."
        self.assertGreater(self.score(entry), self.score(senior))

    def test_uav_algorithm_ranks_above_unrelated_frontend(self) -> None:
        uav = "Junior UAV algorithm role requiring Python UAV route planning thermal data communication teamwork documentation."
        frontend = "Frontend role requiring presentation communication teamwork and documentation for browser interface styling."
        self.assertGreater(self.score(uav), self.score(frontend))

    def test_entry_data_role_ranks_above_five_year_role(self) -> None:
        entry = "Entry-level new grad data role requiring Python SQL pandas data analysis visualization communication documentation."
        mid = "Data role requiring Python SQL pandas data analysis visualization communication documentation and 5+ years professional experience required."
        self.assertGreater(self.score(entry), self.score(mid))

    def test_direct_domain_evidence_ranks_above_adjacent_only(self) -> None:
        job = "Junior role requiring Python robotics sensor data communication documentation and teamwork."
        direct = "Junior engineer with Python robotics sensor data communication documentation teamwork."
        adjacent = "Junior engineer with Python UAV route planning thermal data communication documentation teamwork."
        self.assertGreater(self.score(job, direct), self.score(job, adjacent))


class ScoringInvariantTests(unittest.TestCase):
    base_candidate = "Recent graduate with a bachelor's degree. Python SQL data analysis communication documentation teamwork."

    def result(self, job: str, candidate: str | None = None) -> dict[str, object]:
        return score_job_texts(job, candidate if candidate is not None else self.base_candidate)

    def test_adding_direct_required_match_never_reduces_score(self) -> None:
        job = "Entry-level role requires Python SQL pandas data analysis communication documentation."
        before = self.result(job)
        after = self.result(job, self.base_candidate + " pandas")
        self.assertGreaterEqual(after["score"], before["score"])

    def test_changing_missing_required_to_preferred_never_reduces_score(self) -> None:
        required = "Entry-level role. Required: Python SQL pandas data analysis communication documentation."
        preferred = "Entry-level role. Required: Python SQL data analysis communication documentation. Nice to have: pandas."
        self.assertGreaterEqual(self.result(preferred)["score"], self.result(required)["score"])

    def test_unrelated_boilerplate_does_not_materially_change_score(self) -> None:
        job = "Entry-level role requires Python SQL data analysis communication documentation."
        boilerplate = job + " Fictional employer values curiosity, provides equipment, and reviews applications manually."
        self.assertLessEqual(abs(self.result(job)["score"] - self.result(boilerplate)["score"]), 1)

    def test_reordering_job_lines_does_not_materially_change_score(self) -> None:
        lines = ["Entry-level role", "Python required", "SQL required", "data analysis required", "communication required"]
        self.assertLessEqual(abs(self.result("\n".join(lines))["score"] - self.result("\n".join(reversed(lines)))["score"]), 1)

    def test_reordering_candidate_sections_does_not_materially_change_score(self) -> None:
        job = "Entry-level role requires Python SQL data analysis communication documentation."
        sections = ["Recent graduate", "bachelor's degree", "Python SQL", "data analysis", "communication documentation"]
        self.assertLessEqual(
            abs(self.result(job, "\n".join(sections))["score"] - self.result(job, "\n".join(reversed(sections)))["score"]),
            1,
        )

    def test_unknown_experience_does_not_receive_full_credit(self) -> None:
        result = self.result(
            "Entry-level new grad role requiring Python SQL data analysis communication.",
            "Python SQL data analysis communication portfolio.",
        )
        experience = next(item for item in result["score_breakdown"] if item["category"] == "Experience level fit")
        self.assertEqual(experience["earned"], 0)

    def test_incidental_manager_and_senior_wording_is_not_hard_constraint(self) -> None:
        for phrase in ["Report to a manager.", "Support senior engineers."]:
            with self.subTest(phrase=phrase):
                eligibility = evaluate_eligibility(
                    f"Entry-level role requiring Python SQL communication. {phrase}",
                    self.base_candidate,
                )
                self.assertEqual(eligibility["status"], "passed")

    def test_preferred_years_is_not_required_years(self) -> None:
        preferred = evaluate_eligibility("Python analyst. 3 years experience preferred.", self.base_candidate)
        required = evaluate_eligibility("Python analyst. 3+ years experience required.", self.base_candidate)
        self.assertEqual(preferred["status"], "passed")
        self.assertEqual(required["status"], "failed")

    def test_low_requirement_count_produces_low_confidence(self) -> None:
        result = self.result("Python analyst wanted.")
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertEqual(result["coverage_score"], 100)
        self.assertEqual(result["score"], 62)

    def test_evidence_calibration_never_increases_a_weak_score(self) -> None:
        result = self.result("Unrelated role requiring pandas.")
        self.assertLessEqual(result["score"], result["coverage_score"])

    def test_truncated_api_description_stays_low_confidence(self) -> None:
        job = """# Data Analyst
Source: Jooble
## Job Description
Python SQL pandas data analysis communication role..."""
        result = self.result(job, "Python SQL pandas data analysis communication")
        self.assertEqual(result["coverage_score"], 100)
        self.assertLess(result["score"], result["coverage_score"])
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertTrue(result["confidence"]["job_description_quality"]["appears_incomplete"])

    def test_saved_title_focus_separates_domain_match_from_generic_ai_overlap(self) -> None:
        candidate = "Machine learning and model evaluation project portfolio."
        ml_job = """# Machine Learning Engineer
Source: Jooble
## Job Description
Machine learning role..."""
        physics_job = """# Physics Expert for AI Model Training
Source: Jooble
## Job Description
Machine learning role..."""
        aligned = self.result(ml_job, candidate)
        mismatched = self.result(physics_job, candidate)
        self.assertEqual(aligned["coverage_score"], mismatched["coverage_score"])
        self.assertEqual(aligned["role_alignment"]["score"], 100)
        self.assertEqual(mismatched["role_alignment"]["score"], 0)
        self.assertGreater(aligned["score"], mismatched["score"])

    def test_eligibility_failure_overrides_high_raw_fit(self) -> None:
        result = self.result(
            "Senior Data Engineer requires Python SQL data analysis communication documentation and 5+ years experience required."
        )
        self.assertGreaterEqual(result["score"], 50)
        self.assertEqual(result["eligibility"]["status"], "failed")
        self.assertEqual(result["recommendation"], "Skip / Not Eligible")

    def test_manual_review_never_becomes_unconditional_apply(self) -> None:
        result = self.result(
            "Python SQL data analysis communication documentation role. Visa sponsorship may be available after review."
        )
        self.assertEqual(result["eligibility"]["status"], "manual_review")
        self.assertEqual(result["recommendation"], "Manual Review")

    def test_explicitly_incompatible_authorization_can_fail(self) -> None:
        eligibility = evaluate_eligibility(
            "Applicants must currently have the right to work in the UK. No visa sponsorship is available.",
            "Candidate explicitly requires visa sponsorship.",
        )
        self.assertEqual(eligibility["status"], "failed")
        self.assertIn("work_authorization", {reason["code"] for reason in eligibility["reasons"]})

    def test_unknown_required_experience_requires_manual_review(self) -> None:
        eligibility = evaluate_eligibility(
            "Applicants must have 3+ years of professional experience required.",
            "Candidate portfolio lists Python and SQL but no career stage or years.",
        )
        self.assertEqual(eligibility["status"], "manual_review")

    def test_scores_are_bounded(self) -> None:
        for job in ["", "Python", "Python SQL pandas machine learning model evaluation data analysis communication teamwork documentation"]:
            score = self.result(job)["score"]
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 100)

    def test_inactive_categories_are_excluded_from_denominator(self) -> None:
        parsed = parse_job_description("Python required", "Python")
        breakdown = calculate_score_breakdown(parsed, ["Python"], infer_candidate_experience_profile("Python"))
        inactive = [item for item in breakdown if item["category"] != "Core technical skills"]
        self.assertTrue(all(item["earned"] is None for item in inactive))
        self.assertEqual(calculate_match_score(breakdown, []), 100)

    def test_direct_match_strength_constant_is_full(self) -> None:
        self.assertEqual(DIRECT_MATCH_STRENGTH, 1.0)

    def test_business_analysis_requirements_are_direct_matches(self) -> None:
        candidate = (
            "Business analysis, requirements gathering, stakeholder management, "
            "process improvement, Excel, presentation, and communication experience."
        )
        job = (
            "Business Analyst requiring business analysis, requirements gathering, "
            "stakeholder management, process improvement, Excel, presentation, and communication."
        )
        result = score_job_texts(job, candidate)
        matched = {
            keyword
            for category in result["score_breakdown"]
            for keyword in category["matched"]
        }
        self.assertTrue(
            {
                "business analysis",
                "requirements gathering",
                "stakeholder management",
                "process improvement",
                "Excel",
                "presentation",
                "communication",
            }.issubset(matched)
        )

    def test_broad_business_and_management_words_do_not_match_ba_skills(self) -> None:
        keywords = set(
            find_keywords("Worked in a business environment and reported to management.")
        )
        self.assertTrue(
            {
                "business analysis",
                "requirements gathering",
                "stakeholder management",
                "process improvement",
            }.isdisjoint(keywords)
        )

    def test_adding_business_analysis_direct_match_never_reduces_score(self) -> None:
        job = (
            "Business Analyst requiring SQL, Excel, business analysis, requirements "
            "gathering, stakeholder management, process improvement, communication, "
            "presentation, and documentation."
        )
        candidate = "SQL Excel communication presentation documentation"
        before = score_job_texts(job, candidate)["score"]
        after = score_job_texts(job, candidate + " business analysis")["score"]
        self.assertGreaterEqual(after, before)


if __name__ == "__main__":
    unittest.main()
