"""Validation and regression tests for the public scoring benchmark."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analyze_job import score_job_texts  # noqa: E402
from evaluate_scoring import DEFAULT_FIXTURE, evaluate_cases, load_cases  # noqa: E402


EXPECTED_FIELDS = {"score_min", "score_max", "eligibility", "confidence", "recommendation"}
ELIGIBILITY_VALUES = {"passed", "manual_review", "failed"}
CONFIDENCE_VALUES = {"low", "medium", "high"}
RECOMMENDATION_VALUES = {
    "Apply",
    "Apply / Maybe Apply",
    "Maybe Apply",
    "Manual Review",
    "Skip or Low Priority",
    "Skip / Not Eligible",
}
ROLE_FAMILIES = {
    "Data Analyst",
    "AI / ML Analyst",
    "Machine Learning Engineer",
    "Python / Software",
    "UAV / Robotics",
    "Business / Quantitative Analyst",
}


class ScoringBenchmarkFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(DEFAULT_FIXTURE)

    def test_all_fixtures_parse(self) -> None:
        self.assertGreaterEqual(len(self.cases), 40)
        self.assertLessEqual(len(self.cases), 60)
        for case in self.cases:
            with self.subTest(case=case.get("id")):
                self.assertTrue({"id", "role_family", "candidate_text", "job_text", "expected", "reason"} <= case.keys())
                self.assertIsInstance(case["candidate_text"], str)
                self.assertIsInstance(case["job_text"], str)
                self.assertTrue(case["reason"].strip())

    def test_case_ids_are_unique(self) -> None:
        case_ids = [case["id"] for case in self.cases]
        self.assertEqual(len(case_ids), len(set(case_ids)))

    def test_expected_fields_are_complete_and_valid(self) -> None:
        for case in self.cases:
            with self.subTest(case=case["id"]):
                expected = case["expected"]
                self.assertEqual(set(expected), EXPECTED_FIELDS)
                self.assertIsInstance(expected["score_min"], int)
                self.assertIsInstance(expected["score_max"], int)
                self.assertLessEqual(0, expected["score_min"])
                self.assertLessEqual(expected["score_min"], expected["score_max"])
                self.assertLessEqual(expected["score_max"], 100)
                self.assertIn(expected["eligibility"], ELIGIBILITY_VALUES)
                self.assertIn(expected["confidence"], CONFIDENCE_VALUES)
                self.assertIn(expected["recommendation"], RECOMMENDATION_VALUES)

    def test_all_role_families_and_decision_states_are_covered(self) -> None:
        self.assertEqual({case["role_family"] for case in self.cases}, ROLE_FAMILIES)
        self.assertEqual({case["expected"]["eligibility"] for case in self.cases}, ELIGIBILITY_VALUES)
        self.assertEqual({case["expected"]["confidence"] for case in self.cases}, CONFIDENCE_VALUES)
        expected_recommendations = {case["expected"]["recommendation"] for case in self.cases}
        self.assertEqual(expected_recommendations, RECOMMENDATION_VALUES)


class ScoringBenchmarkRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_cases(DEFAULT_FIXTURE)
        cls.summary = evaluate_cases(cls.cases)

    def test_complete_benchmark_agrees_with_reviewed_expectations(self) -> None:
        failure_details = {
            result["id"]: result["failures"] for result in self.summary["failed"]
        }
        self.assertEqual(failure_details, {})
        self.assertEqual(self.summary["hard_constraint_false_negatives"], 0)
        self.assertEqual(self.summary["unsafe_high_score_false_positives"], 0)

    def test_hard_eligibility_mismatch_never_recommends_apply(self) -> None:
        failed_cases = [case for case in self.cases if case["expected"]["eligibility"] == "failed"]
        self.assertTrue(failed_cases)
        for case in failed_cases:
            with self.subTest(case=case["id"]):
                result = score_job_texts(case["job_text"], case["candidate_text"])
                self.assertEqual(result["eligibility"]["status"], "failed")
                self.assertNotIn(result["recommendation"], {"Apply", "Apply / Maybe Apply"})

    def test_low_confidence_never_recommends_apply_without_review(self) -> None:
        low_confidence_cases = [case for case in self.cases if case["expected"]["confidence"] == "low"]
        self.assertTrue(low_confidence_cases)
        for case in low_confidence_cases:
            with self.subTest(case=case["id"]):
                result = score_job_texts(case["job_text"], case["candidate_text"])
                self.assertEqual(result["confidence"]["level"], "low")
                self.assertEqual(result["recommendation"], "Manual Review")

    def test_one_of_one_match_is_not_a_high_confidence_perfect_fit(self) -> None:
        result = score_job_texts("Python analyst wanted.", "Python project portfolio.")
        self.assertEqual(result["score"], 100)
        self.assertEqual(result["confidence"]["active_requirement_count"], 1)
        self.assertEqual(result["confidence"]["level"], "low")
        self.assertEqual(result["recommendation"], "Manual Review")


if __name__ == "__main__":
    unittest.main()
