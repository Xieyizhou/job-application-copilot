"""Tests for local resume-derived job-search suggestions."""

from __future__ import annotations

import unittest

from dashboard_search_profile import infer_search_profile


class DashboardSearchProfileTests(unittest.TestCase):
    def test_machine_learning_resume_prefers_ml_engineering(self) -> None:
        profile = infer_search_profile(
            "Python machine learning pipelines, neural network training, "
            "reinforcement learning, scikit-learn, and model evaluation."
        )
        self.assertEqual(profile.query, "Machine Learning Engineer")
        self.assertIn("machine learning", profile.keywords)

    def test_analytics_resume_prefers_data_analyst(self) -> None:
        profile = infer_search_profile(
            "Data analysis internship using SQL, Excel, dashboards, and data visualization."
        )
        self.assertEqual(profile.query, "Data Analyst")

    def test_empty_resume_uses_neutral_fallback(self) -> None:
        profile = infer_search_profile("")
        self.assertEqual(profile.query, "Entry Level")
        self.assertFalse(profile.source_ready)


if __name__ == "__main__":
    unittest.main()
