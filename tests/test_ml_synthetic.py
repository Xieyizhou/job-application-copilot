"""Tests for synthetic dataset conversion and split policy."""

from __future__ import annotations

import unittest

from ml.synthetic import render_job, render_resume, stable_job_subset


class SyntheticDataTests(unittest.TestCase):
    def test_renderers_include_structured_evidence(self) -> None:
        resume = render_resume({"role": "Data Analyst", "skills": ["Python", "SQL"]})
        job = render_job({"job_title": "Data Analyst", "must_have_skills": ["SQL"]})
        self.assertIn("Data Analyst", resume)
        self.assertIn("Python. SQL", resume)
        self.assertIn("Must have skills: SQL", job)

    def test_job_split_is_deterministic(self) -> None:
        self.assertEqual(stable_job_subset("job-123"), stable_job_subset("job-123"))
        self.assertIn(stable_job_subset("job-123"), {"train", "validation", "test"})


if __name__ == "__main__":
    unittest.main()
