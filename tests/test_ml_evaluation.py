"""Tests for ML evaluation helpers."""

from __future__ import annotations

import unittest

from ml.evaluation import classification_metrics, mean_job_average_precision, select_f1_threshold


class EvaluationTests(unittest.TestCase):
    def test_threshold_and_metrics_for_separable_data(self) -> None:
        labels = [0, 0, 1, 1]
        probabilities = [0.1, 0.2, 0.8, 0.9]
        threshold = select_f1_threshold(labels, probabilities)
        metrics = classification_metrics(labels, probabilities, threshold=threshold)
        self.assertGreaterEqual(threshold, 0.2)
        self.assertEqual(metrics["f1"], 1.0)
        self.assertEqual(metrics["confusion_matrix"], [[2, 0], [0, 2]])

    def test_mean_job_average_precision(self) -> None:
        score = mean_job_average_precision(
            ["job-a", "job-a", "job-b", "job-b"],
            [1, 0, 1, 0],
            [0.9, 0.1, 0.8, 0.2],
        )
        self.assertEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
