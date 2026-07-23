"""Tests for template-grouped annotation baseline evaluation."""

from __future__ import annotations

import unittest

from ml.annotation_dataset import build_training_pairs
from ml.annotation_experiment import run_annotation_experiment


def reviewed_tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    concepts = ["sql", "python", "docker", "tableau", "kubernetes", "forecasting"]
    for index, concept in enumerate(concepts):
        group = f"template group {index}"
        tasks.append(
            {
                "task_id": f"positive-{index}",
                "role_family": "Data",
                "requirement": f"Use {concept} in practical delivery.",
                "candidates": [
                    {
                        "candidate_id": f"positive-{index}-a",
                        "evidence": f"Built a production workflow using {concept}.",
                    },
                    {
                        "candidate_id": f"positive-{index}-b",
                        "evidence": "Prepared unrelated meeting notes.",
                    },
                ],
                "selected_candidate_id": f"positive-{index}-a",
                "support_label": "Direct",
                "template_group": group,
                "fold": index,
            }
        )
        tasks.append(
            {
                "task_id": f"negative-{index}",
                "role_family": "Data",
                "requirement": f"Deploy {concept} for customer workloads.",
                "candidates": [
                    {
                        "candidate_id": f"negative-{index}-a",
                        "evidence": "Prepared unrelated meeting notes.",
                    },
                    {
                        "candidate_id": f"negative-{index}-b",
                        "evidence": "Reviewed a general project schedule.",
                    },
                ],
                "selected_candidate_id": None,
                "support_label": "No Support",
                "template_group": group,
                "fold": index,
            }
        )
    return tasks


class AnnotationExperimentTests(unittest.TestCase):
    def test_experiment_evaluates_every_template_group_and_method(self) -> None:
        tasks = reviewed_tasks()
        pairs = build_training_pairs(tasks)
        report = run_annotation_experiment(tasks, pairs, random_state=9)
        self.assertEqual(report["task_count"], 12)
        self.assertEqual(report["pair_count"], 18)
        self.assertEqual(report["template_group_count"], 6)
        self.assertEqual(
            set(report["methods"]),
            {
                "concept_lexical_rule",
                "tfidf_cosine",
                "lsa_embedding",
                "trained_pair_classifier",
                "hybrid_lsa_reranker",
                "lexical_guarded_reranker",
                "pairwise_hybrid_reranker",
            },
        )
        for result in report["methods"].values():
            self.assertEqual(result["pair_classification"]["examples"], 18)
            self.assertEqual(result["retrieval"]["support_tasks"], 6)
            self.assertEqual(result["retrieval"]["no_support_tasks"], 6)
            self.assertGreaterEqual(result["retrieval"]["recall_at_1"], 0.0)
            self.assertLessEqual(result["retrieval"]["recall_at_1"], 1.0)
            if "error_analysis" in result:
                self.assertGreaterEqual(result["error_analysis"]["pair_error_count"], 0)
        trained = report["methods"]["trained_pair_classifier"]
        self.assertEqual(len(trained["folds"]), 6)
        self.assertIn("hybrid_lsa_reranker", report["method_threshold_medians"])
        self.assertIn("lexical_guarded_reranker", report["method_threshold_medians"])
        self.assertIn("pairwise_hybrid_reranker", report["method_threshold_medians"])
        self.assertIn(report["model_selection"]["selected_method"], report["methods"])
        self.assertEqual(
            report["model_selection"]["promotion_status"],
            "blocked_until_fixed_real_holdout",
        )
        self.assertIn("experimental", report["interpretation"].lower())


if __name__ == "__main__":
    unittest.main()
