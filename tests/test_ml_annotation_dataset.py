"""Tests for exporting reviewed requirement/evidence annotations."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from ml.annotation_dataset import (
    AnnotationDatasetError,
    build_annotated_tasks,
    build_training_pairs,
    dataset_manifest,
    requirement_template_group,
    write_jsonl,
)


def task(
    task_id: str,
    requirement: str,
    *,
    duplicate_of: str | None = None,
) -> dict[str, object]:
    return {
        "task_id": task_id,
        "role_family": "Data",
        "requirement": requirement,
        "source_dataset": "test_fixture",
        "blind_duplicate_of": duplicate_of,
        "candidates": [
            {"candidate_id": f"{task_id}-a", "evidence": "Built SQL reporting workflows."},
            {"candidate_id": f"{task_id}-b", "evidence": "Prepared project notes."},
        ],
    }


class AnnotationDatasetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tasks = [
            task("direct", "The role calls for practical SQL delivery."),
            task("none", "Candidates should be able to deploy Kubernetes."),
            task("partial", "Success in this role depends on analytics."),
        ]
        self.states = {
            "direct": {
                "action": "label",
                "support_label": "Direct",
                "selected_candidate_id": "direct-a",
                "cover_letter_safe": True,
            },
            "none": {
                "action": "label",
                "support_label": "No Support",
                "selected_candidate_id": None,
                "cover_letter_safe": None,
            },
            "partial": {
                "action": "label",
                "support_label": "Partial",
                "selected_candidate_id": "partial-a",
                "cover_letter_safe": False,
            },
        }

    def test_export_excludes_repeats_and_only_uses_high_confidence_pairs(self) -> None:
        repeat = task(
            "direct-repeat",
            "The role calls for practical SQL delivery.",
            duplicate_of="direct",
        )
        repeat["candidates"] = self.tasks[0]["candidates"]
        tasks = [*self.tasks, repeat]
        states = {
            **self.states,
            "direct-repeat": {
                **self.states["direct"],
                "selected_candidate_id": "direct-a",
            },
        }
        annotated = build_annotated_tasks(tasks, states)
        pairs = build_training_pairs(annotated)
        self.assertEqual(len(annotated), 3)
        self.assertEqual(len(pairs), 4)
        self.assertEqual([row["binary_label"] for row in pairs].count(1), 2)
        self.assertEqual([row["binary_label"] for row in pairs].count(0), 2)
        self.assertNotIn("Prepared project notes.", [row["evidence"] for row in pairs if row["binary_label"]])
        manifest = dataset_manifest(annotated, pairs)
        self.assertEqual(manifest["unique_tasks"], 3)
        self.assertEqual(manifest["training_pairs"], 4)

    def test_export_rejects_conflicts_unresolved_labels_and_invalid_selection(self) -> None:
        repeat = task(
            "direct-repeat",
            "The role calls for practical SQL delivery.",
            duplicate_of="direct",
        )
        repeat["candidates"] = self.tasks[0]["candidates"]
        conflict_states = {
            **self.states,
            "direct-repeat": {
                "action": "label",
                "support_label": "Partial",
                "selected_candidate_id": "direct-a",
            },
        }
        with self.assertRaisesRegex(AnnotationDatasetError, "conflicts"):
            build_annotated_tasks([*self.tasks, repeat], conflict_states)

        unresolved = {**self.states, "partial": {"action": "skip"}}
        with self.assertRaisesRegex(AnnotationDatasetError, "not labeled"):
            build_annotated_tasks(self.tasks, unresolved)

        invalid = {
            **self.states,
            "direct": {**self.states["direct"], "selected_candidate_id": None},
        }
        with self.assertRaisesRegex(AnnotationDatasetError, "selected supporting evidence"):
            build_annotated_tasks(self.tasks, invalid)

    def test_template_group_and_jsonl_output_are_deterministic(self) -> None:
        self.assertEqual(
            requirement_template_group("The role calls for practical SQL delivery."),
            "the role calls",
        )
        annotated = build_annotated_tasks(self.tasks, self.states, random_state=7)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rows.jsonl"
            write_jsonl(annotated, path)
            first = path.read_text(encoding="utf-8")
            write_jsonl(annotated, path)
            self.assertEqual(path.read_text(encoding="utf-8"), first)

    def test_partial_export_keeps_only_completed_resolved_tasks(self) -> None:
        self.tasks[0]["source_resume_hash"] = "resume-hash"
        self.tasks[0]["source_job_hash"] = "job-hash"
        self.tasks[0]["semantic_case_group_id"] = "semantic-group"
        partial_states = {
            "direct": self.states["direct"],
            "partial": {"action": "skip"},
        }

        annotated = build_annotated_tasks(
            self.tasks,
            partial_states,
            require_complete=False,
        )
        manifest = dataset_manifest(
            annotated,
            build_training_pairs(annotated),
            dataset_name="reviewed_v3_seed",
            source_queue_complete=False,
        )

        self.assertEqual([row["task_id"] for row in annotated], ["direct"])
        self.assertEqual(annotated[0]["source_resume_hash"], "resume-hash")
        self.assertEqual(annotated[0]["source_job_hash"], "job-hash")
        self.assertEqual(annotated[0]["semantic_case_group_id"], "semantic-group")
        self.assertEqual(manifest["dataset_name"], "reviewed_v3_seed")
        self.assertFalse(manifest["source_queue_complete"])


if __name__ == "__main__":
    unittest.main()
