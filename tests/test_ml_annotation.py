"""Tests for local requirement/evidence annotation generation and storage."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import run_annotation
from ml import annotation_dashboard
from ml.annotation import (
    annotation_summary,
    append_event,
    latest_task_states,
    load_jsonl,
    load_queue,
    repeat_conflict_task_ids,
    write_queue,
)
from ml.annotation_audit import audit_annotations
from ml.annotation_generation import (
    build_tasks_from_records,
    infer_role_family,
    sanitize_snippet,
)
from ml.annotation_scenarios import STRATA, fictional_challenge_records


def sample_records() -> list[dict[str, str]]:
    families = {
        "Data": "Data analyst requires SQL dashboard reporting and analytics skills.",
        "ML": "Machine learning engineer must build model training and evaluation pipelines.",
        "Software": "Software backend developer must build reliable API services and databases.",
        "Business": "Business product analyst must manage stakeholder strategy and operations.",
    }
    records = []
    for family, requirement in families.items():
        for index in range(2):
            records.append(
                {
                    "resume_text": (
                        "Built Python and SQL reporting systems for operational teams. "
                        "Developed reliable API services with automated quality checks. "
                        "Evaluated machine learning models with cross validation metrics. "
                        "Managed product research and presented findings to stakeholders."
                    ),
                    "job_text": f"{requirement.removesuffix('.')} for cohort {index}.",
                    "resume_hash": f"resume-{family}-{index}",
                    "job_hash": f"job-{family}-{index}",
                    "label": "Unreviewed",
                    "source_dataset": "test_fixture",
                }
            )
    return records


class AnnotationGenerationTests(unittest.TestCase):
    def test_balanced_queue_contains_blind_repeats_without_full_documents(self) -> None:
        tasks = build_tasks_from_records(
            sample_records(),
            unique_count=8,
            blind_repeat_fraction=0.25,
            random_state=7,
        )
        unique = [task for task in tasks if not task["blind_duplicate_of"]]
        self.assertEqual(len(unique), 8)
        self.assertEqual(len(tasks), 10)
        self.assertEqual({task["role_family"] for task in unique}, {"Data", "ML", "Software", "Business"})
        self.assertTrue(all("resume_text" not in task and "job_text" not in task for task in tasks))
        self.assertTrue(all(len(task["candidates"]) >= 3 for task in tasks))
        self.assertTrue(
            all(
                "retrieval_rank" not in candidate and "retrieval_similarity" not in candidate
                for task in tasks
                for candidate in task["candidates"]
            )
        )

    def test_role_priority_and_contact_redaction(self) -> None:
        self.assertEqual(infer_role_family("Machine learning and data analytics"), "ML")
        email = "name" + "@" + "example.com"
        phone = "+1 555" + " 222 3333"
        cleaned = sanitize_snippet(f"Built reports. {email} https://example.com {phone}")
        self.assertNotIn(email, cleaned)
        self.assertNotIn("https://", cleaned)
        self.assertNotIn("555", cleaned)


class AnnotationStorageTests(unittest.TestCase):
    def test_append_only_events_replay_and_clear(self) -> None:
        tasks = build_tasks_from_records(sample_records(), unique_count=8, blind_repeat_fraction=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue_path = root / "queue.jsonl"
            events_path = root / "events.jsonl"
            write_queue(tasks, queue_path)
            loaded = load_queue(queue_path)
            task = loaded[0]
            candidate_id = task["candidates"][0]["candidate_id"]
            append_event(
                task["task_id"],
                "label",
                events_path=events_path,
                selected_candidate_id=candidate_id,
                support_label="Direct",
                cover_letter_safe=True,
            )
            states = latest_task_states(load_jsonl(events_path))
            self.assertEqual(states[task["task_id"]]["support_label"], "Direct")
            self.assertEqual(annotation_summary(loaded, states)["completed"], 1)
            append_event(task["task_id"], "clear", events_path=events_path)
            self.assertNotIn(task["task_id"], latest_task_states(load_jsonl(events_path)))

    def test_blind_repeat_agreement_uses_label_and_evidence(self) -> None:
        tasks = build_tasks_from_records(
            sample_records(),
            unique_count=8,
            blind_repeat_fraction=0.25,
            random_state=11,
        )
        repeat = next(task for task in tasks if task["blind_duplicate_of"])
        original = next(task for task in tasks if task["task_id"] == repeat["blind_duplicate_of"])
        candidate_id = str(original["candidates"][0]["candidate_id"])
        states = {
            original["task_id"]: {
                "action": "label",
                "support_label": "Direct",
                "selected_candidate_id": candidate_id,
            },
            repeat["task_id"]: {
                "action": "label",
                "support_label": "Direct",
                "selected_candidate_id": candidate_id,
            },
        }
        summary = annotation_summary(tasks, states)
        self.assertEqual(summary["repeat_pairs"], 1)
        self.assertEqual(summary["repeat_agreement"], 1.0)
        self.assertEqual(repeat_conflict_task_ids(tasks, states), set())

        states[repeat["task_id"]]["support_label"] = "Partial"
        self.assertEqual(
            repeat_conflict_task_ids(tasks, states),
            {original["task_id"], repeat["task_id"]},
        )

    def test_dashboard_label_validation_and_local_save(self) -> None:
        task = build_tasks_from_records(sample_records(), unique_count=8, blind_repeat_fraction=0)[0]
        with tempfile.TemporaryDirectory() as directory:
            events_path = Path(directory) / "events.jsonl"
            with patch.object(annotation_dashboard, "EVENTS_PATH", events_path):
                error = annotation_dashboard.save_label(task, "__none__", "Direct", True, "")
                self.assertIsNotNone(error)
                candidate_id = task["candidates"][0]["candidate_id"]
                error = annotation_dashboard.save_label(task, candidate_id, "Partial", False, "reviewed")
                self.assertIsNone(error)
            events = load_jsonl(events_path)
            self.assertEqual(events[-1]["support_label"], "Partial")
            self.assertFalse(events[-1]["cover_letter_safe"])

    def test_annotation_launcher_targets_standalone_page(self) -> None:
        argv = run_annotation.build_streamlit_argv(["--server.port", "8510"])
        self.assertEqual(argv[:2], ["streamlit", "run"])
        self.assertTrue(argv[2].endswith("src/ml/annotation_dashboard.py"))
        self.assertEqual(argv[-2:], ["--server.port", "8510"])


class AnnotationDashboardTests(unittest.TestCase):
    def test_views_filters_and_candidate_labels(self) -> None:
        tasks = build_tasks_from_records(sample_records(), unique_count=8, blind_repeat_fraction=0)
        task = tasks[0]
        candidate_id = str(task["candidates"][0]["candidate_id"])
        direct_state = {"action": "label", "support_label": "Direct"}
        uncertain_state = {"action": "label", "support_label": "Uncertain"}
        skipped_state = {"action": "skip"}
        self.assertTrue(annotation_dashboard.task_matches_view(task, None, "Unlabeled"))
        self.assertTrue(annotation_dashboard.task_matches_view(task, direct_state, "Completed"))
        self.assertTrue(annotation_dashboard.task_matches_view(task, uncertain_state, "Uncertain"))
        self.assertTrue(annotation_dashboard.task_matches_view(task, skipped_state, "Skipped"))
        self.assertTrue(annotation_dashboard.task_matches_view(task, direct_state, "All"))
        self.assertFalse(annotation_dashboard.task_matches_view(task, direct_state, "Unlabeled"))
        visible = annotation_dashboard.visible_tasks(
            tasks,
            {},
            view="Unlabeled",
            role_family=str(task["role_family"]),
        )
        self.assertTrue(visible)
        self.assertTrue(all(item["role_family"] == task["role_family"] for item in visible))
        self.assertEqual(
            annotation_dashboard.visible_tasks(
                tasks,
                {},
                view="Conflicts",
                role_family="All",
            ),
            [],
        )
        self.assertIn("—", annotation_dashboard.candidate_label(task, candidate_id))
        self.assertIn(
            "no candidate",
            annotation_dashboard.candidate_label(task, annotation_dashboard.NONE_CANDIDATE),
        )

    def test_compact_rendering_and_missing_queue_message(self) -> None:
        task = build_tasks_from_records(sample_records(), unique_count=8, blind_repeat_fraction=0)[0]
        ui = MagicMock()
        ui.columns.side_effect = lambda value: [MagicMock() for _ in range(value)]
        ui.radio.return_value = annotation_dashboard.NONE_CANDIDATE
        ui.checkbox.return_value = False
        ui.expander.return_value = MagicMock()
        ui.button.return_value = False
        with patch.object(annotation_dashboard, "st", ui):
            annotation_dashboard.render_progress(
                {"total": 8, "completed": 2, "remaining": 6, "repeat_agreement": 1.0}
            )
            annotation_dashboard.render_task(task, None)
        ui.progress.assert_called_once_with(0.25)
        ui.radio.assert_called_once()

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.jsonl"
            with patch.object(annotation_dashboard, "st", ui), patch.object(
                annotation_dashboard,
                "QUEUE_PATH",
                missing,
            ):
                annotation_dashboard.main()
        ui.warning.assert_called_with("No local annotation queue was found.")


class AnnotationV2BiasTests(unittest.TestCase):
    def test_v2_scenarios_balance_challenges_and_candidate_positions(self) -> None:
        records = fictional_challenge_records(random_state=42)
        self.assertEqual(len(records), 48)
        self.assertEqual({record["sampling_stratum"] for record in records}, set(STRATA))
        tasks = build_tasks_from_records(
            records,
            unique_count=48,
            blind_repeat_fraction=0,
            random_state=42,
        )
        record_by_requirement = {
            record["requirement_sentences"][0]: record
            for record in records
        }
        positions: dict[str, int] = {}
        for task in tasks:
            intended = record_by_requirement[task["requirement"]]["evidence_sentences"][0]
            index = next(
                index
                for index, candidate in enumerate(task["candidates"])
                if candidate["evidence"] == intended
            )
            key = chr(65 + index)
            positions[key] = positions.get(key, 0) + 1
        self.assertEqual(set(positions), {"A", "B", "C", "D"})
        self.assertEqual(positions, {"A": 12, "B": 12, "C": 12, "D": 12})

    def test_blind_repeats_reshuffle_and_audit_detects_position_bias(self) -> None:
        tasks = build_tasks_from_records(
            fictional_challenge_records(random_state=9),
            unique_count=48,
            blind_repeat_fraction=0.1,
            random_state=9,
        )
        repeat = next(task for task in tasks if task["blind_duplicate_of"])
        original = next(task for task in tasks if task["task_id"] == repeat["blind_duplicate_of"])
        self.assertNotEqual(
            [candidate["candidate_id"] for candidate in repeat["candidates"]],
            [candidate["candidate_id"] for candidate in original["candidates"]],
        )
        states = {
            task["task_id"]: {
                "action": "label",
                "support_label": "Direct",
                "selected_candidate_id": task["candidates"][0]["candidate_id"],
            }
            for task in tasks
            if not task["blind_duplicate_of"]
        }
        report = audit_annotations(tasks, states)
        self.assertGreater(report["position_max_share"], 0.95)
        self.assertIn("One answer position exceeds 45% of selected evidence.", report["warnings"])
        self.assertEqual(report["forbidden_queue_fields"], [])
        self.assertEqual(report["conflicting_repeat_pairs"], 0)


if __name__ == "__main__":
    unittest.main()
