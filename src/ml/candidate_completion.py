"""Build balanced training-only queues for missing candidate judgments."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
from typing import Any, Iterable

from document_text import normalize_comparison_text as normalize_text
from ml.candidate_judgments import CANDIDATE_SUPPORT_LABELS


def tasks_requiring_candidate_completion(
    tasks: Iterable[dict[str, Any]],
    pairs: Iterable[dict[str, Any]],
    *,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Return incomplete tasks in deterministic role/label-balanced order."""
    labels_by_pair = {
        (
            str(pair["task_id"]),
            normalize_text(str(pair["evidence"])),
        ): str(pair.get("support_label", ""))
        for pair in pairs
    }
    groups: defaultdict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        task_id = str(task["task_id"])
        complete = all(
            labels_by_pair.get(
                (task_id, normalize_text(str(candidate["evidence"]))),
                "",
            )
            in CANDIDATE_SUPPORT_LABELS
            for candidate in task["candidates"]
        )
        if complete:
            continue
        queue_task = {
            "schema_version": 1,
            "task_id": task_id,
            "role_family": str(task["role_family"]),
            "requirement": str(task["requirement"]),
            "candidates": [
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "evidence": str(candidate["evidence"]),
                }
                for candidate in task["candidates"]
            ],
            "source_dataset": str(task.get("source_dataset", "reviewed_training")),
            "source_resume_hash": str(task.get("source_resume_hash", "")),
            "source_job_hash": str(task.get("source_job_hash", "")),
            "completion_source": "existing_reviewed_training_task",
        }
        groups[
            (
                str(task.get("role_family", "unknown")),
                str(task.get("support_label", "unknown")),
            )
        ].append(
            {
                "queue_task": queue_task,
                "support_label": str(task["support_label"]),
                "selected_candidate_id": task.get("selected_candidate_id"),
                "cover_letter_safe": task.get("cover_letter_safe"),
            }
        )
    for key, rows in groups.items():
        rows.sort(
            key=lambda row: hashlib.sha256(
                f"{random_state}:{key}:{row['queue_task']['task_id']}".encode()
            ).hexdigest()
        )

    ordered: list[dict[str, Any]] = []
    keys = sorted(groups)
    while any(groups.values()):
        for key in keys:
            if groups[key]:
                ordered.append(groups[key].pop(0))
    return ordered


def merge_completed_candidate_labels(
    base_tasks: Iterable[dict[str, Any]],
    base_pairs: Iterable[dict[str, Any]],
    completed_tasks: Iterable[dict[str, Any]],
    completed_pairs: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Replace legacy task pairs with fully reviewed candidate judgments."""
    base_task_list = [dict(task) for task in base_tasks]
    base_pair_list = [dict(pair) for pair in base_pairs]
    completed_task_list = [dict(task) for task in completed_tasks]
    completed_pair_list = [dict(pair) for pair in completed_pairs]
    base_by_id = {str(task["task_id"]): task for task in base_task_list}
    completed_ids = {str(task["task_id"]) for task in completed_task_list}
    if len(base_by_id) != len(base_task_list):
        raise ValueError("Base training task ids must be unique.")
    if len(completed_ids) != len(completed_task_list):
        raise ValueError("Completed task ids must be unique.")
    if not completed_ids <= set(base_by_id):
        raise ValueError("Completion data references tasks outside the base training set.")

    def content(task: dict[str, Any]) -> tuple[str, tuple[tuple[str, str], ...]]:
        return (
            normalize_text(str(task["requirement"])),
            tuple(
                sorted(
                    (
                        str(candidate["candidate_id"]),
                        normalize_text(str(candidate["evidence"])),
                    )
                    for candidate in task["candidates"]
                )
            ),
        )

    replacements: dict[str, dict[str, Any]] = {}
    for completed in completed_task_list:
        task_id = str(completed["task_id"])
        base = base_by_id[task_id]
        if content(base) != content(completed):
            raise ValueError(f"Completion content changed for task {task_id}.")
        if not all(
            str(candidate.get("support_label", ""))
            in CANDIDATE_SUPPORT_LABELS
            for candidate in completed["candidates"]
        ):
            raise ValueError(f"Completion task {task_id} lacks candidate labels.")
        replacements[task_id] = {
            **base,
            **completed,
            "evaluation_group": base.get(
                "evaluation_group",
                completed.get("evaluation_group", ""),
            ),
            "review_source": "human_candidate_completion",
            "candidate_label_coverage": "complete",
        }
    merged_tasks = [
        replacements.get(str(task["task_id"]), task)
        for task in base_task_list
    ]
    merged_pairs = [
        pair
        for pair in base_pair_list
        if str(pair["task_id"]) not in completed_ids
    ]
    for pair in completed_pair_list:
        task_id = str(pair["task_id"])
        if task_id not in completed_ids:
            raise ValueError(f"Completion pair references unexpected task {task_id}.")
        base_task = base_by_id[task_id]
        merged_pairs.append(
            {
                **pair,
                "evaluation_group": base_task.get(
                    "evaluation_group",
                    pair.get("evaluation_group", ""),
                ),
                "review_source": "human_candidate_completion",
            }
        )
    pair_ids = [str(pair["pair_id"]) for pair in merged_pairs]
    if len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Merged training pair ids must be unique.")
    manifest = {
        "schema_version": 1,
        "dataset_name": "reviewed_evidence_training_candidate_complete",
        "task_count": len(merged_tasks),
        "pair_count": len(merged_pairs),
        "completed_task_count": len(completed_ids),
        "task_label_counts": dict(
            Counter(str(task["support_label"]) for task in merged_tasks)
        ),
        "pair_support_label_counts": dict(
            Counter(str(pair["support_label"]) for pair in merged_pairs)
        ),
        "pair_binary_label_counts": dict(
            Counter(str(pair["binary_label"]) for pair in merged_pairs)
        ),
        "candidate_label_coverage_counts": dict(
            Counter(
                str(task.get("candidate_label_coverage", "legacy_selected_only"))
                for task in merged_tasks
            )
        ),
        "training_use": "candidate_model_training_and_grouped_development_only",
        "evaluation_use": "prohibited",
    }
    return merged_tasks, merged_pairs, manifest
