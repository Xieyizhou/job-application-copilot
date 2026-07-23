"""Build leakage-aware training rows from reviewed evidence annotations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ml.annotation import repeat_conflict_task_ids


DATASET_SCHEMA_VERSION = 1
POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}


class AnnotationDatasetError(ValueError):
    """Raised when reviewed annotations are not safe to export."""


def requirement_template_group(requirement: str) -> str:
    """Return a compact group key used to keep repeated wording in one fold."""
    words = str(requirement).lower().split()
    return " ".join(words[:3])


def _fold_map(groups: set[str], random_state: int) -> dict[str, int]:
    ordered = sorted(
        groups,
        key=lambda group: hashlib.sha256(
            f"{random_state}:{group}".encode("utf-8")
        ).hexdigest(),
    )
    return {group: index for index, group in enumerate(ordered)}


def build_annotated_tasks(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    random_state: int = 42,
    require_complete: bool = True,
) -> list[dict[str, Any]]:
    """Return one reviewed row per unique task and reject unresolved labels."""
    if repeat_conflict_task_ids(tasks, states):
        raise AnnotationDatasetError("Blind-repeat conflicts must be resolved before export.")
    unique_tasks = [task for task in tasks if not task.get("blind_duplicate_of")]
    groups = {
        requirement_template_group(str(task.get("requirement", "")))
        for task in unique_tasks
    }
    folds = _fold_map(groups, random_state)
    rows: list[dict[str, Any]] = []
    for task in unique_tasks:
        task_id = str(task["task_id"])
        state = states.get(task_id)
        if not state or state.get("action") != "label":
            if require_complete:
                raise AnnotationDatasetError(f"Task {task_id} is not labeled.")
            continue
        support_label = str(state.get("support_label", ""))
        if support_label not in {*POSITIVE_SUPPORT_LABELS, "No Support"}:
            raise AnnotationDatasetError(
                f"Task {task_id} has unresolved label {support_label or 'missing'}."
            )
        selected_id = state.get("selected_candidate_id")
        candidate_ids = {
            str(candidate["candidate_id"])
            for candidate in task["candidates"]
        }
        if support_label in POSITIVE_SUPPORT_LABELS and selected_id not in candidate_ids:
            raise AnnotationDatasetError(f"Task {task_id} needs selected supporting evidence.")
        if support_label == "No Support" and selected_id is not None:
            raise AnnotationDatasetError(f"Task {task_id} cannot select evidence for No Support.")
        template_group = requirement_template_group(str(task["requirement"]))
        rows.append(
            {
                "schema_version": DATASET_SCHEMA_VERSION,
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
                "selected_candidate_id": selected_id,
                "support_label": support_label,
                "cover_letter_safe": state.get("cover_letter_safe"),
                "source_dataset": str(task.get("source_dataset", "unknown")),
                "source_resume_hash": str(task.get("source_resume_hash", "")),
                "source_job_hash": str(task.get("source_job_hash", "")),
                "semantic_case_group_id": str(
                    task.get("semantic_case_group_id", "")
                ),
                "template_group": template_group,
                "fold": folds[template_group],
            }
        )
    return rows


def build_training_pairs(annotated_tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create high-confidence binary pairs without labeling ambiguous distractors."""
    pairs: list[dict[str, Any]] = []
    for task in annotated_tasks:
        selected_id = task.get("selected_candidate_id")
        label = str(task["support_label"])
        if label in POSITIVE_SUPPORT_LABELS:
            candidates = [
                candidate
                for candidate in task["candidates"]
                if candidate["candidate_id"] == selected_id
            ]
            pair_candidates = [(candidates[0], 1, "selected_best_evidence")]
        else:
            pair_candidates = [
                (candidate, 0, "no_support_task_candidate")
                for candidate in task["candidates"]
            ]
        for candidate, binary_label, label_scope in pair_candidates:
            identity = f"{task['task_id']}:{candidate['candidate_id']}:{binary_label}"
            pairs.append(
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "pair_id": hashlib.sha256(identity.encode("utf-8")).hexdigest()[:16],
                    "task_id": task["task_id"],
                    "role_family": task["role_family"],
                    "requirement": task["requirement"],
                    "evidence": candidate["evidence"],
                    "binary_label": binary_label,
                    "support_label": label,
                    "label_scope": label_scope,
                    "source_resume_hash": task.get("source_resume_hash", ""),
                    "source_job_hash": task.get("source_job_hash", ""),
                    "semantic_case_group_id": task.get("semantic_case_group_id", ""),
                    "template_group": task["template_group"],
                    "fold": task["fold"],
                }
            )
    return pairs


def dataset_manifest(
    annotated_tasks: list[dict[str, Any]],
    training_pairs: list[dict[str, Any]],
    *,
    dataset_name: str = "reviewed_evidence_pilot",
    source_queue_complete: bool = True,
) -> dict[str, Any]:
    """Return aggregate, non-sensitive dataset metadata."""
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_name": dataset_name,
        "source_queue_complete": source_queue_complete,
        "unique_tasks": len(annotated_tasks),
        "training_pairs": len(training_pairs),
        "template_groups": len({row["template_group"] for row in annotated_tasks}),
        "task_label_counts": dict(Counter(row["support_label"] for row in annotated_tasks)),
        "pair_label_counts": dict(Counter(str(row["binary_label"]) for row in training_pairs)),
        "role_counts": dict(Counter(row["role_family"] for row in annotated_tasks)),
        "split_protocol": (
            "Fictional pilot: leave-one-requirement-template-group-out evaluation. "
            "Real data: connected resume/job/semantic groups must be isolated before "
            "training; blind repeats excluded."
        ),
        "negative_policy": (
            "Only candidates from tasks labeled No Support are negative training pairs; "
            "unselected candidates from supported tasks remain unlabeled."
        ),
        "limitations": [
            "Pilot-sized fictional calibration data.",
            "Direct and Partial are preserved but not trained as separate classes.",
            "Independent real-resume and real-job validation is still required.",
        ],
    }


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> None:
    """Write deterministic JSONL output."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
