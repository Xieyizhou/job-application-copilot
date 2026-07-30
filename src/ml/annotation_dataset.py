"""Build leakage-aware training rows from reviewed evidence annotations."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from ml.annotation import repeat_conflict_task_ids
from ml.candidate_judgments import (
    CANDIDATE_SUPPORT_LABELS,
    CandidateJudgmentError,
    normalized_candidate_labels,
    validate_complete_candidate_labels,
)


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
    require_candidate_labels: bool = False,
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
        candidate_labels = normalized_candidate_labels(
            state.get("candidate_labels")
        )
        if require_candidate_labels and not candidate_labels:
            raise AnnotationDatasetError(
                f"Task {task_id} still needs complete candidate labels."
            )
        if candidate_labels:
            try:
                derived_label, candidate_labels = validate_complete_candidate_labels(
                    task,
                    candidate_labels,
                    str(selected_id) if selected_id is not None else None,
                )
            except CandidateJudgmentError as error:
                raise AnnotationDatasetError(
                    f"Task {task_id} has invalid candidate labels: {error}"
                ) from error
            if derived_label != support_label:
                raise AnnotationDatasetError(
                    f"Task {task_id} candidate labels derive {derived_label}, "
                    f"not {support_label}."
                )
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
                        **(
                            {
                                "support_label": candidate_labels[
                                    str(candidate["candidate_id"])
                                ]
                            }
                            if candidate_labels
                            else {}
                        ),
                    }
                    for candidate in task["candidates"]
                ],
                "selected_candidate_id": selected_id,
                "support_label": support_label,
                "candidate_label_coverage": (
                    "complete" if candidate_labels else "legacy_selected_only"
                ),
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
    """Create candidate-level pairs, with a conservative legacy fallback."""
    pairs: list[dict[str, Any]] = []
    for task in annotated_tasks:
        selected_id = task.get("selected_candidate_id")
        label = str(task["support_label"])
        candidate_labels = {
            str(candidate["candidate_id"]): str(candidate.get("support_label", ""))
            for candidate in task["candidates"]
        }
        has_complete_candidate_labels = bool(candidate_labels) and all(
            candidate_label in CANDIDATE_SUPPORT_LABELS
            for candidate_label in candidate_labels.values()
        )
        if has_complete_candidate_labels:
            pair_candidates = [
                (
                    candidate,
                    int(candidate_labels[str(candidate["candidate_id"])] in POSITIVE_SUPPORT_LABELS),
                    "fully_reviewed_candidate_judgment",
                    candidate_labels[str(candidate["candidate_id"])],
                )
                for candidate in task["candidates"]
            ]
        elif label in POSITIVE_SUPPORT_LABELS:
            candidates = [
                candidate
                for candidate in task["candidates"]
                if candidate["candidate_id"] == selected_id
            ]
            pair_candidates = [
                (candidates[0], 1, "selected_best_evidence", label)
            ]
        else:
            pair_candidates = [
                (candidate, 0, "no_support_task_candidate", "No Support")
                for candidate in task["candidates"]
            ]
        for candidate, binary_label, label_scope, candidate_label in pair_candidates:
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
                    "support_label": candidate_label,
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
        "pair_support_label_counts": dict(
            Counter(str(row["support_label"]) for row in training_pairs)
        ),
        "candidate_label_coverage_counts": dict(
            Counter(
                str(row.get("candidate_label_coverage", "legacy_selected_only"))
                for row in annotated_tasks
            )
        ),
        "role_counts": dict(Counter(row["role_family"] for row in annotated_tasks)),
        "split_protocol": (
            "Fictional pilot: leave-one-requirement-template-group-out evaluation. "
            "Real data: connected resume/job/semantic groups must be isolated before "
            "training; blind repeats excluded."
        ),
        "negative_policy": (
            "Complete candidate-level reviews contribute every Direct, Partial, and "
            "No Support judgment. Legacy supported tasks retain only selected positive "
            "evidence; legacy No Support tasks contribute all candidates as negatives."
        ),
        "limitations": [
            "Pilot-sized fictional calibration data.",
            "Legacy events do not invent labels for unselected candidates.",
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
