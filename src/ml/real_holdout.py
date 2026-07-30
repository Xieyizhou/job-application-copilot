"""Finalize independently reviewed real-text tasks into a frozen local holdout."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import hashlib
import json
from typing import Any, Mapping

from ml.annotation import latest_task_states, validate_queue
from ml.candidate_judgments import (
    CandidateJudgmentError,
    normalized_candidate_labels,
    validate_complete_candidate_labels,
)


FINAL_LABELS = {"Direct", "Partial", "No Support"}


class RealHoldoutError(ValueError):
    """Raised when the real holdout cannot be finalized safely."""


def _task_content(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task.get("task_id", "")),
        "requirement": str(task.get("requirement", "")),
        "role_family": str(task.get("role_family", "")),
        "source_dataset": str(task.get("source_dataset", "")),
        "source_job_hash": str(task.get("source_job_hash", "")),
        "source_resume_hash": str(task.get("source_resume_hash", "")),
        "candidates": sorted(
            (
                {
                    "candidate_id": str(candidate.get("candidate_id", "")),
                    "evidence": str(candidate.get("evidence", "")),
                }
                for candidate in task.get("candidates", [])
            ),
            key=lambda candidate: candidate["candidate_id"],
        ),
    }


def _decision(
    state: dict[str, Any],
) -> tuple[str, str | None, tuple[tuple[str, str], ...]]:
    label = str(state.get("support_label", ""))
    selected = state.get("selected_candidate_id")
    candidate_labels = state.get("candidate_labels")
    complete_labels = (
        tuple(
            sorted(
                (str(candidate_id), str(candidate_label))
                for candidate_id, candidate_label in candidate_labels.items()
            )
        )
        if isinstance(candidate_labels, Mapping) and candidate_labels
        else ()
    )
    return label, str(selected) if selected is not None else None, complete_labels


def _validated_states(
    tasks: list[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    *,
    reviewer: str,
) -> dict[str, dict[str, Any]]:
    states = latest_task_states(events)
    task_ids = {str(task["task_id"]) for task in tasks}
    if set(states) != task_ids:
        missing = sorted(task_ids - set(states))
        extra = sorted(set(states) - task_ids)
        raise RealHoldoutError(
            f"{reviewer} decisions do not exactly cover the queue; "
            f"missing={missing}, extra={extra}."
        )
    for task in tasks:
        task_id = str(task["task_id"])
        state = states[task_id]
        if state.get("action") != "label":
            raise RealHoldoutError(f"{reviewer} task {task_id} is not labeled.")
        label, selected, _ = _decision(state)
        candidate_ids = {
            str(candidate["candidate_id"]) for candidate in task["candidates"]
        }
        if label not in FINAL_LABELS:
            raise RealHoldoutError(
                f"{reviewer} task {task_id} has non-final label {label or 'missing'}."
            )
        if label == "No Support" and selected is not None:
            raise RealHoldoutError(
                f"{reviewer} task {task_id} selects evidence for No Support."
            )
        if label != "No Support" and selected not in candidate_ids:
            raise RealHoldoutError(
                f"{reviewer} task {task_id} lacks valid selected evidence."
            )
        candidate_labels = normalized_candidate_labels(
            state.get("candidate_labels")
        )
        if candidate_labels:
            try:
                derived_label, _ = validate_complete_candidate_labels(
                    task,
                    candidate_labels,
                    selected,
                )
            except CandidateJudgmentError as error:
                raise RealHoldoutError(
                    f"{reviewer} task {task_id} has invalid candidate labels: {error}"
                ) from error
            if derived_label != label:
                raise RealHoldoutError(
                    f"{reviewer} task {task_id} candidate labels derive "
                    f"{derived_label}, not {label}."
                )
    return states


def finalize_real_reviews(
    reviewer_a_tasks: Iterable[dict[str, Any]],
    reviewer_b_tasks: Iterable[dict[str, Any]],
    reviewer_a_events: Iterable[dict[str, Any]],
    reviewer_b_events: Iterable[dict[str, Any]],
    adjudication_tasks: Iterable[dict[str, Any]],
    adjudication_events: Iterable[dict[str, Any]],
    *,
    dataset_role: str,
    agreement_audit_task_ids: Iterable[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge exact agreements and completed human adjudications."""
    if dataset_role not in {
        "holdout",
        "validation",
        "reserve",
        "development",
        "training_supplement",
    }:
        raise RealHoldoutError(f"Unsupported real-data role: {dataset_role}.")
    a_tasks = validate_queue(reviewer_a_tasks)
    b_tasks = validate_queue(reviewer_b_tasks)
    if {str(task["task_id"]) for task in a_tasks} != {
        str(task["task_id"]) for task in b_tasks
    }:
        raise RealHoldoutError("Reviewer queues do not contain identical task ids.")
    b_by_id = {str(task["task_id"]): task for task in b_tasks}
    for task in a_tasks:
        task_id = str(task["task_id"])
        if _task_content(task) != _task_content(b_by_id[task_id]):
            raise RealHoldoutError(
                f"Reviewer queue content differs for task {task_id}."
            )

    a_states = _validated_states(a_tasks, reviewer_a_events, reviewer="Reviewer A")
    b_states = _validated_states(b_tasks, reviewer_b_events, reviewer="Reviewer B")
    disagreements = {
        task_id
        for task_id in a_states
        if _decision(a_states[task_id]) != _decision(b_states[task_id])
    }
    agreement_audit_ids = {str(task_id) for task_id in agreement_audit_task_ids}
    invalid_audits = agreement_audit_ids & disagreements
    unknown_audits = agreement_audit_ids - set(a_states)
    if invalid_audits or unknown_audits:
        raise RealHoldoutError(
            "Agreement audits must contain only known exact agreements."
        )
    adjudication = validate_queue(adjudication_tasks)
    human_review_ids = disagreements | agreement_audit_ids
    if {str(task["task_id"]) for task in adjudication} != human_review_ids:
        raise RealHoldoutError(
            "Adjudication queue must contain exactly the A/B disagreements "
            "and precommitted agreement audits."
        )
    a_by_id = {str(task["task_id"]): task for task in a_tasks}
    for task in adjudication:
        task_id = str(task["task_id"])
        if _task_content(task) != _task_content(a_by_id[task_id]):
            raise RealHoldoutError(
                f"Adjudication queue content differs for task {task_id}."
            )
    adjudicated_states = _validated_states(
        adjudication,
        adjudication_events,
        reviewer="Human adjudicator",
    )

    gold_tasks: list[dict[str, Any]] = []
    for task in sorted(a_tasks, key=lambda item: str(item["task_id"])):
        task_id = str(task["task_id"])
        if task_id in human_review_ids:
            state = adjudicated_states[task_id]
            decision_source = (
                "human_adjudication"
                if task_id in disagreements
                else "human_agreement_audit"
            )
        else:
            state = a_states[task_id]
            decision_source = "human_model_agreement"
        label, selected, _ = _decision(state)
        content = _task_content(task)
        candidate_labels = normalized_candidate_labels(
            state.get("candidate_labels")
        )
        if candidate_labels:
            content["candidates"] = [
                {
                    **candidate,
                    "support_label": candidate_labels[
                        str(candidate["candidate_id"])
                    ],
                }
                for candidate in content["candidates"]
            ]
        gold_tasks.append(
            {
                "record_type": f"real_{dataset_role}_gold_task",
                "schema_version": 1,
                **content,
                "support_label": label,
                "selected_candidate_id": selected,
                "decision_source": decision_source,
            }
        )
    report = {
        "schema_version": 1,
        "tasks": len(gold_tasks),
        "exact_agreements": len(gold_tasks) - len(disagreements),
        "human_adjudications": len(disagreements),
        "human_agreement_audits": len(agreement_audit_ids),
        "label_counts": dict(
            Counter(str(task["support_label"]) for task in gold_tasks)
        ),
        "decision_source_counts": dict(
            Counter(str(task["decision_source"]) for task in gold_tasks)
        ),
        "dataset_role": dataset_role,
        "status": f"frozen_local_real_{dataset_role}",
        "training_use": (
            "prohibited"
            if dataset_role == "holdout"
            else (
                "threshold_and_model_development_only"
                if dataset_role in {"validation", "development"}
                else (
                    "reviewed_training_allowed"
                    if dataset_role == "training_supplement"
                    else "one_time_evaluation_only"
                )
            )
        ),
    }
    return gold_tasks, report


def finalize_real_holdout(
    reviewer_a_tasks: Iterable[dict[str, Any]],
    reviewer_b_tasks: Iterable[dict[str, Any]],
    reviewer_a_events: Iterable[dict[str, Any]],
    reviewer_b_events: Iterable[dict[str, Any]],
    adjudication_tasks: Iterable[dict[str, Any]],
    adjudication_events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Merge reviews into the protected one-time real holdout."""
    return finalize_real_reviews(
        reviewer_a_tasks,
        reviewer_b_tasks,
        reviewer_a_events,
        reviewer_b_events,
        adjudication_tasks,
        adjudication_events,
        dataset_role="holdout",
    )


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    """Serialize deterministic JSONL suitable for checksum freezing."""
    return "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    """Return a hexadecimal SHA-256 checksum."""
    return hashlib.sha256(payload).hexdigest()
