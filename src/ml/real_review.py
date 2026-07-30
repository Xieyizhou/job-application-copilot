"""Prepare independently shuffled reviewer queues for real-text tasks."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import math
import random
from typing import Any, Iterable, Mapping

from ml.annotation import latest_task_states, validate_queue
from ml.real_holdout import jsonl_bytes, sha256_bytes


def _content_signature(tasks: Iterable[dict[str, Any]]) -> str:
    canonical: list[dict[str, Any]] = []
    for task in tasks:
        candidates: list[dict[str, str]] = [
            {
                "candidate_id": str(candidate["candidate_id"]),
                "evidence": str(candidate["evidence"]),
            }
            for candidate in task["candidates"]
        ]
        candidates.sort(key=lambda candidate: candidate["candidate_id"])
        canonical.append(
            {
                "task_id": str(task["task_id"]),
                "requirement": str(task["requirement"]),
                "source_job_hash": str(task.get("source_job_hash", "")),
                "source_resume_hash": str(task.get("source_resume_hash", "")),
                "candidates": candidates,
            }
        )
    canonical.sort(key=lambda task: str(task["task_id"]))
    return sha256_bytes(jsonl_bytes(canonical))


def shuffled_reviewer_queue(
    tasks: Iterable[dict[str, Any]],
    *,
    reviewer_id: str,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    """Return a content-identical queue with deterministic independent ordering."""
    checked = validate_queue(tasks)
    queue: list[dict[str, Any]] = []
    for task in checked:
        copy = deepcopy(task)
        seed_material = f"{random_state}:{reviewer_id}:{task['task_id']}"
        seed = int(hashlib.sha256(seed_material.encode()).hexdigest()[:16], 16)
        random.Random(seed).shuffle(copy["candidates"])
        queue.append(copy)
    random.Random(f"{random_state}:{reviewer_id}").shuffle(queue)
    return validate_queue(queue)


def reviewer_packet_manifest(
    source_tasks: Iterable[dict[str, Any]],
    reviewer_queues: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    """Describe packet identity without exposing labels or model predictions."""
    source = validate_queue(source_tasks)
    expected_signature = _content_signature(source)
    queue_details: dict[str, dict[str, Any]] = {}
    for reviewer_id, queue in sorted(reviewer_queues.items()):
        checked = validate_queue(queue)
        signature = _content_signature(checked)
        if signature != expected_signature:
            raise ValueError(f"Reviewer {reviewer_id} queue content changed.")
        queue_details[reviewer_id] = {
            "tasks": len(checked),
            "content_sha256": signature,
            "ordered_packet_sha256": sha256_bytes(jsonl_bytes(checked)),
        }
    return {
        "schema_version": 1,
        "tasks": len(source),
        "reviewers": queue_details,
        "content_identity_verified": True,
        "model_predictions_included": False,
        "labels_included": False,
    }


def build_adjudication_queue(
    tasks: Iterable[dict[str, Any]],
    reviewer_a_events: Iterable[dict[str, Any]],
    reviewer_b_events: Iterable[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return content-only tasks whose latest reviewer decisions disagree."""
    checked = validate_queue(tasks)
    task_ids = {str(task["task_id"]) for task in checked}
    a_states = latest_task_states(reviewer_a_events)
    b_states = latest_task_states(reviewer_b_events)
    if set(a_states) != task_ids or set(b_states) != task_ids:
        raise ValueError("Both reviewers must complete every task before comparison.")

    def decision(state: dict[str, Any]) -> tuple[object, ...]:
        candidate_labels = state.get("candidate_labels")
        complete_labels = (
            tuple(
                sorted(
                    (str(candidate_id), str(label))
                    for candidate_id, label in candidate_labels.items()
                )
            )
            if isinstance(candidate_labels, Mapping) and candidate_labels
            else ()
        )
        return (
            state.get("support_label"),
            state.get("selected_candidate_id"),
            complete_labels,
        )

    disagreements = [
        task
        for task in checked
        if decision(a_states[str(task["task_id"])])
        != decision(b_states[str(task["task_id"])])
    ]
    label_agreements = sum(
        a_states[task_id].get("support_label")
        == b_states[task_id].get("support_label")
        for task_id in task_ids
    )
    return disagreements, {
        "schema_version": 1,
        "tasks": len(checked),
        "exact_agreements": len(checked) - len(disagreements),
        "exact_agreement_rate": (len(checked) - len(disagreements)) / len(checked),
        "label_agreements": label_agreements,
        "label_agreement_rate": label_agreements / len(checked),
        "adjudication_tasks": len(disagreements),
        "reviewer_b_role": "model_assisted_independent_review",
        "status": (
            "ready_for_gold"
            if not disagreements
            else "human_adjudication_required"
        ),
    }


def build_adjudication_queue_with_audit(
    tasks: Iterable[dict[str, Any]],
    reviewer_a_events: Iterable[dict[str, Any]],
    reviewer_b_events: Iterable[dict[str, Any]],
    *,
    agreement_audit_fraction: float,
    random_state: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Add a deterministic random audit sample of exact agreements."""
    if not 0.0 <= agreement_audit_fraction <= 1.0:
        raise ValueError("agreement audit fraction must be between zero and one")
    checked = validate_queue(tasks)
    disagreements, report = build_adjudication_queue(
        checked,
        reviewer_a_events,
        reviewer_b_events,
    )
    disagreement_ids = {
        str(task["task_id"]) for task in disagreements
    }
    agreements = [
        task
        for task in checked
        if str(task["task_id"]) not in disagreement_ids
    ]
    audit_count = (
        min(
            len(agreements),
            max(1, math.ceil(len(agreements) * agreement_audit_fraction)),
        )
        if agreements and agreement_audit_fraction > 0.0
        else 0
    )
    ranked = sorted(
        agreements,
        key=lambda task: hashlib.sha256(
            f"{random_state}:{task['task_id']}".encode()
        ).hexdigest(),
    )
    audit = ranked[:audit_count]
    audit_ids = [str(task["task_id"]) for task in audit]
    review_ids = disagreement_ids | set(audit_ids)
    review_queue = [
        task for task in checked if str(task["task_id"]) in review_ids
    ]
    report.update(
        {
            "agreement_audit_fraction": agreement_audit_fraction,
            "agreement_audit_tasks": audit_count,
            "agreement_audit_task_ids": sorted(audit_ids),
            "human_review_tasks": len(review_queue),
            "audit_random_state": random_state,
            "status": (
                "human_adjudication_and_audit_required"
                if review_queue
                else "ready_for_gold"
            ),
        }
    )
    return review_queue, report
