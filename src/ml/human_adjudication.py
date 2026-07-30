"""Convert completed local human adjudications into conservative gold tasks."""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

from ml.annotation import SUPPORT_LABELS, latest_task_states, validate_queue
from ml.annotation_generation import normalize_text
from ml.candidate_judgments import (
    CandidateJudgmentError,
    normalized_candidate_labels,
    validate_complete_candidate_labels,
)


POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}
UNLABELED_CANDIDATE_LABEL = "Unlabeled"


class HumanAdjudicationError(ValueError):
    """Raised when a human-adjudication export is incomplete or unsafe."""


def _case_by_id(cases: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("case_id", "")).strip()
        if not case_id:
            raise HumanAdjudicationError("Source cases require case_id values.")
        if case_id in indexed:
            raise HumanAdjudicationError(f"Duplicate source case id: {case_id}")
        indexed[case_id] = case
    return indexed


def _same_content(queue_task: dict[str, Any], source_case: dict[str, Any]) -> bool:
    if normalize_text(str(queue_task["requirement"])) != normalize_text(
        str(source_case.get("requirement", ""))
    ):
        return False
    queue_candidates = {
        str(candidate["candidate_id"]): normalize_text(str(candidate["evidence"]))
        for candidate in queue_task["candidates"]
    }
    source_candidates = {
        str(candidate.get("candidate_id", "")): normalize_text(
            str(candidate.get("evidence", ""))
        )
        for candidate in source_case.get("candidates", [])
        if isinstance(candidate, dict)
    }
    return queue_candidates == source_candidates


def _gold_id(task_id: str, event_id: str) -> str:
    material = f"{task_id}:human_adjudication:{event_id}"
    return "gold-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def build_human_adjudication_gold(
    queue: Iterable[dict[str, Any]],
    events: Iterable[dict[str, Any]],
    source_cases: Iterable[dict[str, Any]],
    *,
    require_complete: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Create reviewed gold while retaining only explicitly supported pair labels.

    Direct/Partial adjudications label only the selected strongest candidate. Remaining
    candidates are intentionally ``Unlabeled``. A No Support adjudication labels every
    candidate No Support because the reviewer rejected the complete candidate set.
    """
    tasks = validate_queue(queue)
    states = latest_task_states(events)
    source_by_id = _case_by_id(source_cases)
    gold_tasks: list[dict[str, Any]] = []
    incomplete_ids: list[str] = []
    unresolved_ids: list[str] = []
    skipped_ids: list[str] = []

    for task in tasks:
        task_id = str(task["task_id"])
        source_case = source_by_id.get(task_id)
        if source_case is None:
            raise HumanAdjudicationError(
                f"Adjudication task {task_id} does not match a source case."
            )
        if not _same_content(task, source_case):
            raise HumanAdjudicationError(
                f"Adjudication task {task_id} does not match its source case content."
            )
        state = states.get(task_id)
        if state is None:
            incomplete_ids.append(task_id)
            continue
        if state.get("action") == "skip":
            skipped_ids.append(task_id)
            continue
        if state.get("action") != "label":
            incomplete_ids.append(task_id)
            continue
        support_label = str(state.get("support_label", ""))
        if support_label == "Uncertain":
            unresolved_ids.append(task_id)
            continue
        if support_label not in SUPPORT_LABELS:
            raise HumanAdjudicationError(
                f"Adjudication task {task_id} has an invalid support label."
            )
        selected_id = state.get("selected_candidate_id")
        candidate_ids = {str(candidate["candidate_id"]) for candidate in task["candidates"]}
        if support_label in POSITIVE_SUPPORT_LABELS and selected_id not in candidate_ids:
            raise HumanAdjudicationError(
                f"Adjudication task {task_id} requires selected supporting evidence."
            )
        if support_label == "No Support" and selected_id is not None:
            raise HumanAdjudicationError(
                f"Adjudication task {task_id} cannot select evidence for No Support."
            )
        candidate_labels = normalized_candidate_labels(
            state.get("candidate_labels")
        )
        if candidate_labels:
            try:
                derived_label, candidate_labels = validate_complete_candidate_labels(
                    task,
                    candidate_labels,
                    str(selected_id) if selected_id is not None else None,
                )
            except CandidateJudgmentError as error:
                raise HumanAdjudicationError(
                    f"Adjudication task {task_id} has invalid candidate labels: {error}"
                ) from error
            if derived_label != support_label:
                raise HumanAdjudicationError(
                    f"Adjudication task {task_id} candidate labels derive "
                    f"{derived_label}, not {support_label}."
                )
        else:
            candidate_labels = {
                candidate_id: (
                    support_label
                    if support_label == "No Support" or candidate_id == selected_id
                    else UNLABELED_CANDIDATE_LABEL
                )
                for candidate_id in candidate_ids
            }
        event_id = str(state.get("event_id", "")).strip()
        if not event_id:
            raise HumanAdjudicationError(f"Adjudication task {task_id} has no event id.")
        gold_tasks.append(
            {
                "record_type": "gold_task",
                "schema_version": 1,
                "gold_id": _gold_id(task_id, event_id),
                "case_id": task_id,
                "role_family": str(task["role_family"]),
                "requirement": str(task["requirement"]),
                "candidates": [
                    {
                        "candidate_id": str(candidate["candidate_id"]),
                        "evidence": str(candidate["evidence"]),
                        "support_label": candidate_labels[str(candidate["candidate_id"])],
                    }
                    for candidate in task["candidates"]
                ],
                "best_candidate_id": selected_id,
                "support_label": support_label,
                "decision_source": "human_adjudication",
                "reviewer_count": 1,
                "semantic_case_group_id": str(
                    source_case.get("semantic_case_group_id", "")
                ),
                "source_kind": str(source_case.get("source_kind", "synthetic")),
                "cover_letter_safe": state.get("cover_letter_safe"),
            }
        )

    blocked_ids = sorted([*incomplete_ids, *unresolved_ids, *skipped_ids])
    if require_complete and blocked_ids:
        raise HumanAdjudicationError(
            "Every adjudication task must be resolved before export; remaining task ids: "
            + ", ".join(blocked_ids)
        )
    report = {
        "queue_tasks": len(tasks),
        "accepted_gold_tasks": len(gold_tasks),
        "incomplete_task_ids": sorted(incomplete_ids),
        "uncertain_task_ids": sorted(unresolved_ids),
        "skipped_task_ids": sorted(skipped_ids),
        "require_complete": require_complete,
        "label_policy": (
            "Complete candidate judgments are preserved when present. Legacy "
            "Direct/Partial events label only selected evidence and keep other "
            "candidates Unlabeled; legacy No Support events reject every candidate."
        ),
    }
    return gold_tasks, report
