"""Independent model-review and human-authority gold helpers."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

from ml.candidate_judgments import (
    CANDIDATE_SUPPORT_LABELS,
    validate_complete_candidate_labels,
)


_LABEL_STRENGTH = {"No Support": 0, "Partial": 1, "Direct": 2}


def model_candidate_decision(
    task: Mapping[str, Any],
    labels: Sequence[str],
    support_scores: Sequence[float],
) -> dict[str, Any]:
    """Convert independent candidate predictions into one valid review."""
    candidates = list(task["candidates"])
    if len(labels) != len(candidates) or len(support_scores) != len(candidates):
        raise ValueError("Model predictions must align with every candidate.")
    if any(label not in CANDIDATE_SUPPORT_LABELS for label in labels):
        raise ValueError("Model review contains an unsupported candidate label.")
    candidate_labels = {
        str(candidate["candidate_id"]): str(label)
        for candidate, label in zip(candidates, labels, strict=True)
    }
    strongest = max(_LABEL_STRENGTH[label] for label in labels)
    selected_candidate_id: str | None = None
    if strongest:
        eligible = [
            index
            for index, label in enumerate(labels)
            if _LABEL_STRENGTH[label] == strongest
        ]
        selected_index = max(
            eligible,
            key=lambda index: (float(support_scores[index]), -index),
        )
        selected_candidate_id = str(
            candidates[selected_index]["candidate_id"]
        )
    support_label, checked_labels = validate_complete_candidate_labels(
        task,
        candidate_labels,
        selected_candidate_id,
    )
    return {
        "selected_candidate_id": selected_candidate_id,
        "support_label": support_label,
        "candidate_labels": checked_labels,
        "cover_letter_safe": False,
    }


def operational_review_agreement(
    tasks: Sequence[dict[str, Any]],
    reviewer_a: Mapping[str, Mapping[str, Any]],
    reviewer_b: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Compare full candidate decisions while retaining human authority."""
    expected = {str(task["task_id"]) for task in tasks}
    if set(reviewer_a) != expected or set(reviewer_b) != expected:
        raise ValueError("Both reviewers must cover every operational task.")
    task_exact = 0
    task_label = 0
    candidate_exact = 0
    candidate_total = 0
    disagreement_ids: list[str] = []
    for task in tasks:
        task_id = str(task["task_id"])
        left = reviewer_a[task_id]
        right = reviewer_b[task_id]
        left_labels = {
            str(key): str(value)
            for key, value in dict(left.get("candidate_labels", {})).items()
        }
        right_labels = {
            str(key): str(value)
            for key, value in dict(right.get("candidate_labels", {})).items()
        }
        if str(left.get("support_label")) == str(right.get("support_label")):
            task_label += 1
        exact = (
            left.get("support_label") == right.get("support_label")
            and left.get("selected_candidate_id")
            == right.get("selected_candidate_id")
            and left_labels == right_labels
        )
        task_exact += int(exact)
        if not exact:
            disagreement_ids.append(task_id)
        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in task["candidates"]
        ]
        candidate_total += len(candidate_ids)
        candidate_exact += sum(
            left_labels.get(candidate_id) == right_labels.get(candidate_id)
            for candidate_id in candidate_ids
        )
    return {
        "tasks": len(tasks),
        "task_exact_agreements": task_exact,
        "task_exact_agreement_rate": task_exact / len(tasks),
        "task_label_agreements": task_label,
        "task_label_agreement_rate": task_label / len(tasks),
        "candidate_judgments": candidate_total,
        "candidate_label_agreements": candidate_exact,
        "candidate_label_agreement_rate": candidate_exact / candidate_total,
        "disagreement_tasks": len(disagreement_ids),
        "disagreement_task_ids": disagreement_ids,
        "reviewer_a_label_counts": dict(
            Counter(
                str(reviewer_a[str(task["task_id"])]["support_label"])
                for task in tasks
            )
        ),
        "reviewer_b_label_counts": dict(
            Counter(
                str(reviewer_b[str(task["task_id"])]["support_label"])
                for task in tasks
            )
        ),
        "gold_authority": "reviewer_a_human",
    }
