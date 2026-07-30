"""Diagnostic evaluation of completed blind shadow disagreement reviews."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any


SUPPORT_LABELS = {"Direct", "Partial"}


def _method_accepted(comparison: str, method: str) -> bool:
    if comparison == "both_accept_different_evidence":
        return True
    return comparison == f"{method}_only_accept"


def _method_outcome(
    task: dict[str, Any],
    state: dict[str, Any],
    construction: dict[str, Any],
    *,
    method: str,
) -> dict[str, Any]:
    candidate_labels = {
        str(candidate_id): str(label)
        for candidate_id, label in state["candidate_labels"].items()
    }
    origins = {
        str(candidate_id): str(origin)
        for candidate_id, origin in construction["candidate_origins"].items()
    }
    accepted = _method_accepted(str(construction["comparison"]), method)
    origin_ids = [
        candidate_id
        for candidate_id, origin in origins.items()
        if origin == method
    ]
    if accepted and len(origin_ids) != 1:
        raise ValueError(f"Accepted {method} evidence origin is missing.")
    accepted_label = candidate_labels[origin_ids[0]] if origin_ids else None
    observed_supported = any(
        label in SUPPORT_LABELS for label in candidate_labels.values()
    )
    correct = (
        accepted_label in SUPPORT_LABELS
        if accepted
        else not observed_supported
    )
    if accepted and accepted_label == "No Support":
        error = "false_accept"
    elif not accepted and observed_supported:
        error = "observed_false_reject"
    elif accepted and state.get("selected_candidate_id") not in origin_ids:
        error = "accepted_but_not_strongest"
    else:
        error = "none"
    return {
        "task_id": str(task["task_id"]),
        "role_family": str(task["role_family"]),
        "accepted": accepted,
        "accepted_label": accepted_label,
        "observed_supported": observed_supported,
        "correct": bool(correct),
        "error": error,
        "selected_strongest": bool(
            accepted
            and state.get("selected_candidate_id") in origin_ids
        ),
    }


def _aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    correct = sum(bool(item["correct"]) for item in outcomes)
    accepted = [item for item in outcomes if item["accepted"]]
    return {
        "tasks": len(outcomes),
        "diagnostic_accuracy": correct / len(outcomes) if outcomes else 0.0,
        "correct_decisions": correct,
        "accepted_tasks": len(accepted),
        "rejected_tasks": len(outcomes) - len(accepted),
        "accepted_label_counts": dict(
            Counter(str(item["accepted_label"]) for item in accepted)
        ),
        "error_counts": dict(
            Counter(
                str(item["error"])
                for item in outcomes
                if item["error"] != "none"
            )
        ),
        "selected_strongest": sum(
            bool(item["selected_strongest"]) for item in accepted
        ),
    }


def evaluate_shadow_review(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    construction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reveal method origins and return balanced diagnostic metrics."""
    construction = {
        str(row["task_id"]): row for row in construction_rows
    }
    task_ids = {str(task["task_id"]) for task in tasks}
    if set(states) != task_ids or set(construction) != task_ids:
        raise ValueError("Review states and construction must cover every task.")
    outcomes: dict[str, list[dict[str, Any]]] = {
        method: [
            _method_outcome(
                task,
                states[str(task["task_id"])],
                construction[str(task["task_id"])],
                method=method,
            )
            for task in tasks
        ]
        for method in ("baseline", "shadow")
    }
    per_role: defaultdict[str, dict[str, Any]] = defaultdict(dict)
    for role in sorted({str(task["role_family"]) for task in tasks}):
        for method in outcomes:
            per_role[role][method] = _aggregate(
                [
                    item
                    for item in outcomes[method]
                    if item["role_family"] == role
                ]
            )
    method_metrics = {
        method: _aggregate(method_outcomes)
        for method, method_outcomes in outcomes.items()
    }
    shadow_errors = method_metrics["shadow"]["error_counts"]
    baseline_errors = method_metrics["baseline"]["error_counts"]
    safety_stop = (
        int(shadow_errors.get("false_accept", 0))
        > int(baseline_errors.get("false_accept", 0))
        or float(method_metrics["shadow"]["diagnostic_accuracy"])
        <= float(method_metrics["baseline"]["diagnostic_accuracy"])
    )
    return {
        "schema_version": 1,
        "tasks": len(tasks),
        "task_label_counts": dict(
            Counter(
                str(states[str(task["task_id"])]["support_label"])
                for task in tasks
            )
        ),
        "methods": method_metrics,
        "per_role": dict(per_role),
        "diagnostic_boundary": (
            "Balanced disagreement sample only; metrics are not population "
            "accuracy and cannot be used to retune the reserve-tested model."
        ),
        "operational_safety_stop": {
            "triggered": safety_stop,
            "status": (
                "shadow_v1_rejected_for_product_integration"
                if safety_stop
                else "additional_independent_review_required"
            ),
            "reasons": [
                "Shadow false accepts exceed the baseline on the reviewed sample.",
                "Shadow diagnostic accuracy does not exceed the baseline.",
            ]
            if safety_stop
            else [],
            "reviewed_sample_training_use": "prohibited",
        },
        "product_integration_allowed": False,
    }
