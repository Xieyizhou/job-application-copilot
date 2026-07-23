"""Aggregate bias checks for a completed local annotation queue."""

from __future__ import annotations

from collections import Counter
from typing import Any

from ml.annotation import annotation_summary, repeat_conflict_task_ids


FORBIDDEN_TASK_FIELDS = {"weak_source_label"}
FORBIDDEN_CANDIDATE_FIELDS = {"retrieval_rank", "retrieval_similarity"}


def _prefix(text: str, word_count: int = 4) -> str:
    return " ".join(text.lower().split()[:word_count])


def audit_annotations(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return aggregate position, label, phrase, and leakage diagnostics."""
    summary = annotation_summary(tasks, states)
    conflict_pair_count = len(repeat_conflict_task_ids(tasks, states)) // 2
    unique_tasks = [task for task in tasks if not task.get("blind_duplicate_of")]
    role_counts = Counter(str(task.get("role_family", "Unknown")) for task in unique_tasks)
    source_counts = Counter(str(task.get("source_dataset", "Unknown")) for task in unique_tasks)
    prefixes = Counter(_prefix(str(task.get("requirement", ""))) for task in unique_tasks)
    repeated_prefix_share = max(prefixes.values(), default=0) / max(1, len(unique_tasks))
    position_counts = {
        key: int(value)
        for key, value in summary["selected_position_counts"].items()
        if key != "None"
    }
    selected_total = sum(position_counts.values())
    position_max_share = max(position_counts.values(), default=0) / max(1, selected_total)
    label_counts = {key: int(value) for key, value in summary["label_counts"].items()}
    label_total = sum(label_counts.values())
    label_max_share = max(label_counts.values(), default=0) / max(1, label_total)
    leaked_field_set = {
        f"task.{field}"
        for task in tasks
        for field in FORBIDDEN_TASK_FIELDS
        if field in task
    }
    leaked_field_set.update(
        f"candidate.{field}"
        for task in tasks
        for candidate in task.get("candidates", [])
        for field in FORBIDDEN_CANDIDATE_FIELDS
        if field in candidate
    )
    leaked_fields = sorted(leaked_field_set)
    warnings: list[str] = []
    if selected_total >= 20 and position_max_share > 0.45:
        warnings.append("One answer position exceeds 45% of selected evidence.")
    if label_total >= 20 and label_max_share > 0.7:
        warnings.append("One support label exceeds 70% of completed labels.")
    if repeated_prefix_share > 0.35:
        warnings.append("One requirement opening exceeds 35% of unique tasks.")
    if leaked_fields:
        warnings.append("Queue records contain fields that could reveal generation hints.")
    return {
        **summary,
        "conflicting_repeat_pairs": conflict_pair_count,
        "unique_tasks": len(unique_tasks),
        "role_counts": dict(role_counts),
        "source_counts": dict(source_counts),
        "position_max_share": position_max_share,
        "label_max_share": label_max_share,
        "repeated_requirement_prefix_share": repeated_prefix_share,
        "forbidden_queue_fields": leaked_fields,
        "warnings": warnings,
    }
