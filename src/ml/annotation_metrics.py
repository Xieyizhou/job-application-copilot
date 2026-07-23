"""Metrics for requirement/evidence pair and retrieval experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def decision_metrics(
    labels: Sequence[int],
    predictions: Sequence[int],
    scores: Sequence[float],
) -> dict[str, Any]:
    """Return aggregate binary metrics for fold-specific decisions."""
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "examples": len(labels),
        "positive_rate": float(np.mean(labels)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def retrieval_metrics(
    ranks: list[int],
    rejected_no_support: list[bool],
    task_decisions: list[bool],
) -> dict[str, Any]:
    """Return ranking and no-support rejection metrics."""
    return {
        "support_tasks": len(ranks),
        "recall_at_1": float(np.mean([rank == 1 for rank in ranks])),
        "recall_at_3": float(np.mean([rank <= 3 for rank in ranks])),
        "mean_reciprocal_rank": float(np.mean([1 / rank for rank in ranks])),
        "no_support_tasks": len(rejected_no_support),
        "no_support_rejection_rate": float(np.mean(rejected_no_support)),
        "task_decision_accuracy": float(np.mean(task_decisions)),
    }


def record_task_scores(
    task: dict[str, Any],
    scores: Sequence[float],
    threshold: float,
    *,
    ranks: list[int],
    rejected: list[bool],
    task_decisions: list[bool],
) -> None:
    """Accumulate one task's retrieval and rejection outcome."""
    label = str(task["support_label"])
    if label == "No Support":
        decision = max(scores) < threshold
        rejected.append(decision)
        task_decisions.append(decision)
        return
    candidate_ids = [str(candidate["candidate_id"]) for candidate in task["candidates"]]
    gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
    order = np.argsort(-np.asarray(scores))
    rank = int(np.where(order == gold_index)[0][0]) + 1
    ranks.append(rank)
    task_decisions.append(rank == 1 and max(scores) >= threshold)
