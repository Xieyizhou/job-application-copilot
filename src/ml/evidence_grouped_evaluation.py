"""Shared metrics for candidate-complete evidence evaluation."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
)

from ml.evidence_multiclass import SUPPORT_CLASSES


def classification_metrics(
    labels: list[str],
    predictions: list[str],
) -> dict[str, Any]:
    """Return stable three-class pair metrics."""
    return {
        "examples": len(labels),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels, predictions)
        ),
        "macro_f1": float(
            f1_score(
                labels,
                predictions,
                labels=list(SUPPORT_CLASSES),
                average="macro",
                zero_division=0,
            )
        ),
        "per_class_f1": {
            label: float(value)
            for label, value in zip(
                SUPPORT_CLASSES,
                f1_score(
                    labels,
                    predictions,
                    labels=list(SUPPORT_CLASSES),
                    average=None,
                    zero_division=0,
                ),
                strict=True,
            )
        },
        "labels": list(SUPPORT_CLASSES),
        "confusion_matrix": confusion_matrix(
            labels,
            predictions,
            labels=list(SUPPORT_CLASSES),
        )
        .astype(int)
        .tolist(),
        "prediction_counts": dict(Counter(predictions)),
    }


def _mean(values: list[bool] | list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def task_retrieval_metrics(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    predictions: list[str],
    support_scores: list[float],
) -> dict[str, Any]:
    """Evaluate ranking and support rejection across complete candidate sets."""
    metrics, _ = task_retrieval_evaluation(
        tasks,
        pairs,
        predictions,
        support_scores,
    )
    return metrics


def task_retrieval_evaluation(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    predictions: list[str],
    support_scores: list[float],
) -> tuple[dict[str, Any], dict[str, bool]]:
    """Return aggregate retrieval metrics and paired task outcomes."""
    if not (len(pairs) == len(predictions) == len(support_scores)):
        raise ValueError("pair predictions and support scores must align")
    pair_indices: defaultdict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        pair_indices[str(pair["task_id"])].append(index)

    ranks: list[int] = []
    supported_acceptance: list[bool] = []
    supported_success: list[bool] = []
    no_support_rejection: list[bool] = []
    task_outcomes: dict[str, bool] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        indices = pair_indices.get(task_id, [])
        candidates = list(task["candidates"])
        if len(indices) != len(candidates):
            raise ValueError(f"candidate pairs do not cover task {task_id}")
        ordered = sorted(
            indices,
            key=lambda index: (
                -support_scores[index],
                str(pairs[index]["pair_id"]),
            ),
        )
        accepted = any(predictions[index] != "No Support" for index in indices)
        if str(task["support_label"]) == "No Support":
            no_support_rejection.append(not accepted)
            task_outcomes[task_id] = not accepted
            continue
        selected_id = str(task["selected_candidate_id"])
        selected_evidence = next(
            str(candidate["evidence"])
            for candidate in candidates
            if str(candidate["candidate_id"]) == selected_id
        )
        gold_indices = [
            index
            for index in indices
            if str(pairs[index]["evidence"]) == selected_evidence
        ]
        if len(gold_indices) != 1:
            raise ValueError(f"selected evidence does not map uniquely for {task_id}")
        rank = ordered.index(gold_indices[0]) + 1
        ranks.append(rank)
        supported_acceptance.append(accepted)
        passed = accepted and rank == 1
        supported_success.append(passed)
        task_outcomes[task_id] = passed
    support_success_rate = _mean(supported_success)
    rejection_rate = _mean(no_support_rejection)
    metrics = {
        "support_tasks": len(ranks),
        "no_support_tasks": len(no_support_rejection),
        "recall_at_1": _mean([rank == 1 for rank in ranks]),
        "recall_at_3": _mean([rank <= 3 for rank in ranks]),
        "mean_reciprocal_rank": _mean([1.0 / rank for rank in ranks]),
        "supported_task_acceptance_rate": _mean(supported_acceptance),
        "supported_task_success_rate": support_success_rate,
        "no_support_rejection_rate": rejection_rate,
        "task_balanced_accuracy": (
            support_success_rate + rejection_rate
        ) / 2.0,
    }
    return metrics, task_outcomes
