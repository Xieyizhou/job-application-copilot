"""Leakage-safe evaluation helpers for the frozen real-text holdout."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from typing import Any

import numpy as np

from ml.annotation_generation import normalize_text
from ml.annotation_metrics import decision_metrics, record_task_scores, retrieval_metrics


class RealHoldoutEvaluationError(ValueError):
    """Raised when holdout evaluation would violate split boundaries."""


def assert_holdout_isolated(
    holdout_tasks: Sequence[dict[str, Any]],
    training_tasks: Sequence[dict[str, Any]],
    training_pairs: Sequence[dict[str, Any]],
) -> None:
    """Reject content or source-document overlap with the training corpus."""
    holdout_pairs = {
        (
            normalize_text(str(task["requirement"])),
            normalize_text(str(candidate["evidence"])),
        )
        for task in holdout_tasks
        for candidate in task["candidates"]
    }
    train_pairs = {
        (
            normalize_text(str(pair["requirement"])),
            normalize_text(str(pair["evidence"])),
        )
        for pair in training_pairs
    }
    if holdout_pairs & train_pairs:
        raise RealHoldoutEvaluationError(
            "Holdout requirement/evidence content overlaps with training."
        )
    for field in ("source_job_hash", "source_resume_hash"):
        holdout_hashes = {
            str(task.get(field, ""))
            for task in holdout_tasks
            if str(task.get(field, ""))
        }
        training_hashes = {
            str(task.get(field, ""))
            for task in training_tasks
            if str(task.get(field, ""))
        }
        if holdout_hashes & training_hashes:
            raise RealHoldoutEvaluationError(
                f"Holdout {field} values overlap with training."
            )


def evaluate_holdout_method(
    tasks: Sequence[dict[str, Any]],
    *,
    score: Callable[[Sequence[str], Sequence[str]], Iterable[float]],
    threshold: float,
) -> dict[str, Any]:
    """Evaluate pair decisions, strongest-evidence ranking, and task rejection."""
    scored_tasks = score_holdout_tasks(tasks, score=score)
    return evaluate_scored_holdout(tasks, scored_tasks, threshold=threshold)


def score_holdout_tasks(
    tasks: Sequence[dict[str, Any]],
    *,
    score: Callable[[Sequence[str], Sequence[str]], Iterable[float]],
) -> dict[str, list[float]]:
    """Score every candidate once so threshold calibration cannot refit a model."""
    scored: dict[str, list[float]] = {}
    for task in tasks:
        candidates = list(task["candidates"])
        values = [
            float(value)
            for value in score(
                [str(task["requirement"])] * len(candidates),
                [str(candidate["evidence"]) for candidate in candidates],
            )
        ]
        if len(values) != len(candidates):
            raise RealHoldoutEvaluationError(
                f"Scorer returned wrong length for {task['task_id']}."
            )
        scored[str(task["task_id"])] = values
    return scored


def evaluate_scored_holdout(
    tasks: Sequence[dict[str, Any]],
    scored_tasks: dict[str, list[float]],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate cached candidate scores at one preselected threshold."""
    pair_labels: list[int] = []
    pair_scores: list[float] = []
    ranks: list[int] = []
    rejected: list[bool] = []
    task_decisions: list[bool] = []
    failures: list[dict[str, Any]] = []
    supported_successes: list[bool] = []
    for task in tasks:
        candidates = list(task["candidates"])
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        scores = scored_tasks.get(str(task["task_id"]), [])
        if len(scores) != len(candidates):
            raise RealHoldoutEvaluationError(
                f"Cached scores have wrong length for {task['task_id']}."
            )
        selected = task.get("selected_candidate_id")
        pair_labels.extend(
            int(selected is not None and candidate_id == str(selected))
            for candidate_id in candidate_ids
        )
        pair_scores.extend(scores)
        record_task_scores(
            task,
            scores,
            threshold,
            ranks=ranks,
            rejected=rejected,
            task_decisions=task_decisions,
        )
        top_index = int(np.argmax(np.asarray(scores)))
        if str(task["support_label"]) == "No Support":
            passed = max(scores) < threshold
            failure = "false_accept"
        else:
            gold_index = candidate_ids.index(str(selected))
            passed = top_index == gold_index and max(scores) >= threshold
            supported_successes.append(passed)
            failure = "wrong_rank" if top_index != gold_index else "below_threshold"
        if not passed:
            failures.append(
                {
                    "task_id": task["task_id"],
                    "failure": failure,
                    "top_candidate_id": candidate_ids[top_index],
                    "top_score": max(scores),
                }
            )
    predictions = [int(value >= threshold) for value in pair_scores]
    retrieval = retrieval_metrics(ranks, rejected, task_decisions)
    supported_rate = float(np.mean(supported_successes))
    rejection_rate = float(retrieval["no_support_rejection_rate"])
    retrieval.update(
        {
            "supported_task_success_rate": supported_rate,
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
        }
    )
    return {
        "threshold": threshold,
        "pair_classification": decision_metrics(
            pair_labels,
            predictions,
            pair_scores,
        ),
        "retrieval": retrieval,
        "failure_count": len(failures),
        "failures": failures,
    }


def select_validation_threshold(
    tasks: Sequence[dict[str, Any]],
    scored_tasks: dict[str, list[float]],
) -> tuple[float, dict[str, Any]]:
    """Select a development threshold without changing candidate rankings."""
    top_scores = [max(scored_tasks[str(task["task_id"])]) for task in tasks]
    thresholds = sorted(
        {
            min(top_scores),
            max(top_scores) + 1e-12,
            *(
                float(np.nextafter(score, float("inf")))
                for score in top_scores
            ),
        }
    )
    evaluated = [
        (
            threshold,
            evaluate_scored_holdout(tasks, scored_tasks, threshold=threshold),
        )
        for threshold in thresholds
    ]

    def selection_key(item: tuple[float, dict[str, Any]]) -> tuple[float, ...]:
        threshold, result = item
        retrieval = result["retrieval"]
        pair = result["pair_classification"]
        return (
            float(retrieval["task_balanced_accuracy"]),
            float(retrieval["task_decision_accuracy"]),
            float(retrieval["no_support_rejection_rate"]),
            float(pair["f1"]),
            threshold,
        )

    return max(evaluated, key=selection_key)
