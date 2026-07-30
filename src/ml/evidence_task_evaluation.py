"""Grouped evaluation helpers for candidate-complete evidence tasks."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from ml.evidence_multiclass import MulticlassEvidenceReranker
from ml.evidence_task_rejection import (
    TaskSupportRejector,
    task_rejection_features,
)


PairRow = tuple[str, str, str]
TaskProbabilities = dict[str, list[list[float]]]


def fit_candidate(
    rows: Sequence[PairRow],
    *,
    random_state: int,
) -> MulticlassEvidenceReranker:
    """Fit the multiclass candidate model on explicit text-label rows."""
    return MulticlassEvidenceReranker(random_state=random_state).fit(
        [row[0] for row in rows],
        [row[1] for row in rows],
        [row[2] for row in rows],
    )


def pair_rows(pairs: Sequence[Mapping[str, Any]]) -> list[PairRow]:
    """Convert serialized pair records to model input rows."""
    return [
        (
            str(pair["requirement"]),
            str(pair["evidence"]),
            str(pair["support_label"]),
        )
        for pair in pairs
    ]


def predict_tasks(
    model: MulticlassEvidenceReranker,
    tasks: Sequence[Mapping[str, Any]],
) -> TaskProbabilities:
    """Predict all candidates while retaining task boundaries."""
    predictions: TaskProbabilities = {}
    for task in tasks:
        candidates = list(task["candidates"])
        requirements = [str(task["requirement"])] * len(candidates)
        evidence = [str(candidate["evidence"]) for candidate in candidates]
        predictions[str(task["task_id"])] = (
            model.predict_class_proba(requirements, evidence)
            .astype(float)
            .tolist()
        )
    return predictions


def base_oof_candidate_probabilities(
    tasks: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    random_state: int,
    n_splits: int = 5,
) -> TaskProbabilities:
    """Generate candidate probabilities without in-fold training leakage."""
    labels = np.asarray([str(pair["support_label"]) for pair in pairs])
    groups = np.asarray([str(pair["evaluation_group"]) for pair in pairs])
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    probabilities = np.empty((len(pairs), 3), dtype=np.float64)
    rows = pair_rows(pairs)
    for fold, (train, test) in enumerate(
        splitter.split(np.zeros(len(pairs)), labels, groups)
    ):
        model = fit_candidate(
            [rows[int(index)] for index in train],
            random_state=random_state + fold,
        )
        probabilities[test] = model.predict_class_proba(
            [rows[int(index)][0] for index in test],
            [rows[int(index)][1] for index in test],
        )
    return _align_task_probabilities(tasks, pairs, probabilities)


def task_feature_rows(
    tasks: Sequence[Mapping[str, Any]],
    probabilities: Mapping[str, Sequence[Sequence[float]]],
) -> tuple[list[list[float]], list[int], list[str]]:
    """Build rejector features, binary labels, and leakage-safe groups."""
    return (
        [
            task_rejection_features(
                probabilities[str(task["task_id"])]
            ).tolist()
            for task in tasks
        ],
        [
            int(str(task["support_label"]) != "No Support")
            for task in tasks
        ],
        [
            str(
                task.get("evaluation_group")
                or task.get("source_resume_hash")
                or task["task_id"]
            )
            for task in tasks
        ],
    )


def crossfit_rejector(
    features: Sequence[Sequence[float]],
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    random_state: int,
    n_splits: int = 5,
) -> tuple[TaskSupportRejector, np.ndarray]:
    """Return a final rejector and leakage-safe out-of-fold scores."""
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    matrix = np.asarray(features, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    oof = np.empty(len(labels), dtype=np.float64)
    for fold, (train, test) in enumerate(
        splitter.split(matrix, y, group_values)
    ):
        model = TaskSupportRejector(
            random_state=random_state + fold
        ).fit(matrix[train], y[train])
        oof[test] = model.predict_proba(matrix[test])
    final = TaskSupportRejector(random_state=random_state).fit(
        matrix.tolist(),
        y.tolist(),
    )
    return final, oof


def grouped_task_assignments(
    tasks: Sequence[Mapping[str, Any]],
    *,
    random_state: int,
    n_splits: int = 4,
) -> dict[str, int]:
    """Assign all tasks from one resume to the same evaluation fold."""
    labels = [
        int(str(task["support_label"]) != "No Support") for task in tasks
    ]
    groups = [str(task["source_resume_hash"]) for task in tasks]
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    assignments: dict[str, int] = {}
    for fold, (_, test) in enumerate(splitter.split(tasks, labels, groups)):
        for index in test:
            assignments[str(tasks[int(index)]["task_id"])] = fold
    return assignments


def aggregate_fold_results(
    results: Sequence[Mapping[str, Any]],
) -> dict[str, float | int]:
    """Aggregate fold metrics with their correct task denominators."""
    support_tasks = sum(
        int(result["retrieval"]["support_tasks"]) for result in results
    )
    no_support_tasks = sum(
        int(result["retrieval"]["no_support_tasks"]) for result in results
    )
    total = support_tasks + no_support_tasks

    def weighted(metric: str, denominator: str) -> float:
        count = sum(
            int(result["retrieval"][denominator]) for result in results
        )
        return (
            sum(
                float(result["retrieval"][metric])
                * int(result["retrieval"][denominator])
                for result in results
            )
            / count
            if count
            else 0.0
        )

    supported = weighted("supported_task_success_rate", "support_tasks")
    rejection = weighted("no_support_rejection_rate", "no_support_tasks")
    accuracy = (
        sum(
            float(result["retrieval"]["task_decision_accuracy"])
            * (
                int(result["retrieval"]["support_tasks"])
                + int(result["retrieval"]["no_support_tasks"])
            )
            for result in results
        )
        / total
    )
    return {
        "support_tasks": support_tasks,
        "no_support_tasks": no_support_tasks,
        "recall_at_1": weighted("recall_at_1", "support_tasks"),
        "recall_at_3": weighted("recall_at_3", "support_tasks"),
        "mean_reciprocal_rank": weighted(
            "mean_reciprocal_rank", "support_tasks"
        ),
        "supported_task_success_rate": supported,
        "no_support_rejection_rate": rejection,
        "task_decision_accuracy": accuracy,
        "task_balanced_accuracy": (supported + rejection) / 2.0,
    }


def summarize_failure_slices(
    tasks: Sequence[Mapping[str, Any]],
    results: Sequence[Mapping[str, Any]],
    metadata_by_task: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Aggregate failures without including requirement or evidence text."""
    outcomes = {
        str(task_id): bool(passed)
        for result in results
        for task_id, passed in result["outcomes"].items()
    }
    failure_types = {
        str(failure["task_id"]): str(failure["failure"])
        for result in results
        for failure in result["failures"]
    }
    slices: dict[str, Any] = {
        "failure_counts": dict(Counter(failure_types.values()))
    }
    dimensions = (
        "role_family",
        "profile_role_family",
        "pairing_type",
        "source_dataset",
    )
    for dimension in dimensions:
        buckets: dict[str, dict[str, Any]] = {}
        for task in tasks:
            task_id = str(task["task_id"])
            metadata = metadata_by_task.get(task_id, {})
            value = str(metadata.get(dimension, "unknown"))
            bucket = buckets.setdefault(
                value,
                {"tasks": 0, "passed": 0, "failure_counts": Counter()},
            )
            bucket["tasks"] += 1
            if outcomes[task_id]:
                bucket["passed"] += 1
            else:
                bucket["failure_counts"][failure_types[task_id]] += 1
        slices[f"by_{dimension}"] = {
            value: {
                "tasks": bucket["tasks"],
                "accuracy": bucket["passed"] / bucket["tasks"],
                "failure_counts": dict(bucket["failure_counts"]),
            }
            for value, bucket in sorted(buckets.items())
        }
    return slices


def _align_task_probabilities(
    tasks: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    probabilities: np.ndarray,
) -> TaskProbabilities:
    if len(pairs) != len(probabilities):
        raise ValueError("Pair probabilities do not align.")
    by_task: dict[str, dict[str, list[float]]] = {}
    for pair, row in zip(pairs, probabilities, strict=True):
        task_id = str(pair["task_id"])
        evidence = str(pair["evidence"])
        if evidence in by_task.setdefault(task_id, {}):
            raise ValueError(f"Duplicate candidate evidence in {task_id}.")
        by_task[task_id][evidence] = row.astype(float).tolist()
    result: TaskProbabilities = {}
    for task in tasks:
        task_id = str(task["task_id"])
        result[task_id] = [
            by_task[task_id][str(candidate["evidence"])]
            for candidate in task["candidates"]
        ]
    return result
