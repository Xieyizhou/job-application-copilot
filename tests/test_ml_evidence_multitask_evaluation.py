"""Tests for leakage-safe multi-task student evaluation helpers."""

from __future__ import annotations

import numpy as np

from ml.evidence_multitask_evaluation import (
    CandidateOutputs,
    evaluate_outputs,
    fit_rejector,
    grouped_task_folds,
)
from ml.evidence_multitask_student import validate_candidate_complete_dataset


def _task(
    task_id: str,
    group: str,
    label: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_labels = (
        ["Direct", "Partial", "No Support", "No Support"]
        if label != "No Support"
        else ["No Support"] * 4
    )
    candidates = [
        {
            "candidate_id": f"{task_id}-candidate-{index}",
            "evidence": f"{task_id}-evidence-{index}",
            "support_label": candidate_label,
        }
        for index, candidate_label in enumerate(candidate_labels)
    ]
    task = {
        "task_id": task_id,
        "evaluation_group": group,
        "role_family": "Data" if int(task_id[-1]) % 2 else "Software",
        "support_label": label,
        "selected_candidate_id": (
            candidates[0]["candidate_id"] if label != "No Support" else None
        ),
        "candidates": candidates,
    }
    pairs = [
        {
            "pair_id": f"{task_id}-pair-{index}",
            "task_id": task_id,
            "evaluation_group": group,
            "role_family": task["role_family"],
            "requirement": f"requirement {task_id}",
            "evidence": candidate["evidence"],
            "support_label": candidate["support_label"],
        }
        for index, candidate in enumerate(candidates)
    ]
    return task, pairs


def _dataset() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    tasks: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for index in range(8):
        task, task_pairs = _task(
            f"task-{index}",
            f"group-{index // 2}",
            "Direct" if index % 2 else "No Support",
        )
        tasks.append(task)
        pairs.extend(task_pairs)
    return tasks, pairs


def test_grouped_task_folds_keep_evaluation_groups_isolated() -> None:
    tasks, _ = _dataset()
    folds = grouped_task_folds(
        tasks,
        group_field="evaluation_group",
        n_splits=2,
        random_state=17,
    )
    task_groups = {
        str(task["task_id"]): str(task["evaluation_group"])
        for task in tasks
    }

    assert len(folds) == 2
    assert folds == grouped_task_folds(
        tasks,
        group_field="evaluation_group",
        n_splits=2,
        random_state=17,
    )
    for train, test in folds:
        assert {task_groups[task_id] for task_id in train}.isdisjoint(
            {task_groups[task_id] for task_id in test}
        )


def test_rejector_threshold_uses_complete_grouped_oof_predictions() -> None:
    features = np.asarray(
        [[index, index % 2, index / 10] for index in range(12)],
        dtype=np.float64,
    )
    labels = [index % 2 for index in range(12)]
    groups = [f"group-{index // 2}" for index in range(12)]

    model, threshold, oof = fit_rejector(
        features,
        labels,
        groups,
        random_state=23,
        n_splits=3,
    )

    assert np.isfinite(oof).all()
    assert 0.0 <= threshold <= 1.0
    assert model.predict_proba(features).shape == (12, 2)


def test_evaluation_keeps_ranking_rejection_and_strength_independent() -> None:
    support_task, support_pairs = _task("task-1", "group-1", "Direct")
    reject_task, reject_pairs = _task("task-2", "group-2", "No Support")
    tasks = [support_task, reject_task]
    pairs = [*support_pairs, *reject_pairs]
    validated = validate_candidate_complete_dataset(
        tasks,
        pairs,
        expected_tasks=2,
        expected_pairs=8,
        pairs_per_task=4,
    )
    outputs = CandidateOutputs(
        support=np.asarray([0.9, 0.7, 0.1, 0.1, 0.2, 0.1, 0.1, 0.1]),
        strength=np.asarray([0.8, 0.2, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        rank=np.asarray([3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0]),
    )

    result = evaluate_outputs(
        {str(task["task_id"]): task for task in tasks},
        pairs,
        validated,
        ["task-1", "task-2"],
        outputs,
        [0.9, 0.1],
        threshold=0.5,
    )

    assert result["retrieval"]["task_balanced_accuracy"] == 1.0
    assert result["task_label_metrics"]["confusion_matrix"] == [
        [1, 0, 0],
        [0, 0, 0],
        [0, 0, 1],
    ]
    assert result["slices"]["role_family"]["Data"][
        "task_balanced_accuracy"
    ] == 1.0
