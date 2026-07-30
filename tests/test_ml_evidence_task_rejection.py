"""Tests for task-level evidence rejection."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evidence_task_evaluation import summarize_failure_slices
from ml.evidence_task_rejection import (
    TASK_REJECTION_FEATURE_NAMES,
    TaskSupportRejector,
    evaluate_task_policy,
    select_task_rejection_threshold,
    task_rejection_features,
)


def _probabilities() -> list[list[float]]:
    return [
        [0.75, 0.15, 0.10],
        [0.10, 0.20, 0.70],
        [0.05, 0.10, 0.85],
        [0.02, 0.08, 0.90],
    ]


def _task(label: str = "Direct") -> dict[str, object]:
    return {
        "task_id": "task-1",
        "support_label": label,
        "selected_candidate_id": "a" if label != "No Support" else None,
        "candidates": [
            {"candidate_id": key, "evidence": key}
            for key in ("a", "b", "c", "d")
        ],
    }


def test_task_features_capture_support_distribution() -> None:
    features = task_rejection_features(_probabilities())

    assert features.shape == (len(TASK_REJECTION_FEATURE_NAMES),)
    assert features[0] == pytest.approx(0.90)
    assert features[1] == pytest.approx(0.30)
    assert features[10] == 1


def test_rejector_and_threshold_require_both_classes() -> None:
    positive = task_rejection_features(_probabilities())
    negative = task_rejection_features(
        [[0.02, 0.08, 0.90]] * 4
    )
    model = TaskSupportRejector().fit(
        [positive, negative, positive * 0.9, negative * 1.1],
        [1, 0, 1, 0],
    )
    scores = model.predict_proba([positive, negative])
    threshold = select_task_rejection_threshold([1, 0], scores)

    assert 0.0 <= threshold <= 1.0
    assert model.feature_manifest()["features"]


def test_task_policy_preserves_ranking_and_can_reject() -> None:
    accepted = evaluate_task_policy(
        [_task()],
        {"task-1": _probabilities()},
        acceptance_probabilities={"task-1": 0.8},
        acceptance_threshold=0.5,
    )
    rejected = evaluate_task_policy(
        [_task()],
        {"task-1": _probabilities()},
        acceptance_probabilities={"task-1": 0.2},
        acceptance_threshold=0.5,
    )

    assert accepted["retrieval"]["recall_at_1"] == 1.0
    assert accepted["retrieval"]["supported_task_success_rate"] == 1.0
    assert rejected["failures"][0]["failure"] == "support_reject"


def test_invalid_probability_shape_is_rejected() -> None:
    with pytest.raises(ValueError, match="class count"):
        task_rejection_features(np.ones((4, 2)))


def test_failure_slices_exclude_text_and_group_metadata() -> None:
    tasks = [
        {"task_id": "a", "requirement": "private text"},
        {"task_id": "b", "requirement": "other private text"},
    ]
    results = [
        {
            "outcomes": {"a": False, "b": True},
            "failures": [{"task_id": "a", "failure": "wrong_rank"}],
        }
    ]
    metadata = {
        "a": {
            "role_family": "Data",
            "profile_role_family": "Data",
            "pairing_type": "aligned",
            "source_dataset": "synthetic",
        },
        "b": {
            "role_family": "ML",
            "profile_role_family": "Software",
            "pairing_type": "cross_role",
            "source_dataset": "synthetic",
        },
    }

    slices = summarize_failure_slices(tasks, results, metadata)

    assert slices["failure_counts"] == {"wrong_rank": 1}
    assert slices["by_role_family"]["Data"]["accuracy"] == 0.0
    assert slices["by_pairing_type"]["cross_role"]["accuracy"] == 1.0
    assert "private text" not in str(slices)
