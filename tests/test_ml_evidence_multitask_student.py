"""Tests for candidate-complete multi-task MiniLM contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.evidence_multitask_student import (
    MULTITASK_REJECTION_FEATURE_NAMES,
    multitask_loss,
    multitask_rejection_features,
    multitask_targets,
    validate_candidate_complete_dataset,
)


def _dataset() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    candidates = [
        {"candidate_id": "a", "evidence": "strong", "support_label": "Direct"},
        {"candidate_id": "b", "evidence": "partial", "support_label": "Partial"},
        {"candidate_id": "c", "evidence": "weak", "support_label": "No Support"},
        {"candidate_id": "d", "evidence": "none", "support_label": "No Support"},
    ]
    task = {
        "task_id": "task",
        "evaluation_group": "group",
        "role_family": "Data",
        "support_label": "Direct",
        "selected_candidate_id": "a",
        "candidates": candidates,
    }
    pairs = [
        {
            "pair_id": f"pair-{index}",
            "task_id": "task",
            "evaluation_group": "group",
            "role_family": "Data",
            "support_label": candidate["support_label"],
            "evidence": candidate["evidence"],
        }
        for index, candidate in enumerate(candidates)
    ]
    return [task], pairs


def test_candidate_complete_gate_aligns_all_four_candidates() -> None:
    tasks, pairs = _dataset()
    result = validate_candidate_complete_dataset(
        tasks,
        pairs,
        expected_tasks=1,
        expected_pairs=4,
        pairs_per_task=4,
    )

    assert result.task_pair_indices == {"task": (0, 1, 2, 3)}
    assert result.selected_pair_index == {"task": 0}
    assert result.manifest["teacher_scores_used"] is False


def test_candidate_complete_gate_rejects_missing_candidate_pair() -> None:
    tasks, pairs = _dataset()
    with pytest.raises(ValueError, match="unexpected size"):
        validate_candidate_complete_dataset(
            tasks,
            pairs[:-1],
            expected_tasks=1,
            expected_pairs=4,
            pairs_per_task=4,
        )


def test_candidate_complete_gate_requires_selected_label_to_match_task() -> None:
    tasks, pairs = _dataset()
    tasks[0]["support_label"] = "Partial"

    with pytest.raises(ValueError, match="labels must agree"):
        validate_candidate_complete_dataset(
            tasks,
            pairs,
            expected_tasks=1,
            expected_pairs=4,
            pairs_per_task=4,
        )


def test_multitask_targets_mask_no_support_strength() -> None:
    _, pairs = _dataset()
    support, strength = multitask_targets(pairs)

    assert support.tolist() == [1.0, 1.0, 0.0, 0.0]
    assert strength.tolist() == [1, 0, -100, -100]


def test_multitask_loss_masks_strength_and_no_support_ranking() -> None:
    support_logits = torch.zeros(8, requires_grad=True)
    strength_logits = torch.zeros((8, 2), requires_grad=True)
    rank_scores = torch.tensor(
        [2.0, 1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
        requires_grad=True,
    )
    support_targets = torch.tensor([1, 1, 0, 0, 0, 0, 0, 0], dtype=torch.float32)
    strength_targets = torch.tensor([1, 0, -100, -100, -100, -100, -100, -100])

    loss, diagnostics = multitask_loss(
        support_logits,
        strength_logits,
        rank_scores,
        support_targets,
        strength_targets,
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        [0, None],
        support_pos_weight=torch.tensor(1.0),
    )

    assert diagnostics["strength_rows"] == 2.0
    assert diagnostics["ranking_pairs"] == 3.0
    assert diagnostics["ranking_loss"] < float(torch.nn.functional.softplus(torch.tensor(0.0)))
    loss.backward()
    assert rank_scores.grad[0] < 0
    assert torch.all(rank_scores.grad[4:] == 0)


def test_rejection_features_are_fixed_and_finite() -> None:
    features = multitask_rejection_features(
        [0.9, 0.6, 0.2, 0.1],
        [0.8, 0.4, 0.5, 0.5],
        [2.0, 1.0, 0.0, -1.0],
    )

    assert features.shape == (11,)
    assert len(MULTITASK_REJECTION_FEATURE_NAMES) == len(features)
    assert np.isfinite(features).all()
