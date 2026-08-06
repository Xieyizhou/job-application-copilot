"""Tests for the frozen MiniLM v3 objective and validation boundary."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from ml.evidence_multitask_v3 import (
    RETIRED_BATCH4_OUTER_FOLDS,
    assert_source_isolated_validation,
    coverage_aware_task_batches,
    coverage_logits,
    coverage_margin_loss,
    multitask_v3_loss,
    select_coverage_threshold,
)


def test_coverage_aware_batches_are_deterministic_and_preserve_every_task() -> None:
    task_ids = [f"task-{index}" for index in range(9)]
    labels = ["Direct", "Direct", "Partial", "Partial", "Partial", *(["No Support"] * 4)]

    batches = coverage_aware_task_batches(
        task_ids,
        labels,
        task_batch_size=4,
        random_state=23,
    )

    assert batches == coverage_aware_task_batches(
        task_ids,
        labels,
        task_batch_size=4,
        random_state=23,
    )
    assert sorted(task_id for batch in batches for task_id in batch) == sorted(task_ids)
    labels_by_id = dict(zip(task_ids, labels, strict=True))
    for batch in batches[:2]:
        batch_labels = {labels_by_id[task_id] for task_id in batch}
        assert {"Direct", "Partial"}.issubset(batch_labels)


def test_coverage_margin_pushes_direct_above_partial_and_masks_no_support() -> None:
    logits = torch.zeros((4, 2), requires_grad=True)
    targets = torch.tensor([1, 0, -100, -100])

    loss, diagnostics = coverage_margin_loss(logits, targets)
    loss.backward()

    margins = coverage_logits(logits)
    assert diagnostics["coverage_rows"] == 2.0
    assert diagnostics["coverage_pairs"] == 1.0
    assert margins.grad_fn is not None
    assert logits.grad is not None
    assert logits.grad[0, 1] < 0.0
    assert logits.grad[1, 1] > 0.0
    assert torch.equal(logits.grad[2:], torch.zeros((2, 2)))


def test_v3_support_loss_is_unweighted_and_heads_remain_independent() -> None:
    support = torch.zeros(8, requires_grad=True)
    strength = torch.zeros((8, 2), requires_grad=True)
    rank = torch.zeros(8, requires_grad=True)
    support_targets = torch.tensor([1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    strength_targets = torch.tensor([1, 0, -100, -100, -100, -100, -100, -100])

    loss, diagnostics = multitask_v3_loss(
        support,
        strength,
        rank,
        support_targets,
        strength_targets,
        [(0, 1, 2, 3), (4, 5, 6, 7)],
        [0, None],
    )
    loss.backward()

    assert diagnostics["support_loss"] == pytest.approx(np.log(2.0))
    assert diagnostics["coverage_pairs"] == 1.0
    assert diagnostics["ranking_pairs"] == 3.0
    assert support.grad is not None
    assert strength.grad is not None
    assert rank.grad is not None


def test_coverage_threshold_requires_both_class_recalls() -> None:
    decision = select_coverage_threshold(
        [0.10, 0.20, 0.40, 0.45, 0.55, 0.70],
        [0, 0, 1, 0, 1, 1],
        minimum_class_recall=0.60,
    )

    assert decision.eligible is True
    assert decision.direct_recall >= 0.60
    assert decision.partial_recall >= 0.60


def test_batch4_folds_are_retired_and_new_holdout_is_preassigned() -> None:
    assert RETIRED_BATCH4_OUTER_FOLDS == (0, 1, 2, 3, 4)
    with pytest.raises(ValueError, match="source-isolated"):
        assert_source_isolated_validation(
            "reviewed_evidence_training_v4_batch4",
            source_partition_frozen=True,
            sealed_holdout_assigned=True,
        )
    with pytest.raises(ValueError, match="frozen together"):
        assert_source_isolated_validation(
            "successor_v9_development",
            source_partition_frozen=True,
            sealed_holdout_assigned=False,
        )
    assert_source_isolated_validation(
        "successor_v9_development",
        source_partition_frozen=True,
        sealed_holdout_assigned=True,
    ) is None
