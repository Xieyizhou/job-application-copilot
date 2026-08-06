"""Frozen objective contract for the next human-label-only MiniLM successor."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import math
import random
from typing import Any

import numpy as np
import torch


RETIRED_BATCH4_OUTER_FOLDS = (0, 1, 2, 3, 4)
RETIRED_EVALUATION_DATASETS = frozenset(
    {
        "reviewed_evidence_training_v4_batch4",
        "multitask_minilm_v1_batch4",
        "multitask_minilm_v2_batch4",
        "multitask_minilm_v2_1_batch4",
    }
)


@dataclass(frozen=True)
class CoverageThreshold:
    """One inner-development Direct/Partial threshold decision."""

    threshold: float
    eligible: bool
    balanced_accuracy: float
    direct_recall: float
    partial_recall: float


def coverage_aware_task_batches(
    task_ids: Sequence[str],
    task_labels: Sequence[str],
    *,
    task_batch_size: int,
    random_state: int,
) -> list[list[str]]:
    """Build deterministic batches with Direct/Partial pairs whenever available."""
    if (
        len(task_ids) != len(task_labels)
        or len(task_ids) != len(set(task_ids))
        or any(not task_id for task_id in task_ids)
        or any(label not in {"Direct", "Partial", "No Support"} for label in task_labels)
        or task_batch_size < 2
        or not isinstance(random_state, int)
    ):
        raise ValueError("Coverage-aware batch inputs are invalid.")
    by_label = {
        label: [
            task_id
            for task_id, task_label in zip(task_ids, task_labels, strict=True)
            if task_label == label
        ]
        for label in ("Direct", "Partial", "No Support")
    }
    if not by_label["Direct"] or not by_label["Partial"]:
        raise ValueError("Coverage-aware batching requires Direct and Partial tasks.")
    generator = random.Random(random_state)
    for values in by_label.values():
        generator.shuffle(values)
    remaining = {label: list(values) for label, values in by_label.items()}
    batches: list[list[str]] = []
    while any(remaining.values()):
        batch: list[str] = []
        if remaining["Direct"] and remaining["Partial"]:
            batch.extend([remaining["Direct"].pop(), remaining["Partial"].pop()])
        paired_capacity = min(len(remaining["Direct"]), len(remaining["Partial"]))
        pool = [("No Support", task_id) for task_id in remaining["No Support"]]
        pool.extend(
            (label, task_id)
            for label in ("Direct", "Partial")
            for task_id in remaining[label][paired_capacity:]
        )
        if not remaining["Direct"] or not remaining["Partial"]:
            pool.extend(
                (label, task_id)
                for label in ("Direct", "Partial")
                for task_id in remaining[label][:paired_capacity]
            )
        generator.shuffle(pool)
        for label, task_id in pool:
            if len(batch) >= task_batch_size:
                break
            if task_id not in remaining[label]:
                continue
            remaining[label].remove(task_id)
            batch.append(task_id)
        batches.append(batch)
    flattened = [task_id for batch in batches for task_id in batch]
    if len(flattened) != len(task_ids) or set(flattened) != set(task_ids):
        raise ValueError("Coverage-aware batching lost or duplicated tasks.")
    return batches


def coverage_logits(strength_logits: Any) -> Any:
    """Return the scalar Direct-over-Partial margin from the existing head."""
    if strength_logits.ndim != 2 or strength_logits.shape[1] != 2:
        raise ValueError("Coverage logits require two strength columns.")
    return strength_logits[:, 1] - strength_logits[:, 0]


def coverage_margin_loss(
    strength_logits: Any,
    strength_targets: Any,
    *,
    margin: float = 0.5,
    pointwise_weight: float = 0.25,
) -> tuple[Any, dict[str, float]]:
    """Optimize Direct above Partial while masking all No Support candidates."""
    import torch.nn.functional as functional

    if (
        strength_logits.ndim != 2
        or strength_logits.shape[1] != 2
        or strength_targets.shape != (strength_logits.shape[0],)
        or not math.isfinite(margin)
        or margin < 0.0
        or not math.isfinite(pointwise_weight)
        or pointwise_weight < 0.0
    ):
        raise ValueError("Coverage-margin inputs are invalid.")
    supported = strength_targets != -100
    if not bool(supported.any()):
        zero = strength_logits.sum() * 0.0
        return zero, {
            "coverage_pairwise_loss": 0.0,
            "coverage_pointwise_loss": 0.0,
            "coverage_rows": 0.0,
            "coverage_pairs": 0.0,
        }
    targets = strength_targets[supported]
    if bool(torch.any((targets != 0) & (targets != 1))):
        raise ValueError("Coverage targets must be Direct, Partial, or masked.")
    scores = coverage_logits(strength_logits)[supported]
    pointwise = functional.binary_cross_entropy_with_logits(
        scores,
        targets.to(dtype=scores.dtype),
    )
    direct_scores = scores[targets == 1]
    partial_scores = scores[targets == 0]
    if len(direct_scores) and len(partial_scores):
        differences = direct_scores[:, None] - partial_scores[None, :]
        pairwise = functional.softplus(margin - differences).mean()
        pair_count = int(differences.numel())
    else:
        pairwise = scores.sum() * 0.0
        pair_count = 0
    total = pairwise + pointwise_weight * pointwise
    return total, {
        "coverage_pairwise_loss": float(pairwise.detach().cpu()),
        "coverage_pointwise_loss": float(pointwise.detach().cpu()),
        "coverage_rows": float(supported.sum().item()),
        "coverage_pairs": float(pair_count),
    }


def multitask_v3_loss(
    support_logits: Any,
    strength_logits: Any,
    rank_scores: Any,
    support_targets: Any,
    strength_targets: Any,
    task_pair_indices: Sequence[Sequence[int]],
    selected_pair_indices: Sequence[int | None],
    *,
    coverage_margin: float = 0.5,
    coverage_pointwise_weight: float = 0.25,
    support_weight: float = 1.0,
    coverage_weight: float = 0.5,
    ranking_weight: float = 0.5,
) -> tuple[Any, dict[str, float]]:
    """Keep support, coverage strength, and strongest ranking independent."""
    import torch.nn.functional as functional

    count = int(support_logits.shape[0])
    if (
        support_logits.ndim != 1
        or rank_scores.shape != support_logits.shape
        or strength_logits.shape != (count, 2)
        or support_targets.shape != support_logits.shape
        or strength_targets.shape != support_logits.shape
        or any(
            not math.isfinite(weight) or weight < 0.0
            for weight in (support_weight, coverage_weight, ranking_weight)
        )
    ):
        raise ValueError("MiniLM v3 loss inputs are misaligned.")
    support_loss = functional.binary_cross_entropy_with_logits(
        support_logits,
        support_targets,
    )
    coverage_loss, coverage_diagnostics = coverage_margin_loss(
        strength_logits,
        strength_targets,
        margin=coverage_margin,
        pointwise_weight=coverage_pointwise_weight,
    )
    ranking_terms: list[torch.Tensor] = []
    for indices, selected in zip(
        task_pair_indices,
        selected_pair_indices,
        strict=True,
    ):
        if selected is None:
            continue
        if selected not in indices:
            raise ValueError("Selected pair must belong to its task.")
        ranking_terms.extend(
            functional.softplus(-(rank_scores[selected] - rank_scores[index]))
            for index in indices
            if index != selected
        )
    ranking_loss = (
        torch.stack(ranking_terms).mean() if ranking_terms else support_logits.sum() * 0.0
    )
    total = (
        support_weight * support_loss
        + coverage_weight * coverage_loss
        + ranking_weight * ranking_loss
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("MiniLM v3 loss is not finite.")
    return total, {
        "support_loss": float(support_loss.detach().cpu()),
        "coverage_loss": float(coverage_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "ranking_pairs": float(len(ranking_terms)),
        **coverage_diagnostics,
    }


def select_coverage_threshold(
    direct_probabilities: Sequence[float],
    direct_labels: Sequence[int | bool],
    *,
    minimum_class_recall: float,
) -> CoverageThreshold:
    """Choose a threshold only when both Direct and Partial recall are adequate."""
    probabilities = np.asarray(direct_probabilities, dtype=np.float64)
    labels = np.asarray(direct_labels, dtype=np.int64)
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or labels.shape != probabilities.shape
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
        or set(labels.tolist()) != {0, 1}
        or not math.isfinite(minimum_class_recall)
        or not 0.0 <= minimum_class_recall <= 1.0
    ):
        raise ValueError("Coverage threshold inputs are invalid.")
    unique = np.sort(np.unique(probabilities))
    thresholds = sorted(
        {
            0.0,
            1.0,
            *unique.tolist(),
            *(
                float((left + right) / 2.0)
                for left, right in zip(unique[:-1], unique[1:], strict=True)
            ),
        }
    )
    candidates: list[CoverageThreshold] = []
    for threshold in thresholds:
        predicted_direct = probabilities >= threshold
        direct_recall = float(np.mean(predicted_direct[labels == 1]))
        partial_recall = float(np.mean(~predicted_direct[labels == 0]))
        candidates.append(
            CoverageThreshold(
                threshold=threshold,
                eligible=(
                    direct_recall >= minimum_class_recall and partial_recall >= minimum_class_recall
                ),
                balanced_accuracy=(direct_recall + partial_recall) / 2.0,
                direct_recall=direct_recall,
                partial_recall=partial_recall,
            )
        )

    def key(item: CoverageThreshold) -> tuple[float, float, float, float]:
        return (
            item.balanced_accuracy,
            min(item.direct_recall, item.partial_recall),
            item.partial_recall,
            item.threshold,
        )

    eligible = [item for item in candidates if item.eligible]
    return max(eligible or candidates, key=key)


def assert_source_isolated_validation(
    dataset_name: str,
    *,
    source_partition_frozen: bool,
    sealed_holdout_assigned: bool,
) -> None:
    """Block reused development data and post-result holdout construction."""
    if not dataset_name or dataset_name in RETIRED_EVALUATION_DATASETS:
        raise ValueError("MiniLM v3 requires a new source-isolated development set.")
    if not source_partition_frozen or not sealed_holdout_assigned:
        raise ValueError(
            "Development and sealed holdout source assignments must be frozen together."
        )
