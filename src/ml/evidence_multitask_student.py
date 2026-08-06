"""Candidate-complete contracts and multi-head MiniLM evidence student."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch

from ml.evidence_multiclass import SUPPORT_CLASSES


MULTITASK_REJECTION_FEATURE_NAMES = (
    "support_top",
    "support_second",
    "support_margin",
    "support_mean",
    "support_standard_deviation",
    "support_count_at_0_50",
    "support_count_at_0_70",
    "rank_top",
    "rank_margin",
    "rank_standard_deviation",
    "top_rank_strength_probability",
)


@dataclass(frozen=True)
class CandidateCompleteDataset:
    """Validated task-to-pair alignment without retaining new source copies."""

    task_pair_indices: dict[str, tuple[int, ...]]
    selected_pair_index: dict[str, int | None]
    manifest: dict[str, Any]


def validate_candidate_complete_dataset(
    tasks: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    *,
    expected_tasks: int,
    expected_pairs: int,
    pairs_per_task: int,
) -> CandidateCompleteDataset:
    """Fail closed unless every task and candidate judgment aligns exactly."""
    if len(tasks) != expected_tasks or len(pairs) != expected_pairs:
        raise ValueError("Candidate-complete dataset has an unexpected size.")
    if len(tasks) * pairs_per_task != len(pairs):
        raise ValueError("Candidate-complete task and pair counts do not align.")
    pair_ids = [str(pair.get("pair_id", "")) for pair in pairs]
    task_ids = [str(task.get("task_id", "")) for task in tasks]
    if not all(pair_ids) or len(pair_ids) != len(set(pair_ids)):
        raise ValueError("Candidate-complete pair IDs must be unique and non-empty.")
    if not all(task_ids) or len(task_ids) != len(set(task_ids)):
        raise ValueError("Candidate-complete task IDs must be unique and non-empty.")
    pairs_by_task: dict[str, list[int]] = defaultdict(list)
    for index, pair in enumerate(pairs):
        label = str(pair.get("support_label", ""))
        if label not in SUPPORT_CLASSES:
            raise ValueError("Every candidate pair needs an explicit support label.")
        pairs_by_task[str(pair.get("task_id", ""))].append(index)
    task_pair_indices: dict[str, tuple[int, ...]] = {}
    selected_pair_index: dict[str, int | None] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        indices = pairs_by_task.get(task_id, [])
        candidates = list(task.get("candidates", []))
        if len(indices) != pairs_per_task or len(candidates) != pairs_per_task:
            raise ValueError(f"Task {task_id} is not candidate-complete.")
        evidence_to_index: dict[str, int] = {}
        for index in indices:
            evidence = str(pairs[index].get("evidence", ""))
            if not evidence or evidence in evidence_to_index:
                raise ValueError(f"Task {task_id} has duplicate or empty pair evidence.")
            evidence_to_index[evidence] = index
        ordered: list[int] = []
        candidate_ids: set[str] = set()
        for candidate in candidates:
            candidate_id = str(candidate.get("candidate_id", ""))
            evidence = str(candidate.get("evidence", ""))
            label = str(candidate.get("support_label", ""))
            if (
                not candidate_id
                or candidate_id in candidate_ids
                or evidence not in evidence_to_index
            ):
                raise ValueError(f"Task {task_id} has invalid candidate metadata.")
            candidate_ids.add(candidate_id)
            pair_index = evidence_to_index[evidence]
            if label and (
                label not in SUPPORT_CLASSES
                or str(pairs[pair_index]["support_label"]) != label
            ):
                raise ValueError(f"Task {task_id} candidate and pair labels disagree.")
            if (
                str(pairs[pair_index].get("evaluation_group", ""))
                != str(task.get("evaluation_group", ""))
                or str(pairs[pair_index].get("role_family", ""))
                != str(task.get("role_family", ""))
            ):
                raise ValueError(f"Task {task_id} group metadata is inconsistent.")
            ordered.append(pair_index)
        task_label = str(task.get("support_label", ""))
        selected_id = task.get("selected_candidate_id")
        if task_label == "No Support":
            if selected_id not in (None, ""):
                raise ValueError("No Support tasks cannot select strongest evidence.")
            if any(
                str(pairs[index]["support_label"]) != "No Support"
                for index in ordered
            ):
                raise ValueError("No Support task contains a supported candidate.")
            selected_pair_index[task_id] = None
        else:
            matches = [
                ordered[index]
                for index, candidate in enumerate(candidates)
                if str(candidate["candidate_id"]) == str(selected_id)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Supported task {task_id} needs one selected candidate."
                )
            if str(pairs[matches[0]]["support_label"]) == "No Support":
                raise ValueError("Selected strongest evidence cannot be No Support.")
            if str(pairs[matches[0]]["support_label"]) != task_label:
                raise ValueError(
                    "Selected strongest evidence and task labels must agree."
                )
            selected_pair_index[task_id] = matches[0]
        task_pair_indices[task_id] = tuple(ordered)
    if set(pairs_by_task) != set(task_pair_indices):
        raise ValueError("Candidate pairs reference unknown tasks.")
    manifest = {
        "schema_version": 1,
        "candidate_complete": True,
        "tasks": len(tasks),
        "pairs": len(pairs),
        "pairs_per_task": pairs_per_task,
        "task_label_counts": dict(
            Counter(str(task["support_label"]) for task in tasks)
        ),
        "pair_label_counts": dict(
            Counter(str(pair["support_label"]) for pair in pairs)
        ),
        "evaluation_groups": len(
            {
                str(
                    task.get("evaluation_group")
                    or task.get("source_resume_hash")
                    or task["task_id"]
                )
                for task in tasks
            }
        ),
        "role_families": dict(
            Counter(str(task["role_family"]) for task in tasks)
        ),
        "teacher_scores_used": False,
        "product_integration_allowed": False,
    }
    return CandidateCompleteDataset(
        task_pair_indices=task_pair_indices,
        selected_pair_index=selected_pair_index,
        manifest=manifest,
    )


def multitask_targets(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """Return binary support and masked Direct/Partial targets."""
    support = np.asarray(
        [int(str(pair["support_label"]) != "No Support") for pair in pairs],
        dtype=np.float32,
    )
    strength = np.asarray(
        [
            1 if str(pair["support_label"]) == "Direct"
            else 0 if str(pair["support_label"]) == "Partial"
            else -100
            for pair in pairs
        ],
        dtype=np.int64,
    )
    return support, strength


def multitask_loss(
    support_logits: Any,
    strength_logits: Any,
    rank_scores: Any,
    support_targets: Any,
    strength_targets: Any,
    task_pair_indices: Sequence[Sequence[int]],
    selected_pair_indices: Sequence[int | None],
    *,
    support_pos_weight: Any,
    support_weight: float = 1.0,
    strength_weight: float = 0.5,
    ranking_weight: float = 0.5,
) -> tuple[Any, dict[str, float]]:
    """Compute masked human-only support, strength, and ranking objectives."""
    import torch.nn.functional as functional

    count = int(support_logits.shape[0])
    if (
        support_logits.ndim != 1
        or rank_scores.shape != support_logits.shape
        or strength_logits.shape != (count, 2)
        or support_targets.shape != support_logits.shape
        or strength_targets.shape != support_logits.shape
    ):
        raise ValueError("Multi-task tensors are misaligned.")
    if any(weight < 0 for weight in (support_weight, strength_weight, ranking_weight)):
        raise ValueError("Multi-task loss weights cannot be negative.")
    support_loss = functional.binary_cross_entropy_with_logits(
        support_logits,
        support_targets,
        pos_weight=support_pos_weight,
    )
    strength_mask = strength_targets != -100
    if bool(strength_mask.any()):
        strength_loss = functional.cross_entropy(
            strength_logits[strength_mask],
            strength_targets[strength_mask],
        )
    else:
        strength_loss = support_logits.new_zeros(())
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
        torch.stack(ranking_terms).mean()
        if ranking_terms
        else support_logits.new_zeros(())
    )
    total = (
        support_weight * support_loss
        + strength_weight * strength_loss
        + ranking_weight * ranking_loss
    )
    if not bool(torch.isfinite(total)):
        raise ValueError("Multi-task loss is not finite.")
    return total, {
        "support_loss": float(support_loss.detach().cpu()),
        "strength_loss": float(strength_loss.detach().cpu()),
        "ranking_loss": float(ranking_loss.detach().cpu()),
        "strength_rows": float(strength_mask.sum().item()),
        "ranking_pairs": float(len(ranking_terms)),
    }


def multitask_rejection_features(
    support_probabilities: Sequence[float],
    strength_probabilities: Sequence[float],
    rank_scores: Sequence[float],
) -> np.ndarray:
    """Return auditable task-distribution features for the OOF rejector."""
    support = np.asarray(support_probabilities, dtype=np.float64)
    strength = np.asarray(strength_probabilities, dtype=np.float64)
    ranks = np.asarray(rank_scores, dtype=np.float64)
    if (
        support.ndim != 1
        or len(support) < 2
        or support.shape != strength.shape
        or support.shape != ranks.shape
        or not np.all(np.isfinite(np.column_stack([support, strength, ranks])))
    ):
        raise ValueError("Multi-task rejection inputs are invalid.")
    support_order = np.sort(support)[::-1]
    rank_order = np.sort(ranks)[::-1]
    top_rank = int(np.argmax(ranks))
    features = np.asarray(
        [
            support_order[0],
            support_order[1],
            support_order[0] - support_order[1],
            float(np.mean(support)),
            float(np.std(support)),
            float(np.sum(support >= 0.50)),
            float(np.sum(support >= 0.70)),
            rank_order[0],
            rank_order[0] - rank_order[1],
            float(np.std(ranks)),
            strength[top_rank],
        ],
        dtype=np.float64,
    )
    if len(features) != len(MULTITASK_REJECTION_FEATURE_NAMES):
        raise ValueError("Multi-task rejection feature manifest drifted.")
    return features


class MultiTaskMiniLM(torch.nn.Module):
    """Shared MiniLM encoder with support, strength, and ranking heads."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoConfig, AutoModel

        super().__init__()
        config = AutoConfig.from_pretrained(model_path, local_files_only=True)
        self.encoder = AutoModel.from_pretrained(
            model_path,
            config=config,
            local_files_only=True,
        )
        hidden = int(config.hidden_size)
        self.support_head = torch.nn.Linear(hidden, 1)
        self.strength_head = torch.nn.Linear(hidden, 2)
        self.ranking_head = torch.nn.Linear(hidden, 1)

    def forward(self, **inputs: Any) -> tuple[Any, Any, Any]:
        outputs = self.encoder(**inputs)
        pooled = outputs.last_hidden_state[:, 0]
        return (
            self.support_head(pooled).squeeze(-1),
            self.strength_head(pooled),
            self.ranking_head(pooled).squeeze(-1),
        )


def finite_probability_vector(values: Sequence[float]) -> None:
    """Validate persisted probability outputs without retaining source text."""
    if not values or any(
        not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0
        for value in values
    ):
        raise ValueError("Probability outputs must be finite and bounded.")
