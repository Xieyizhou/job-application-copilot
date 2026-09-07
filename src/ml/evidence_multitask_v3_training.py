"""Isolated offline trainer for the frozen MiniLM v3 objective."""

from __future__ import annotations

from ml.evidence_multitask_evaluation import _encoded

from ml.evidence_multitask_evaluation import _seed_training

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import math
from typing import Any

import numpy as np

from ml.evidence_multitask_evaluation import CandidateOutputs, StudentHyperparameters
from ml.evidence_multitask_student import (
    CandidateCompleteDataset,
    MultiTaskMiniLM,
    multitask_targets,
)
from ml.evidence_multitask_v3 import (
    coverage_aware_task_batches,
    coverage_logits,
    multitask_v3_loss,
)


@dataclass(frozen=True)
class V3TrainingManifest:
    """Text-free training diagnostics for one fit/predict call."""

    epochs: int
    batches: int
    batches_with_coverage_pairs: int
    training_tasks: int
    prediction_tasks: int
    device: str
    teacher_scores_used: bool = False
    artifact_created: bool = False
    product_integration_allowed: bool = False


def _pair_indices(
    task_ids: Sequence[str],
    validated: CandidateCompleteDataset,
) -> list[int]:
    return [
        pair_index for task_id in task_ids for pair_index in validated.task_pair_indices[task_id]
    ]


def _validate_task_boundary(
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    train_task_ids: Sequence[str],
    predict_task_ids: Sequence[str],
) -> None:
    train = set(train_task_ids)
    predict = set(predict_task_ids)
    if (
        not train
        or not predict
        or len(train) != len(train_task_ids)
        or len(predict) != len(predict_task_ids)
        or train & predict
        or not (train | predict).issubset(tasks_by_id)
    ):
        raise ValueError("V3 training and prediction task IDs are invalid.")
    train_groups = {str(tasks_by_id[task_id].get("evaluation_group", "")) for task_id in train}
    predict_groups = {str(tasks_by_id[task_id].get("evaluation_group", "")) for task_id in predict}
    if "" in train_groups or "" in predict_groups or train_groups & predict_groups:
        raise ValueError("V3 training and prediction groups must be non-empty and disjoint.")


def fit_predict_multitask_v3(
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    validated: CandidateCompleteDataset,
    train_task_ids: Sequence[str],
    predict_task_ids: Sequence[str],
    *,
    model_path: str,
    hyperparameters: StudentHyperparameters,
    batch_size: int,
    max_length: int,
    weight_decay: float,
    gradient_clip_norm: float,
    loss_weights: Mapping[str, float],
    coverage_margin: float,
    coverage_pointwise_weight: float,
    random_state: int,
    device: str = "cpu",
) -> tuple[list[int], CandidateOutputs, V3TrainingManifest]:
    """Fit the frozen v3 objective and return raw held-out candidate outputs."""
    import torch
    from transformers import AutoTokenizer

    _validate_task_boundary(tasks_by_id, train_task_ids, predict_task_ids)
    if device != "cpu":
        raise ValueError("The deterministic MiniLM v3 training contract requires CPU.")
    if (
        batch_size < int(validated.manifest["pairs_per_task"]) * 2
        or max_length < 1
        or hyperparameters.epochs < 1
        or not math.isfinite(hyperparameters.learning_rate)
        or hyperparameters.learning_rate <= 0.0
        or not math.isfinite(weight_decay)
        or weight_decay < 0.0
        or not math.isfinite(gradient_clip_norm)
        or gradient_clip_norm <= 0.0
        or set(loss_weights) != {"support", "coverage", "ranking"}
    ):
        raise ValueError("MiniLM v3 training settings are invalid.")
    _seed_training(random_state)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MultiTaskMiniLM(model_path).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=weight_decay,
    )
    all_support, all_strength = multitask_targets(pairs)
    task_batch_size = max(
        2,
        batch_size // int(validated.manifest["pairs_per_task"]),
    )
    task_labels = [str(tasks_by_id[task_id]["support_label"]) for task_id in train_task_ids]
    batches_seen = 0
    coverage_batches = 0
    for epoch in range(hyperparameters.epochs):
        task_batches = coverage_aware_task_batches(
            train_task_ids,
            task_labels,
            task_batch_size=task_batch_size,
            random_state=random_state + epoch,
        )
        model.train()
        for batch_tasks in task_batches:
            global_indices = _pair_indices(batch_tasks, validated)
            global_to_local = {
                global_index: local_index for local_index, global_index in enumerate(global_indices)
            }
            task_groups = [
                tuple(global_to_local[index] for index in validated.task_pair_indices[task_id])
                for task_id in batch_tasks
            ]
            selected_indices: list[int | None] = []
            for task_id in batch_tasks:
                selected = validated.selected_pair_index[task_id]
                selected_indices.append(None if selected is None else global_to_local[selected])
            support_logits, strength_logits, rank_scores = model(
                **_encoded(
                    tokenizer,
                    pairs,
                    global_indices,
                    max_length=max_length,
                    device=device,
                )
            )
            loss, diagnostics = multitask_v3_loss(
                support_logits,
                strength_logits,
                rank_scores,
                torch.tensor(
                    all_support[global_indices],
                    dtype=torch.float32,
                    device=device,
                ),
                torch.tensor(
                    all_strength[global_indices],
                    dtype=torch.long,
                    device=device,
                ),
                task_groups,
                selected_indices,
                coverage_margin=coverage_margin,
                coverage_pointwise_weight=coverage_pointwise_weight,
                support_weight=float(loss_weights["support"]),
                coverage_weight=float(loss_weights["coverage"]),
                ranking_weight=float(loss_weights["ranking"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
            batches_seen += 1
            coverage_batches += int(diagnostics["coverage_pairs"] > 0.0)

    prediction_indices = _pair_indices(predict_task_ids, validated)
    support_rows: list[np.ndarray] = []
    coverage_rows: list[np.ndarray] = []
    rank_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prediction_indices), batch_size):
            batch = prediction_indices[start : start + batch_size]
            support_logits, strength_logits, rank_scores = model(
                **_encoded(
                    tokenizer,
                    pairs,
                    batch,
                    max_length=max_length,
                    device=device,
                )
            )
            support_rows.append(torch.sigmoid(support_logits).cpu().numpy())
            coverage_rows.append(torch.sigmoid(coverage_logits(strength_logits)).cpu().numpy())
            rank_rows.append(rank_scores.cpu().numpy())
    outputs = CandidateOutputs(
        support=np.concatenate(support_rows).astype(np.float64),
        strength=np.concatenate(coverage_rows).astype(np.float64),
        rank=np.concatenate(rank_rows).astype(np.float64),
    )
    del model
    gc.collect()
    return (
        prediction_indices,
        outputs,
        V3TrainingManifest(
            epochs=hyperparameters.epochs,
            batches=batches_seen,
            batches_with_coverage_pairs=coverage_batches,
            training_tasks=len(train_task_ids),
            prediction_tasks=len(predict_task_ids),
            device=device,
        ),
    )
