"""Leakage-safe evaluation helpers for the human-only multi-task student."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import gc
import random
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evidence_grouped_evaluation import classification_metrics
from ml.evidence_multiclass import (
    SUPPORT_CLASSES,
    MulticlassEvidenceReranker,
    PairTextTfidfClassifier,
)
from ml.evidence_multitask_student import (
    CandidateCompleteDataset,
    MultiTaskMiniLM,
    multitask_loss,
    multitask_rejection_features,
    multitask_targets,
)
from ml.evidence_task_rejection import select_task_rejection_threshold


@dataclass(frozen=True)
class StudentHyperparameters:
    """Inner-selected MiniLM training settings."""

    learning_rate: float
    epochs: int


@dataclass
class CandidateOutputs:
    """Candidate-level outputs aligned to the serialized pair order."""

    support: np.ndarray
    strength: np.ndarray
    rank: np.ndarray

    @classmethod
    def empty(cls, pairs: int) -> CandidateOutputs:
        return cls(
            support=np.full(pairs, np.nan, dtype=np.float64),
            strength=np.full(pairs, np.nan, dtype=np.float64),
            rank=np.full(pairs, np.nan, dtype=np.float64),
        )

    def assign(self, indices: Sequence[int], values: CandidateOutputs) -> None:
        index = np.asarray(indices, dtype=np.int64)
        if any(len(array) != len(index) for array in (
            values.support,
            values.strength,
            values.rank,
        )):
            raise ValueError("Candidate output assignment is misaligned.")
        self.support[index] = values.support
        self.strength[index] = values.strength
        self.rank[index] = values.rank


def grouped_task_folds(
    tasks: Sequence[Mapping[str, Any]],
    *,
    group_field: str,
    n_splits: int,
    random_state: int,
) -> list[tuple[list[str], list[str]]]:
    """Split complete tasks without exposing one group across train and test."""
    if n_splits < 2:
        raise ValueError("Grouped evaluation needs at least two folds.")
    task_ids = np.asarray([str(task["task_id"]) for task in tasks])
    labels = np.asarray(
        [int(str(task["support_label"]) != "No Support") for task in tasks]
    )
    groups = np.asarray([str(task.get(group_field, "")) for task in tasks])
    if any(not group for group in groups):
        raise ValueError(f"Every task needs non-empty {group_field}.")
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    folds = []
    for train, test in splitter.split(task_ids, labels, groups):
        train_ids = task_ids[train].tolist()
        test_ids = task_ids[test].tolist()
        train_groups = set(groups[train].tolist())
        test_groups = set(groups[test].tolist())
        if train_groups & test_groups:
            raise ValueError("Grouped fold leakage detected.")
        folds.append((train_ids, test_ids))
    return folds


def pair_indices_for_tasks(
    task_ids: Sequence[str],
    validated: CandidateCompleteDataset,
) -> list[int]:
    """Flatten complete candidate rows for a task-id sequence."""
    return [
        index
        for task_id in task_ids
        for index in validated.task_pair_indices[task_id]
    ]


def _resolve_device(requested: str) -> str:
    import torch

    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was requested but is unavailable.")
    if requested == "auto":
        return "mps" if torch.backends.mps.is_available() else "cpu"
    return requested


def _seed_training(random_state: int) -> None:
    """Seed every local training RNG and require deterministic operations."""
    import torch

    random.seed(random_state)
    np.random.seed(random_state)
    torch.manual_seed(random_state)
    torch.use_deterministic_algorithms(True)


def _encoded(
    tokenizer: Any,
    pairs: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    *,
    max_length: int,
    device: str,
) -> dict[str, Any]:
    return {
        key: value.to(device)
        for key, value in tokenizer(
            [str(pairs[index]["requirement"]) for index in indices],
            [str(pairs[index]["evidence"]) for index in indices],
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).items()
    }


def _cleanup_model(model: Any, device: str) -> None:
    import torch

    del model
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()


def fit_predict_multitask(
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
    random_state: int,
    device: str,
) -> tuple[list[int], CandidateOutputs]:
    """Fit one human-only multi-task model and predict complete held-out tasks."""
    import torch
    from transformers import AutoTokenizer

    resolved_device = _resolve_device(device)
    _seed_training(random_state)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model = MultiTaskMiniLM(model_path).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=weight_decay,
    )
    all_support, all_strength = multitask_targets(pairs)
    train_pair_indices = pair_indices_for_tasks(train_task_ids, validated)
    positives = float(all_support[train_pair_indices].sum())
    negatives = float(len(train_pair_indices) - positives)
    if not positives or not negatives:
        raise ValueError("Support training fold must contain both classes.")
    pos_weight = torch.tensor(
        negatives / positives,
        dtype=torch.float32,
        device=resolved_device,
    )
    task_batch_size = max(1, batch_size // int(validated.manifest["pairs_per_task"]))
    generator = np.random.default_rng(random_state)
    task_order = np.asarray(list(train_task_ids), dtype=object)
    for _ in range(hyperparameters.epochs):
        generator.shuffle(task_order)
        model.train()
        for start in range(0, len(task_order), task_batch_size):
            batch_tasks = [str(value) for value in task_order[start : start + task_batch_size]]
            global_indices = pair_indices_for_tasks(batch_tasks, validated)
            global_to_local = {
                global_index: local_index
                for local_index, global_index in enumerate(global_indices)
            }
            local_groups = [
                tuple(global_to_local[index] for index in validated.task_pair_indices[task_id])
                for task_id in batch_tasks
            ]
            local_selected: list[int | None] = []
            for task_id in batch_tasks:
                selected_index = validated.selected_pair_index[task_id]
                local_selected.append(
                    None
                    if selected_index is None
                    else global_to_local[selected_index]
                )
            support_logits, strength_logits, rank_scores = model(
                **_encoded(
                    tokenizer,
                    pairs,
                    global_indices,
                    max_length=max_length,
                    device=resolved_device,
                )
            )
            loss, _ = multitask_loss(
                support_logits,
                strength_logits,
                rank_scores,
                torch.tensor(
                    all_support[global_indices],
                    dtype=torch.float32,
                    device=resolved_device,
                ),
                torch.tensor(
                    all_strength[global_indices],
                    dtype=torch.long,
                    device=resolved_device,
                ),
                local_groups,
                local_selected,
                support_pos_weight=pos_weight,
                support_weight=float(loss_weights["support"]),
                strength_weight=float(loss_weights["strength"]),
                ranking_weight=float(loss_weights["ranking"]),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
    prediction_indices = pair_indices_for_tasks(predict_task_ids, validated)
    support_rows: list[np.ndarray] = []
    strength_rows: list[np.ndarray] = []
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
                    device=resolved_device,
                )
            )
            support_rows.append(torch.sigmoid(support_logits).cpu().numpy())
            strength_rows.append(
                torch.softmax(strength_logits, dim=1)[:, 1].cpu().numpy()
            )
            rank_rows.append(rank_scores.cpu().numpy())
    outputs = CandidateOutputs(
        support=np.concatenate(support_rows).astype(np.float64),
        strength=np.concatenate(strength_rows).astype(np.float64),
        rank=np.concatenate(rank_rows).astype(np.float64),
    )
    _cleanup_model(model, resolved_device)
    return prediction_indices, outputs


def fit_predict_pair_classifier(
    kind: str,
    pairs: Sequence[Mapping[str, Any]],
    validated: CandidateCompleteDataset,
    train_task_ids: Sequence[str],
    predict_task_ids: Sequence[str],
    *,
    random_state: int,
) -> tuple[list[int], CandidateOutputs]:
    """Fit one transparent/LSA baseline and expose comparable three-head outputs."""
    train_indices = pair_indices_for_tasks(train_task_ids, validated)
    predict_indices = pair_indices_for_tasks(predict_task_ids, validated)
    if kind == "pair_text_tfidf":
        model: Any = PairTextTfidfClassifier(random_state=random_state)
    elif kind == "lsa_transparent_multiclass":
        model = MulticlassEvidenceReranker(random_state=random_state)
    else:
        raise ValueError(f"Unknown pair classifier: {kind}")
    model.fit(
        [str(pairs[index]["requirement"]) for index in train_indices],
        [str(pairs[index]["evidence"]) for index in train_indices],
        [str(pairs[index]["support_label"]) for index in train_indices],
    )
    probabilities = model.predict_class_proba(
        [str(pairs[index]["requirement"]) for index in predict_indices],
        [str(pairs[index]["evidence"]) for index in predict_indices],
    )
    direct = probabilities[:, SUPPORT_CLASSES.index("Direct")]
    partial = probabilities[:, SUPPORT_CLASSES.index("Partial")]
    no_support = probabilities[:, SUPPORT_CLASSES.index("No Support")]
    supported = direct + partial
    strength = np.divide(
        direct,
        supported,
        out=np.full_like(direct, 0.5),
        where=supported > 0,
    )
    return predict_indices, CandidateOutputs(
        support=1.0 - no_support,
        strength=strength,
        rank=direct + 0.5 * partial,
    )


def fit_predict_flat_minilm(
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
    random_state: int,
    device: str,
) -> tuple[list[int], CandidateOutputs]:
    """Fit the legacy flat three-class MiniLM on candidate-complete rows."""
    import torch
    from transformers import (
        AutoConfig,
        AutoModelForSequenceClassification,
        AutoTokenizer,
    )

    resolved_device = _resolve_device(device)
    _seed_training(random_state)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model_config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    model_config.num_labels = len(SUPPORT_CLASSES)
    model_config.id2label = {
        index: label for index, label in enumerate(SUPPORT_CLASSES)
    }
    model_config.label2id = {
        label: index for index, label in enumerate(SUPPORT_CLASSES)
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        config=model_config,
        ignore_mismatched_sizes=True,
        local_files_only=True,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=hyperparameters.learning_rate,
        weight_decay=weight_decay,
    )
    train_indices = np.asarray(
        pair_indices_for_tasks(train_task_ids, validated),
        dtype=np.int64,
    )
    label_to_id = {label: index for index, label in enumerate(SUPPORT_CLASSES)}
    labels = np.asarray(
        [label_to_id[str(pair["support_label"])] for pair in pairs],
        dtype=np.int64,
    )
    counts = Counter(int(labels[index]) for index in train_indices)
    class_weights = torch.tensor(
        [
            len(train_indices) / (len(SUPPORT_CLASSES) * counts[index])
            for index in range(len(SUPPORT_CLASSES))
        ],
        dtype=torch.float32,
        device=resolved_device,
    )
    generator = np.random.default_rng(random_state)
    for _ in range(hyperparameters.epochs):
        generator.shuffle(train_indices)
        model.train()
        for start in range(0, len(train_indices), batch_size):
            batch = train_indices[start : start + batch_size].tolist()
            logits = model(
                **_encoded(
                    tokenizer,
                    pairs,
                    batch,
                    max_length=max_length,
                    device=resolved_device,
                )
            ).logits
            loss = torch.nn.functional.cross_entropy(
                logits,
                torch.tensor(
                    labels[batch],
                    dtype=torch.long,
                    device=resolved_device,
                ),
                weight=class_weights,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()
    prediction_indices = pair_indices_for_tasks(predict_task_ids, validated)
    probability_rows: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(prediction_indices), batch_size):
            batch = prediction_indices[start : start + batch_size]
            logits = model(
                **_encoded(
                    tokenizer,
                    pairs,
                    batch,
                    max_length=max_length,
                    device=resolved_device,
                )
            ).logits
            probability_rows.append(torch.softmax(logits, dim=1).cpu().numpy())
    probabilities = np.vstack(probability_rows).astype(np.float64)
    direct = probabilities[:, SUPPORT_CLASSES.index("Direct")]
    partial = probabilities[:, SUPPORT_CLASSES.index("Partial")]
    no_support = probabilities[:, SUPPORT_CLASSES.index("No Support")]
    supported = direct + partial
    strength = np.divide(
        direct,
        supported,
        out=np.full_like(direct, 0.5),
        where=supported > 0,
    )
    outputs = CandidateOutputs(
        support=1.0 - no_support,
        strength=strength,
        rank=direct + 0.5 * partial,
    )
    _cleanup_model(model, resolved_device)
    return prediction_indices, outputs


def task_feature_matrix(
    task_ids: Sequence[str],
    validated: CandidateCompleteDataset,
    outputs: CandidateOutputs,
) -> np.ndarray:
    """Build one rejector feature row per complete task."""
    return np.vstack(
        [
            multitask_rejection_features(
                outputs.support[
                    list(validated.task_pair_indices[task_id])
                ].tolist(),
                outputs.strength[
                    list(validated.task_pair_indices[task_id])
                ].tolist(),
                outputs.rank[
                    list(validated.task_pair_indices[task_id])
                ].tolist(),
            )
            for task_id in task_ids
        ]
    )


def fit_rejector(
    features: np.ndarray,
    labels: Sequence[int],
    groups: Sequence[str],
    *,
    random_state: int,
    n_splits: int,
) -> tuple[Pipeline, float, np.ndarray]:
    """Cross-fit the rejector and select its threshold from grouped OOF scores."""
    if (
        features.ndim != 2
        or len(features) != len(labels)
        or len(features) != len(groups)
    ):
        raise ValueError("Rejector features, labels, and groups must align.")

    def new_pipeline(seed: int) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=seed,
                        solver="liblinear",
                    ),
                ),
            ]
        )

    y = np.asarray(labels, dtype=np.int64)
    group_values = np.asarray(groups)
    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state,
    )
    oof_probabilities = np.full(len(y), np.nan, dtype=np.float64)
    for fold, (train, validation) in enumerate(
        splitter.split(features, y, group_values)
    ):
        train_groups = set(group_values[train].tolist())
        validation_groups = set(group_values[validation].tolist())
        if train_groups & validation_groups:
            raise ValueError("Rejector group leakage detected.")
        fold_pipeline = new_pipeline(random_state + fold)
        fold_pipeline.fit(features[train], y[train])
        oof_probabilities[validation] = fold_pipeline.predict_proba(
            features[validation]
        )[:, 1]
    if not np.all(np.isfinite(oof_probabilities)):
        raise ValueError("Rejector OOF predictions do not cover every task.")
    threshold = select_task_rejection_threshold(
        y.tolist(),
        oof_probabilities.tolist(),
    )
    pipeline = new_pipeline(random_state)
    pipeline.fit(features, y)
    return pipeline, threshold, oof_probabilities


def evaluate_outputs(
    tasks_by_id: Mapping[str, Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    validated: CandidateCompleteDataset,
    task_ids: Sequence[str],
    outputs: CandidateOutputs,
    acceptance_probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, Any]:
    """Evaluate candidate labels, ranking, rejection, and role-family outcomes."""
    if len(task_ids) != len(acceptance_probabilities):
        raise ValueError("Task acceptance probabilities must align.")
    pair_indices = pair_indices_for_tasks(task_ids, validated)
    if any(
        not np.all(np.isfinite(values[pair_indices]))
        for values in (outputs.support, outputs.strength, outputs.rank)
    ):
        raise ValueError("Candidate outputs do not cover evaluated pairs.")
    candidate_predictions = [
        (
            "No Support"
            if outputs.support[index] < 0.5
            else "Direct" if outputs.strength[index] >= 0.5
            else "Partial"
        )
        for index in pair_indices
    ]
    candidate_labels = [str(pairs[index]["support_label"]) for index in pair_indices]
    supported_ranks: list[int] = []
    supported_successes: list[bool] = []
    rejection_successes: list[bool] = []
    outcomes: dict[str, bool] = {}
    failures: dict[str, str] = {}
    task_labels: list[str] = []
    task_predictions: list[str] = []
    slice_outcomes: dict[
        str,
        dict[str, dict[str, list[bool]]],
    ] = {
        "role_family": {},
        "construction_stratum": {},
    }
    for task_id, acceptance in zip(
        task_ids,
        acceptance_probabilities,
        strict=True,
    ):
        task = tasks_by_id[task_id]
        indices = list(validated.task_pair_indices[task_id])
        order = sorted(
            indices,
            key=lambda index: (
                -outputs.rank[index],
                str(pairs[index]["pair_id"]),
            ),
        )
        accepted = float(acceptance) >= threshold
        top_index = order[0]
        predicted_task_label = (
            "No Support"
            if not accepted
            else "Direct" if outputs.strength[top_index] >= 0.5 else "Partial"
        )
        task_labels.append(str(task["support_label"]))
        task_predictions.append(predicted_task_label)
        if str(task["support_label"]) == "No Support":
            passed = not accepted
            rejection_successes.append(passed)
            if not passed:
                failures[task_id] = "false_accept"
        else:
            selected = validated.selected_pair_index[task_id]
            if selected is None:
                raise ValueError("Supported task is missing selected evidence.")
            rank = order.index(selected) + 1
            supported_ranks.append(rank)
            passed = accepted and rank == 1
            supported_successes.append(passed)
            if not passed:
                failures[task_id] = "support_reject" if not accepted else "wrong_rank"
        outcomes[task_id] = passed
        support_key = (
            "supported"
            if str(task["support_label"]) != "No Support"
            else "no_support"
        )
        for field in slice_outcomes:
            value = str(task.get(field) or "not_available")
            bucket = slice_outcomes[field].setdefault(
                value,
                {"supported": [], "no_support": []},
            )
            bucket[support_key].append(passed)
    supported_rate = float(np.mean(supported_successes)) if supported_successes else 0.0
    rejection_rate = float(np.mean(rejection_successes)) if rejection_successes else 0.0
    return {
        "candidate_metrics": classification_metrics(
            candidate_labels,
            candidate_predictions,
        ),
        "retrieval": {
            "support_tasks": len(supported_ranks),
            "no_support_tasks": len(rejection_successes),
            "recall_at_1": float(np.mean([rank == 1 for rank in supported_ranks])),
            "recall_at_3": float(np.mean([rank <= 3 for rank in supported_ranks])),
            "mean_reciprocal_rank": float(
                np.mean([1.0 / rank for rank in supported_ranks])
            ),
            "supported_task_success_rate": supported_rate,
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": float(
                np.mean(list(outcomes.values()))
            ),
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2.0,
        },
        "task_label_metrics": classification_metrics(
            task_labels,
            task_predictions,
        ),
        "outcomes": outcomes,
        "failure_counts": dict(Counter(failures.values())),
        "slices": {
            field: {
                value: {
                    "support_tasks": len(bucket["supported"]),
                    "no_support_tasks": len(bucket["no_support"]),
                    "supported_task_success_rate": (
                        float(np.mean(bucket["supported"]))
                        if bucket["supported"]
                        else 0.0
                    ),
                    "no_support_rejection_rate": (
                        float(np.mean(bucket["no_support"]))
                        if bucket["no_support"]
                        else 0.0
                    ),
                    "task_balanced_accuracy": (
                        (
                            float(np.mean(bucket["supported"]))
                            + float(np.mean(bucket["no_support"]))
                        )
                        / 2.0
                        if bucket["supported"] and bucket["no_support"]
                        else float(
                            np.mean(
                                [
                                    *bucket["supported"],
                                    *bucket["no_support"],
                                ]
                            )
                        )
                    ),
                }
                for value, bucket in sorted(values.items())
            }
            for field, values in slice_outcomes.items()
        },
    }
