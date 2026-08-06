"""Synthetic-only tests for the isolated MiniLM v3 trainer."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest
import torch

from ml.evidence_multitask_evaluation import StudentHyperparameters
from ml.evidence_multitask_student import validate_candidate_complete_dataset
from ml import evidence_multitask_v3_training as v3_training


def _task(
    task_id: str,
    group: str,
    label: str,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    candidate_labels = {
        "Direct": ["Direct", "Partial", "No Support", "No Support"],
        "Partial": ["Partial", "No Support", "No Support", "No Support"],
        "No Support": ["No Support"] * 4,
    }[label]
    candidates = [
        {
            "candidate_id": f"{task_id}-candidate-{index}",
            "evidence": f"synthetic evidence {task_id} {index}",
            "support_label": candidate_label,
        }
        for index, candidate_label in enumerate(candidate_labels)
    ]
    task = {
        "task_id": task_id,
        "evaluation_group": group,
        "role_family": "Synthetic",
        "support_label": label,
        "selected_candidate_id": (candidates[0]["candidate_id"] if label != "No Support" else None),
        "candidates": candidates,
    }
    pairs = [
        {
            "pair_id": f"{task_id}-pair-{index}",
            "task_id": task_id,
            "evaluation_group": group,
            "role_family": "Synthetic",
            "requirement": f"synthetic requirement {task_id}",
            "evidence": candidate["evidence"],
            "support_label": candidate["support_label"],
        }
        for index, candidate in enumerate(candidates)
    ]
    return task, pairs


def _dataset() -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
]:
    tasks: list[dict[str, object]] = []
    pairs: list[dict[str, object]] = []
    for task_id, group, label in (
        ("direct", "train-direct", "Direct"),
        ("partial", "train-partial", "Partial"),
        ("rejected", "train-rejected", "No Support"),
        ("prediction", "prediction-group", "Direct"),
    ):
        task, task_pairs = _task(task_id, group, label)
        tasks.append(task)
        pairs.extend(task_pairs)
    return tasks, pairs


def _install_fake_model(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeTokenizer:
        def __call__(
            self,
            requirements: list[str],
            evidence: list[str],
            **_: object,
        ) -> dict[str, torch.Tensor]:
            assert len(requirements) == len(evidence)
            return {"input_ids": torch.ones((len(requirements), 2), dtype=torch.long)}

    class FakeAutoTokenizer:
        @classmethod
        def from_pretrained(
            cls,
            _model_path: str,
            *,
            local_files_only: bool,
        ) -> FakeTokenizer:
            assert local_files_only is True
            return FakeTokenizer()

    class FakeModel(torch.nn.Module):
        def __init__(self, _model_path: str) -> None:
            super().__init__()
            self.support_bias = torch.nn.Parameter(torch.tensor(0.1))
            self.coverage_bias = torch.nn.Parameter(torch.tensor(0.0))
            self.rank_bias = torch.nn.Parameter(torch.tensor(0.0))

        def forward(
            self,
            input_ids: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            rows = input_ids.shape[0]
            support = self.support_bias.expand(rows)
            coverage = self.coverage_bias.expand(rows)
            strength = torch.stack((-coverage, coverage), dim=1)
            rank = self.rank_bias + torch.arange(rows, dtype=torch.float32)
            return support, strength, rank

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoTokenizer = FakeAutoTokenizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setattr(v3_training, "MultiTaskMiniLM", FakeModel)


def test_v3_trainer_uses_raw_outputs_and_is_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, pairs = _dataset()
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    validated = validate_candidate_complete_dataset(
        tasks,
        pairs,
        expected_tasks=4,
        expected_pairs=16,
        pairs_per_task=4,
    )
    _install_fake_model(monkeypatch)
    arguments = {
        "model_path": "synthetic-local-model",
        "hyperparameters": StudentHyperparameters(learning_rate=0.01, epochs=1),
        "batch_size": 12,
        "max_length": 16,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "loss_weights": {"support": 1.0, "coverage": 0.5, "ranking": 0.5},
        "coverage_margin": 0.5,
        "coverage_pointwise_weight": 0.25,
        "random_state": 17,
        "device": "cpu",
    }

    first = v3_training.fit_predict_multitask_v3(
        tasks_by_id,
        pairs,
        validated,
        ["direct", "partial", "rejected"],
        ["prediction"],
        **arguments,
    )
    second = v3_training.fit_predict_multitask_v3(
        tasks_by_id,
        pairs,
        validated,
        ["direct", "partial", "rejected"],
        ["prediction"],
        **arguments,
    )

    assert first[0] == [12, 13, 14, 15]
    assert np.array_equal(first[1].support, second[1].support)
    assert np.array_equal(first[1].strength, second[1].strength)
    assert np.array_equal(first[1].rank, second[1].rank)
    assert np.all((first[1].support >= 0.0) & (first[1].support <= 1.0))
    assert np.all((first[1].strength >= 0.0) & (first[1].strength <= 1.0))
    assert first[2].batches_with_coverage_pairs == 1
    assert first[2].teacher_scores_used is False
    assert first[2].artifact_created is False
    assert first[2].product_integration_allowed is False


def test_v3_trainer_rejects_group_overlap_and_non_cpu_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks, pairs = _dataset()
    tasks_by_id = {str(task["task_id"]): task for task in tasks}
    validated = validate_candidate_complete_dataset(
        tasks,
        pairs,
        expected_tasks=4,
        expected_pairs=16,
        pairs_per_task=4,
    )
    _install_fake_model(monkeypatch)
    tasks_by_id["prediction"]["evaluation_group"] = "train-direct"
    common = {
        "model_path": "synthetic-local-model",
        "hyperparameters": StudentHyperparameters(learning_rate=0.01, epochs=1),
        "batch_size": 12,
        "max_length": 16,
        "weight_decay": 0.01,
        "gradient_clip_norm": 1.0,
        "loss_weights": {"support": 1.0, "coverage": 0.5, "ranking": 0.5},
        "coverage_margin": 0.5,
        "coverage_pointwise_weight": 0.25,
        "random_state": 17,
    }
    with pytest.raises(ValueError, match="groups"):
        v3_training.fit_predict_multitask_v3(
            tasks_by_id,
            pairs,
            validated,
            ["direct", "partial", "rejected"],
            ["prediction"],
            device="cpu",
            **common,
        )
    tasks_by_id["prediction"]["evaluation_group"] = "prediction-group"
    with pytest.raises(ValueError, match="requires CPU"):
        v3_training.fit_predict_multitask_v3(
            tasks_by_id,
            pairs,
            validated,
            ["direct", "partial", "rejected"],
            ["prediction"],
            device="mps",
            **common,
        )
