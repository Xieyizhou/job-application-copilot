"""Regression tests for versioned development evaluation entry points."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "ml" / name
    spec = importlib.util.spec_from_file_location(f"test_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}.")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frozen_v3(base: Path) -> None:
    gold = b'{"task_id":"development-v3"}\n'
    (base / "real_development_gold_v3.jsonl").write_bytes(gold)
    (base / "real_development_frozen_manifest_v3.json").write_text(
        json.dumps({"gold_sha256": hashlib.sha256(gold).hexdigest()}),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("script_name", "loader_name"),
    [
        ("evaluate_two_stage_development.py", "_load_frozen_development"),
        ("evaluate_constraint_gate_development.py", "_load_development"),
    ],
)
def test_development_evaluator_loads_requested_v3_files(
    tmp_path: Path,
    script_name: str,
    loader_name: str,
) -> None:
    module = _load_script(script_name)
    _frozen_v3(tmp_path)

    loaded = getattr(module, loader_name)(tmp_path, "v3")
    tasks = loaded[0]

    assert tasks == [{"task_id": "development-v3"}]


@pytest.mark.parametrize(
    "script_name",
    [
        "evaluate_two_stage_development.py",
        "evaluate_constraint_gate_development.py",
    ],
)
def test_development_evaluator_accepts_v3_cli(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script(script_name)
    monkeypatch.setattr(
        sys,
        "argv",
        [script_name, "--dataset-version", "3"],
    )

    assert module.parse_args().dataset_version == 3


def test_candidate_complete_metrics_evaluate_ranking_and_rejection() -> None:
    module = _load_script("evaluate_candidate_complete_multiclass.py")
    tasks = [
        {
            "task_id": "supported",
            "support_label": "Direct",
            "selected_candidate_id": "best",
            "candidates": [
                {"candidate_id": "best", "evidence": "Built SQL pipelines."},
                {"candidate_id": "other", "evidence": "Used spreadsheets."},
            ],
        },
        {
            "task_id": "unsupported",
            "support_label": "No Support",
            "selected_candidate_id": None,
            "candidates": [
                {"candidate_id": "a", "evidence": "Prepared notes."},
                {"candidate_id": "b", "evidence": "Joined meetings."},
            ],
        },
    ]
    pairs = [
        {
            "pair_id": "p1",
            "task_id": "supported",
            "evidence": "Built SQL pipelines.",
        },
        {
            "pair_id": "p2",
            "task_id": "supported",
            "evidence": "Used spreadsheets.",
        },
        {
            "pair_id": "p3",
            "task_id": "unsupported",
            "evidence": "Prepared notes.",
        },
        {
            "pair_id": "p4",
            "task_id": "unsupported",
            "evidence": "Joined meetings.",
        },
    ]

    result = module._task_retrieval_metrics(
        tasks,
        pairs,
        ["Direct", "No Support", "No Support", "No Support"],
        [0.9, 0.1, 0.2, 0.1],
    )

    assert result["recall_at_1"] == 1.0
    assert result["supported_task_success_rate"] == 1.0
    assert result["no_support_rejection_rate"] == 1.0
    assert result["task_balanced_accuracy"] == 1.0


def test_real_embedding_evaluator_flattens_candidate_tasks() -> None:
    module = _load_script("evaluate_sentence_embedding_real_development.py")
    tasks = [
        {
            "task_id": "task-1",
            "requirement": "Use SQL",
            "candidates": [
                {"candidate_id": "a", "evidence": "Built SQL reports."},
                {"candidate_id": "b", "evidence": "Used spreadsheets."},
            ],
        }
    ]

    assert module._flatten_tasks(tasks) == [
        {
            "pair_id": "task-1:a",
            "task_id": "task-1",
            "requirement": "Use SQL",
            "evidence": "Built SQL reports.",
        },
        {
            "pair_id": "task-1:b",
            "task_id": "task-1",
            "requirement": "Use SQL",
            "evidence": "Used spreadsheets.",
        },
    ]


def test_task_rejector_inner_oof_keeps_resume_groups_out_of_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("evaluate_task_rejector_operational_v2.py")
    tasks = [
        {
            "task_id": f"task-{group}-{index}",
            "source_resume_hash": f"resume-{group}",
            "requirement": f"requirement-{group}-{index}",
            "support_label": "Direct" if group % 2 else "No Support",
            "selected_candidate_id": "a" if group % 2 else None,
            "candidates": [
                {
                    "candidate_id": candidate,
                    "evidence": f"evidence-{group}-{index}-{candidate}",
                    "support_label": (
                        "Direct"
                        if group % 2 and candidate == "a"
                        else "No Support"
                    ),
                }
                for candidate in ("a", "b")
            ],
        }
        for group in range(6)
        for index in range(2)
    ]

    class FakeModel:
        def __init__(self, training_requirements: set[str]) -> None:
            self.training_requirements = training_requirements

    def fake_fit(
        rows: list[tuple[str, str, str]],
        *,
        random_state: int,
    ) -> FakeModel:
        del random_state
        return FakeModel({row[0] for row in rows})

    def fake_predict(
        model: FakeModel,
        test_tasks: list[dict[str, object]],
    ) -> dict[str, list[list[float]]]:
        training_groups = {
            requirement.split("-")[1]
            for requirement in model.training_requirements
            if requirement.startswith("requirement-")
        }
        assert all(
            str(task["source_resume_hash"]).split("-")[1]
            not in training_groups
            for task in test_tasks
        )
        return {
            str(task["task_id"]): [[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]]
            for task in test_tasks
        }

    monkeypatch.setattr(module, "fit_candidate", fake_fit)
    monkeypatch.setattr(module, "predict_tasks", fake_predict)

    probabilities = module._inner_oof_candidate_probabilities(
        tasks,
        [("base requirement", "base evidence", "No Support")],
        random_state=7,
    )

    assert set(probabilities) == {
        str(task["task_id"]) for task in tasks
    }
