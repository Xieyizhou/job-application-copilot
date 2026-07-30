"""Tests for the precommitted sentence-embedding reserve evaluator."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "ml"
    / "evaluate_sentence_embedding_real_reserve.py"
)


def _load_script() -> Any:
    spec = importlib.util.spec_from_file_location("reserve_v4_script", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_gate_requires_all_precommitted_criteria() -> None:
    module = _load_script()
    baseline = {
        "task_balanced_accuracy": 0.50,
        "recall_at_1": 0.70,
        "no_support_rejection_rate": 0.50,
    }
    candidate = {
        "task_balanced_accuracy": 0.65,
        "recall_at_1": 0.70,
        "no_support_rejection_rate": 0.75,
    }

    passed = module._selection_gate(
        baseline,
        candidate,
        {"lower_90": 0.01},
    )
    failed = module._selection_gate(
        baseline,
        candidate,
        {"lower_90": 0.0},
    )

    assert passed["passed"] is True
    assert failed["passed"] is False


def test_failure_review_separates_improvements_and_regressions() -> None:
    module = _load_script()
    tasks = [
        {
            "task_id": "supported",
            "role_family": "Data",
            "support_label": "Direct",
            "selected_candidate_id": "a",
            "candidates": [
                {"candidate_id": "a", "evidence": "Built pipelines."},
                {"candidate_id": "b", "evidence": "Made slides."},
            ],
        },
        {
            "task_id": "unsupported",
            "role_family": "Business",
            "support_label": "No Support",
            "selected_candidate_id": None,
            "candidates": [
                {"candidate_id": "a", "evidence": "Unrelated."},
                {"candidate_id": "b", "evidence": "Also unrelated."},
            ],
        },
    ]
    pairs = [
        {
            "pair_id": f"{task['task_id']}:{candidate['candidate_id']}",
            "task_id": task["task_id"],
            "evidence": candidate["evidence"],
        }
        for task in tasks
        for candidate in task["candidates"]
    ]

    review = module._failure_review(
        tasks,
        pairs,
        ["No Support", "No Support", "Partial", "No Support"],
        [0.8, 0.1, 0.7, 0.2],
        {"supported": True, "unsupported": False},
        {"supported": False, "unsupported": False},
    )

    assert review["failure_type_counts"] == {
        "false_reject": 1,
        "false_accept": 1,
    }
    assert review["paired_outcome_counts"] == {
        "candidate_regressed": 1,
        "both_fail": 1,
    }
