"""Tests for operational model review and human-authority agreement."""

from __future__ import annotations

import pytest

from ml.operational_review import (
    model_candidate_decision,
    operational_review_agreement,
)


def _task() -> dict[str, object]:
    return {
        "task_id": "task-1",
        "candidates": [
            {"candidate_id": "a", "evidence": "First"},
            {"candidate_id": "b", "evidence": "Second"},
            {"candidate_id": "c", "evidence": "Third"},
            {"candidate_id": "d", "evidence": "Fourth"},
        ],
    }


def test_model_decision_selects_strongest_supported_candidate() -> None:
    decision = model_candidate_decision(
        _task(),
        ["Partial", "Direct", "Direct", "No Support"],
        [0.8, 0.6, 0.9, 0.1],
    )

    assert decision["support_label"] == "Direct"
    assert decision["selected_candidate_id"] == "c"
    assert decision["cover_letter_safe"] is False


def test_model_decision_rejects_misaligned_predictions() -> None:
    with pytest.raises(ValueError, match="align"):
        model_candidate_decision(
            _task(),
            ["No Support"],
            [0.1],
        )


def test_agreement_preserves_reviewer_a_as_gold_authority() -> None:
    task = _task()
    left = {
        "support_label": "Partial",
        "selected_candidate_id": "a",
        "candidate_labels": {
            "a": "Partial",
            "b": "No Support",
            "c": "No Support",
            "d": "No Support",
        },
    }
    right = {
        **left,
        "support_label": "No Support",
        "selected_candidate_id": None,
        "candidate_labels": {
            key: "No Support" for key in ("a", "b", "c", "d")
        },
    }

    report = operational_review_agreement(
        [task],
        {"task-1": left},
        {"task-1": right},
    )

    assert report["task_exact_agreements"] == 0
    assert report["candidate_label_agreements"] == 3
    assert report["gold_authority"] == "reviewer_a_human"
