from __future__ import annotations

import pytest

from ml.candidate_judgments import (
    CandidateJudgmentError,
    legacy_candidate_labels,
    validate_complete_candidate_labels,
)


def _task() -> dict[str, object]:
    return {
        "candidates": [
            {"candidate_id": "a", "evidence": "Built production SQL pipelines."},
            {"candidate_id": "b", "evidence": "Prepared weekly project notes."},
            {"candidate_id": "c", "evidence": "Used spreadsheets for reporting."},
        ]
    }


def test_complete_labels_derive_task_decision_and_preserve_all_candidates() -> None:
    label, labels = validate_complete_candidate_labels(
        _task(),
        {"a": "Direct", "b": "No Support", "c": "Partial"},
        "a",
    )

    assert label == "Direct"
    assert labels == {"a": "Direct", "b": "No Support", "c": "Partial"}


def test_complete_labels_require_full_coverage_and_strongest_selection() -> None:
    with pytest.raises(CandidateJudgmentError, match="Every candidate"):
        validate_complete_candidate_labels(
            _task(),
            {"a": "Direct", "b": "No Support"},
            "a",
        )
    with pytest.raises(CandidateJudgmentError, match="Partial candidate"):
        validate_complete_candidate_labels(
            _task(),
            {"a": "Direct", "b": "No Support", "c": "Partial"},
            "c",
        )


def test_no_support_derives_none_and_legacy_positive_does_not_invent_negatives() -> None:
    label, _ = validate_complete_candidate_labels(
        _task(),
        {"a": "No Support", "b": "No Support", "c": "No Support"},
        None,
    )
    assert label == "No Support"

    defaults = legacy_candidate_labels(
        _task(),
        {
            "support_label": "Direct",
            "selected_candidate_id": "a",
        },
    )
    assert defaults == {
        "a": "Direct",
        "b": "Unlabeled",
        "c": "Unlabeled",
    }
