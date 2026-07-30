"""Tests for independent evidence ranking and support acceptance."""

from __future__ import annotations

import pytest

from ml.evidence_decision import (
    EvidenceDecisionError,
    SupportSignal,
    evaluate_ranked_support_gate,
    paired_stratified_bootstrap_delta,
    select_support_gate_threshold,
)


def _tasks() -> list[dict[str, object]]:
    return [
        {
            "task_id": "supported",
            "requirement": "Build pipelines",
            "support_label": "Direct",
            "selected_candidate_id": "a",
            "candidates": [
                {"candidate_id": "a", "evidence": "Built ETL pipelines"},
                {"candidate_id": "b", "evidence": "Made reports"},
            ],
        },
        {
            "task_id": "unsupported",
            "requirement": "Use Kubernetes",
            "support_label": "No Support",
            "selected_candidate_id": None,
            "candidates": [
                {"candidate_id": "c", "evidence": "Used spreadsheets"},
                {"candidate_id": "d", "evidence": "Wrote documentation"},
            ],
        },
    ]


def _signal(similarity: float, *, overlap: bool = True) -> SupportSignal:
    return {
        "similarity": similarity,
        "has_overlap": overlap,
        "numeric_constraint_supported": True,
        "compound_requirement_supported": True,
    }


def test_rank_and_gate_are_evaluated_as_independent_signals() -> None:
    result = evaluate_ranked_support_gate(
        _tasks(),
        {
            "supported": [0.9, 0.2],
            "unsupported": [0.8, 0.1],
        },
        {
            "supported": [_signal(0.6), _signal(0.2)],
            "unsupported": [_signal(0.1), _signal(0.0, overlap=False)],
        },
        support_threshold=0.4,
    )

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["no_support_rejection_rate"] == 1.0
    assert result["retrieval"]["task_decision_accuracy"] == 1.0
    assert result["accepted_task_count"] == 1
    assert result["decisions"][0]["top_is_gold"] is True
    assert result["decisions"][1]["top_is_gold"] is None


def test_gate_never_searches_past_the_rankers_top_candidate() -> None:
    result = evaluate_ranked_support_gate(
        _tasks()[:1],
        {"supported": [0.9, 0.2]},
        {"supported": [_signal(0.1), _signal(0.9)]},
        support_threshold=0.4,
    )

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["supported_task_success_rate"] == 0.0
    assert result["failures"][0]["failure"] == "gate_reject"


def test_threshold_selection_balances_support_and_rejection() -> None:
    threshold, result = select_support_gate_threshold(
        _tasks(),
        {
            "supported": [0.9, 0.2],
            "unsupported": [0.8, 0.1],
        },
        {
            "supported": [_signal(0.6), _signal(0.2)],
            "unsupported": [_signal(0.3), _signal(0.1)],
        },
    )

    assert 0.3 < threshold <= 0.6
    assert result["retrieval"]["task_balanced_accuracy"] == 1.0


def test_misaligned_signals_are_rejected() -> None:
    with pytest.raises(EvidenceDecisionError, match="wrong length"):
        evaluate_ranked_support_gate(
            _tasks()[:1],
            {"supported": [0.9, 0.2]},
            {"supported": [_signal(0.6)]},
            support_threshold=0.4,
        )


def test_paired_bootstrap_reports_uncertain_small_difference() -> None:
    result = paired_stratified_bootstrap_delta(
        [True, True, False, False],
        [True, False, True, False],
        [True, True, False, False],
        samples=500,
    )

    assert result["lower_90"] < 0
    assert result["upper_90"] > 0


def test_paired_bootstrap_rejects_missing_stratum() -> None:
    with pytest.raises(ValueError, match="support and No Support"):
        paired_stratified_bootstrap_delta(
            [True, True],
            [True, False],
            [True, True],
        )
