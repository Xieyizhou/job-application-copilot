"""Tests for leakage-safe frozen-holdout evaluation."""

from __future__ import annotations

import pytest

from ml.real_holdout_evaluation import (
    RealHoldoutEvaluationError,
    assert_holdout_isolated,
    evaluate_holdout_method,
    score_holdout_tasks,
    select_validation_threshold,
)


def _tasks() -> list[dict[str, object]]:
    return [
        {
            "task_id": "supported",
            "requirement": "Build pipelines",
            "support_label": "Direct",
            "selected_candidate_id": "a",
            "source_job_hash": "job-holdout",
            "source_resume_hash": "resume-holdout",
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
            "source_job_hash": "job-holdout-2",
            "source_resume_hash": "resume-holdout-2",
            "candidates": [
                {"candidate_id": "c", "evidence": "Used spreadsheets"},
                {"candidate_id": "d", "evidence": "Wrote documentation"},
            ],
        },
    ]


def test_evaluate_holdout_method_reports_retrieval_and_rejection() -> None:
    score_map = {
        "Built ETL pipelines": 0.9,
        "Made reports": 0.2,
        "Used spreadsheets": 0.1,
        "Wrote documentation": 0.15,
    }

    def score(_: list[str], evidence: list[str]) -> list[float]:
        return [score_map[item] for item in evidence]

    result = evaluate_holdout_method(_tasks(), score=score, threshold=0.5)

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["no_support_rejection_rate"] == 1.0
    assert result["retrieval"]["task_decision_accuracy"] == 1.0
    assert result["failure_count"] == 0


def test_isolation_rejects_content_overlap() -> None:
    with pytest.raises(RealHoldoutEvaluationError, match="content overlaps"):
        assert_holdout_isolated(
            _tasks(),
            [],
            [
                {
                    "requirement": "Build pipelines",
                    "evidence": "Built ETL pipelines",
                }
            ],
        )


def test_isolation_rejects_source_group_overlap() -> None:
    with pytest.raises(RealHoldoutEvaluationError, match="source_resume_hash"):
        assert_holdout_isolated(
            _tasks(),
            [{"source_resume_hash": "resume-holdout"}],
            [],
        )


def test_validation_threshold_balances_support_and_rejection() -> None:
    tasks = _tasks()
    score_map = {
        "Built ETL pipelines": 0.9,
        "Made reports": 0.3,
        "Used spreadsheets": 0.4,
        "Wrote documentation": 0.2,
    }

    def score(_: list[str], evidence: list[str]) -> list[float]:
        return [score_map[item] for item in evidence]

    scored = score_holdout_tasks(tasks, score=score)
    threshold, result = select_validation_threshold(tasks, scored)

    assert 0.4 < threshold <= 0.9
    assert result["retrieval"]["task_balanced_accuracy"] == 1.0
