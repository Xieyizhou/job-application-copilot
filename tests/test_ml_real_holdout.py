"""Tests for freezing independently reviewed real-text holdout tasks."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ml.real_holdout import (
    RealHoldoutError,
    finalize_real_holdout,
    finalize_real_reviews,
    jsonl_bytes,
)


def _task(task_id: str = "task-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "requirement": "Build data pipelines.",
        "role_family": "Data",
        "source_dataset": "real",
        "source_job_hash": f"job-{task_id}",
        "source_resume_hash": f"resume-{task_id}",
        "candidates": [
            {"candidate_id": "a", "evidence": "Built an ETL pipeline."},
            {"candidate_id": "b", "evidence": "Created a dashboard."},
        ],
    }


def _event(
    task_id: str,
    label: str,
    selected: str | None,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "action": "label",
        "support_label": label,
        "selected_candidate_id": selected,
    }


def test_finalize_merges_agreement_and_human_adjudication() -> None:
    agreed = _task("agreed")
    disputed = _task("disputed")
    a_events = [
        _event("agreed", "Direct", "a"),
        _event("disputed", "Partial", "a"),
    ]
    b_events = [
        _event("agreed", "Direct", "a"),
        _event("disputed", "No Support", None),
    ]
    gold, report = finalize_real_holdout(
        [agreed, disputed],
        [deepcopy(disputed), deepcopy(agreed)],
        a_events,
        b_events,
        [deepcopy(disputed)],
        [_event("disputed", "Direct", "b")],
    )

    by_id = {task["task_id"]: task for task in gold}
    assert by_id["agreed"]["decision_source"] == "human_model_agreement"
    assert by_id["disputed"]["decision_source"] == "human_adjudication"
    assert by_id["disputed"]["selected_candidate_id"] == "b"
    assert report["human_adjudications"] == 1
    assert jsonl_bytes(gold).endswith(b"\n")


def test_finalize_rejects_incomplete_adjudication_queue() -> None:
    task = _task()
    with pytest.raises(RealHoldoutError, match="exactly the A/B disagreements"):
        finalize_real_holdout(
            [task],
            [deepcopy(task)],
            [_event("task-1", "Direct", "a")],
            [_event("task-1", "No Support", None)],
            [],
            [],
        )


def test_finalize_rejects_reviewer_content_mismatch() -> None:
    a_task = _task()
    b_task = deepcopy(a_task)
    b_task["requirement"] = "Different requirement."
    with pytest.raises(RealHoldoutError, match="content differs"):
        finalize_real_holdout(
            [a_task],
            [b_task],
            [_event("task-1", "Direct", "a")],
            [_event("task-1", "Direct", "a")],
            [],
            [],
        )


def test_finalize_validation_marks_development_boundary() -> None:
    task = _task()
    event = _event("task-1", "Direct", "a")
    gold, report = finalize_real_reviews(
        [task],
        [deepcopy(task)],
        [event],
        [deepcopy(event)],
        [],
        [],
        dataset_role="validation",
    )

    assert gold[0]["record_type"] == "real_validation_gold_task"
    assert report["training_use"] == "threshold_and_model_development_only"


def test_finalize_reserve_marks_one_time_evaluation_boundary() -> None:
    task = _task()
    event = _event("task-1", "Direct", "a")
    gold, report = finalize_real_reviews(
        [task],
        [deepcopy(task)],
        [event],
        [deepcopy(event)],
        [],
        [],
        dataset_role="reserve",
    )

    assert gold[0]["record_type"] == "real_reserve_gold_task"
    assert report["training_use"] == "one_time_evaluation_only"


def test_finalize_development_marks_model_development_boundary() -> None:
    task = _task()
    event = _event("task-1", "Direct", "a")
    gold, report = finalize_real_reviews(
        [task],
        [deepcopy(task)],
        [event],
        [deepcopy(event)],
        [],
        [],
        dataset_role="development",
    )

    assert gold[0]["record_type"] == "real_development_gold_task"
    assert report["training_use"] == "threshold_and_model_development_only"


def test_finalize_training_supplement_allows_reviewed_training_only() -> None:
    task = _task()
    event = _event("task-1", "Direct", "a")
    gold, report = finalize_real_reviews(
        [task],
        [deepcopy(task)],
        [event],
        [deepcopy(event)],
        [],
        [],
        dataset_role="training_supplement",
    )

    assert gold[0]["record_type"] == "real_training_supplement_gold_task"
    assert report["training_use"] == "reviewed_training_allowed"


def test_finalize_allows_precommitted_agreement_audit_override() -> None:
    task = _task()
    reviewer_event = _event("task-1", "Direct", "a")
    gold, report = finalize_real_reviews(
        [task],
        [deepcopy(task)],
        [reviewer_event],
        [deepcopy(reviewer_event)],
        [deepcopy(task)],
        [_event("task-1", "Partial", "b")],
        dataset_role="reserve",
        agreement_audit_task_ids=["task-1"],
    )

    assert gold[0]["decision_source"] == "human_agreement_audit"
    assert gold[0]["support_label"] == "Partial"
    assert report["human_agreement_audits"] == 1
