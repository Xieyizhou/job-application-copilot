"""Tests for independently shuffled real-text reviewer packets."""

from __future__ import annotations

from copy import deepcopy

import pytest

from ml.real_review import (
    build_adjudication_queue,
    build_adjudication_queue_with_audit,
    reviewer_packet_manifest,
    shuffled_reviewer_queue,
)


def _tasks() -> list[dict[str, object]]:
    return [
        {
            "schema_version": 1,
            "task_id": f"task-{index}",
            "requirement": f"Requirement {index}",
            "source_job_hash": f"job-{index}",
            "source_resume_hash": f"resume-{index}",
            "candidates": [
                {"candidate_id": f"{index}-a", "evidence": "Evidence A"},
                {"candidate_id": f"{index}-b", "evidence": "Evidence B"},
                {"candidate_id": f"{index}-c", "evidence": "Evidence C"},
            ],
        }
        for index in range(4)
    ]


def test_reviewer_queues_preserve_content_with_independent_order() -> None:
    source = _tasks()
    queue_a = shuffled_reviewer_queue(source, reviewer_id="a")
    queue_b = shuffled_reviewer_queue(source, reviewer_id="b")
    manifest = reviewer_packet_manifest(source, {"a": queue_a, "b": queue_b})

    assert manifest["content_identity_verified"] is True
    assert manifest["reviewers"]["a"]["content_sha256"] == (
        manifest["reviewers"]["b"]["content_sha256"]
    )
    assert manifest["reviewers"]["a"]["ordered_packet_sha256"] != (
        manifest["reviewers"]["b"]["ordered_packet_sha256"]
    )
    assert source == _tasks()


def test_packet_manifest_rejects_changed_candidate_text() -> None:
    source = _tasks()
    changed = deepcopy(source)
    changed[0]["candidates"][0]["evidence"] = "Changed"
    with pytest.raises(ValueError, match="queue content changed"):
        reviewer_packet_manifest(source, {"changed": changed})


def test_adjudication_queue_contains_only_exact_disagreements() -> None:
    tasks = _tasks()
    a_events = [
        {
            "task_id": task["task_id"],
            "action": "label",
            "support_label": "Direct",
            "selected_candidate_id": task["candidates"][0]["candidate_id"],
        }
        for task in tasks
    ]
    b_events = deepcopy(a_events)
    b_events[2]["support_label"] = "No Support"
    b_events[2]["selected_candidate_id"] = None

    queue, report = build_adjudication_queue(tasks, a_events, b_events)

    assert [task["task_id"] for task in queue] == ["task-2"]
    assert report["exact_agreements"] == 3
    assert report["adjudication_tasks"] == 1


def test_adjudication_queue_requires_complete_reviews() -> None:
    with pytest.raises(ValueError, match="complete every task"):
        build_adjudication_queue(_tasks(), [], [])


def test_agreement_audit_is_deterministic_and_excludes_disagreements() -> None:
    tasks = _tasks()
    a_events = [
        {
            "task_id": task["task_id"],
            "action": "label",
            "support_label": "Direct",
            "selected_candidate_id": task["candidates"][0]["candidate_id"],
        }
        for task in tasks
    ]
    b_events = deepcopy(a_events)
    b_events[0]["support_label"] = "No Support"
    b_events[0]["selected_candidate_id"] = None

    queue, report = build_adjudication_queue_with_audit(
        tasks,
        a_events,
        b_events,
        agreement_audit_fraction=0.1,
        random_state=7,
    )

    assert report["adjudication_tasks"] == 1
    assert report["agreement_audit_tasks"] == 1
    assert len(queue) == 2
    assert "task-0" not in report["agreement_audit_task_ids"]
