"""Tests for the local teacher-error diagnostic dashboard."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.teacher_error_dashboard import latest_decisions, save_decision


def test_latest_teacher_error_decision_wins() -> None:
    rows = [
        {"audit_id": "a", "error_type": "Other"},
        {"audit_id": "a", "error_type": "Teacher ranking failure"},
    ]

    assert latest_decisions(rows)["a"]["error_type"] == "Teacher ranking failure"


def test_teacher_error_decision_is_append_only_and_diagnostic(tmp_path: Path) -> None:
    path = tmp_path / "decisions.jsonl"
    task = {"audit_id": "audit-1", "task_id": "task-1"}

    save_decision(
        task,
        error_type="Teacher ranking failure",
        note="reviewed",
        events_path=path,
    )

    content = path.read_text(encoding="utf-8")
    assert '"diagnostic_only": true' in content
    assert '"human_gold_changed": false' in content
    assert '"model_selection_allowed": false' in content

    with pytest.raises(ValueError, match="Unknown"):
        save_decision(
            task,
            error_type="Promote model",
            note="",
            events_path=path,
        )
