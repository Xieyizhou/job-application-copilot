from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.grouped_validation import (
    GroupedValidationError,
    grouped_split,
    grouped_split_report,
)


def _row(
    task_id: str,
    resume_hash: str,
    job_hash: str,
    semantic_group: str,
) -> dict[str, str]:
    return {
        "task_id": task_id,
        "source_resume_hash": resume_hash,
        "source_job_hash": job_hash,
        "semantic_case_group_id": semantic_group,
        "support_label": "Direct",
    }


def test_connected_resume_job_and_semantic_groups_never_cross_splits() -> None:
    rows = [
        _row("task-1", "resume-a", "job-a", "semantic-a"),
        _row("task-2", "resume-a", "job-b", "semantic-b"),
        _row("task-3", "resume-b", "job-c", "semantic-b"),
        _row("task-4", "resume-c", "job-d", "semantic-c"),
        _row("task-5", "resume-d", "job-e", "semantic-d"),
        _row("task-6", "resume-e", "job-f", "semantic-e"),
    ]

    split_rows = grouped_split(rows, random_state=17)
    report = grouped_split_report(split_rows)
    task_splits = {
        row["task_id"]: split
        for split, split_items in split_rows.items()
        for row in split_items
    }

    assert task_splits["task-1"] == task_splits["task-2"] == task_splits["task-3"]
    assert all(value == 0 for value in report["resume_overlap"].values())
    assert all(value == 0 for value in report["job_overlap"].values())
    assert all(value == 0 for value in report["semantic_overlap"].values())
    assert len(report["holdout_checksum_sha256"]) == 64


def test_grouped_split_is_deterministic() -> None:
    rows = [
        _row(f"task-{index}", f"resume-{index}", f"job-{index}", f"group-{index}")
        for index in range(12)
    ]

    first = grouped_split(rows, random_state=9)
    second = grouped_split(list(reversed(rows)), random_state=9)

    assert first == second


def test_real_rows_require_anonymous_resume_and_job_hashes() -> None:
    row = _row("task-1", "", "job-a", "semantic-a")

    with pytest.raises(GroupedValidationError, match="resume and job hashes"):
        grouped_split([row])
