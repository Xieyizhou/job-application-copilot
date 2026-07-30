"""Tests for active-learning acceptance supplement sampling."""

from __future__ import annotations

from collections import Counter

import pytest

from ml.acceptance_sampling import (
    ACCEPTANCE_STRATA,
    select_balanced_acceptance_tasks,
)
from ml.real_development import ROLE_FAMILIES


def _scored_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in ROLE_FAMILIES:
        for index, preferred in enumerate(ACCEPTANCE_STRATA):
            task_id = f"{family.lower()}-{index}"
            affinities = {
                stratum: (10.0 if stratum == preferred else float(index))
                for stratum in ACCEPTANCE_STRATA
            }
            rows.append(
                {
                    "task": {
                        "task_id": task_id,
                        "role_family": family,
                        "source_job_hash": f"job-{task_id}",
                        "source_resume_hash": f"resume-{task_id}",
                    },
                    "affinities": affinities,
                    "signals": {"support_probability_at_lsa_top": 0.5},
                }
            )
    return rows


def test_balanced_acceptance_selection_uses_every_stratum_and_role() -> None:
    tasks, construction = select_balanced_acceptance_tasks(
        _scored_rows(),
        tasks_per_family=4,
    )

    assert len(tasks) == 16
    assert Counter(str(task["role_family"]) for task in tasks) == {
        family: 4 for family in ROLE_FAMILIES
    }
    assert Counter(str(row["stratum"]) for row in construction) == {
        stratum: 4 for stratum in ACCEPTANCE_STRATA
    }
    assert len({str(task["source_job_hash"]) for task in tasks}) == 16
    assert len({str(task["source_resume_hash"]) for task in tasks}) == 16


def test_acceptance_selection_rejects_unbalanced_target() -> None:
    with pytest.raises(ValueError, match="divisible by four"):
        select_balanced_acceptance_tasks(
            _scored_rows(),
            tasks_per_family=5,
        )


def test_acceptance_selection_rejects_reused_sources() -> None:
    rows = _scored_rows()
    for row in rows:
        task = row["task"]
        assert isinstance(task, dict)
        task["source_resume_hash"] = "same-resume"

    with pytest.raises(ValueError, match="Insufficient unique"):
        select_balanced_acceptance_tasks(rows, tasks_per_family=4)
