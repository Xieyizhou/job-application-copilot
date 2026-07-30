"""Contracts for realistic one-resume-to-many-jobs development data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from typing import Any

from ml.real_development import DevelopmentDataError, ROLE_FAMILIES
from ml.real_development_isolation import (
    build_development_isolation_index,
    development_task_isolated,
)


CROSS_ROLE = {
    "Data": "ML",
    "ML": "Software",
    "Software": "Business",
    "Business": "Data",
}


def assert_operational_development_contract(
    tasks: Sequence[dict[str, Any]],
    excluded_tasks: Iterable[dict[str, Any]],
    excluded_pairs: Iterable[dict[str, Any]],
    *,
    tasks_per_resume: int,
    resumes_per_profile_family: int,
) -> dict[str, Any]:
    """Validate source isolation while allowing intentional resume reuse."""
    excluded_task_list = list(excluded_tasks)
    excluded_pair_list = list(excluded_pairs)
    index = build_development_isolation_index(
        excluded_task_list,
        excluded_pair_list,
    )
    task_ids = [str(task["task_id"]) for task in tasks]
    job_hashes = [str(task["source_job_hash"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise DevelopmentDataError("Operational task ids must be unique.")
    if len(job_hashes) != len(set(job_hashes)):
        raise DevelopmentDataError(
            "Operational tasks must use distinct job sources."
        )
    if any(not development_task_isolated(task, index) for task in tasks):
        raise DevelopmentDataError(
            "Operational development overlaps consumed source or text."
        )
    resume_counts = Counter(str(task["source_resume_hash"]) for task in tasks)
    if set(resume_counts.values()) != {tasks_per_resume}:
        raise DevelopmentDataError(
            "Each operational resume must have the required task count."
        )
    expected_resumes = resumes_per_profile_family * len(ROLE_FAMILIES)
    if len(resume_counts) != expected_resumes:
        raise DevelopmentDataError(
            "Operational development has the wrong number of resume groups."
        )
    profile_counts = Counter(
        str(task["profile_role_family"]) for task in tasks
    )
    if any(
        profile_counts[family]
        != resumes_per_profile_family * tasks_per_resume
        for family in ROLE_FAMILIES
    ):
        raise DevelopmentDataError(
            "Operational profile role families are not balanced."
        )
    requirement_counts = Counter(
        str(task["role_family"]) for task in tasks
    )
    if len(set(requirement_counts.values())) != 1:
        raise DevelopmentDataError(
            "Operational requirement role families are not balanced."
        )
    pairing_counts = Counter(str(task["pairing_type"]) for task in tasks)
    return {
        "tasks": len(tasks),
        "resume_groups": len(resume_counts),
        "tasks_per_resume": tasks_per_resume,
        "profile_role_counts": dict(profile_counts),
        "requirement_role_counts": dict(requirement_counts),
        "pairing_counts": dict(pairing_counts),
        "source_job_overlap": 0,
        "source_resume_overlap": 0,
        "content_overlap": 0,
        "resume_group_split_required": True,
    }
