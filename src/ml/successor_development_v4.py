"""Materialize only the frozen successor-development source assignment."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ml.annotation import validate_queue
from ml.operational_development_v3 import TAXONOMY_REFERENCE
from ml.real_development import (
    ROLE_FAMILIES,
    build_development_task,
    evidence_extraction_manifest,
)
from ml.real_development_isolation import (
    DevelopmentIsolationIndex,
    development_task_isolated,
)
from ml.source_pool_partition import (
    ASSIGNMENT_FIELDS,
    DEVELOPMENT_SPLIT,
    SOURCE_POOL_CONSTRUCTION_STRATA,
    assignment_sha256,
)


DEVELOPMENT_ASSIGNMENT_FILENAME = "development_source_assignments.json"
PARTITION_MANIFEST_FILENAME = "partition_manifest.json"
DATASET_ID = "successor_development_v4"
PARTITION_ID = "successor_v4_source_pool"
FORBIDDEN_TASK_FIELDS = {
    "support_label",
    "selected_candidate_id",
    "candidate_labels",
    "model_prediction",
    "model_score",
    "similarity",
}


def assert_development_assignment_path(path: Path) -> Path:
    """Reject non-development inputs before any assignment file is opened."""
    resolved = path.resolve(strict=False)
    if (
        path.name != DEVELOPMENT_ASSIGNMENT_FILENAME
        or resolved.name != DEVELOPMENT_ASSIGNMENT_FILENAME
    ):
        raise ValueError(
            "Materialization accepts only development_source_assignments.json."
        )
    if any(
        "holdout" in part.lower()
        for candidate in (path, resolved)
        for part in candidate.parts
    ):
        raise ValueError("A holdout path cannot enter development materialization.")
    return resolved


def _validate_development_rows(
    assignments: Sequence[Mapping[str, Any]],
    *,
    resumes_per_family: int,
) -> dict[str, Any]:
    if resumes_per_family <= 0:
        raise ValueError("Development needs at least one resume per role family.")
    for row in assignments:
        if set(row) != ASSIGNMENT_FIELDS:
            raise ValueError(
                "Development assignments contain unknown or missing fields."
            )
        if row["split"] != DEVELOPMENT_SPLIT:
            raise ValueError("Development assignment has the wrong split.")
    allocation_ids = [str(row["allocation_id"]) for row in assignments]
    job_hashes = [str(row["source_job_hash"]) for row in assignments]
    if len(allocation_ids) != len(set(allocation_ids)):
        raise ValueError("Development allocation ids must be unique.")
    if len(job_hashes) != len(set(job_hashes)):
        raise ValueError("Development job sources must be unique.")

    groups: dict[str, list[Mapping[str, Any]]] = {}
    for row in assignments:
        groups.setdefault(str(row["source_resume_hash"]), []).append(row)
    expected_groups = resumes_per_family * len(ROLE_FAMILIES)
    if len(groups) != expected_groups:
        raise ValueError(
            f"Expected {expected_groups} development resume groups, "
            f"found {len(groups)}."
        )
    expected_strata = set(SOURCE_POOL_CONSTRUCTION_STRATA)
    for group in groups.values():
        if len(group) != len(expected_strata):
            raise ValueError("Every development resume needs six assignments.")
        if {str(row["construction_stratum"]) for row in group} != expected_strata:
            raise ValueError("Every development resume must contain every stratum.")
        if len({str(row["role_family"]) for row in group}) != 1:
            raise ValueError("A development resume cannot cross role families.")
    role_counts = Counter(str(row["role_family"]) for row in assignments)
    expected_role_tasks = resumes_per_family * len(expected_strata)
    if any(role_counts[family] != expected_role_tasks for family in ROLE_FAMILIES):
        raise ValueError("Development role-family assignments are imbalanced.")
    return {
        "tasks": len(assignments),
        "resume_groups": len(groups),
        "tasks_per_resume": len(expected_strata),
        "role_counts": dict(role_counts),
        "construction_stratum_counts": dict(
            Counter(str(row["construction_stratum"]) for row in assignments)
        ),
    }


def validate_development_assignment_input(
    assignment_path: Path,
    payload: Mapping[str, Any],
    partition_manifest: Mapping[str, Any],
    *,
    expected_partition_id: str = PARTITION_ID,
    expected_text_quality_policy: str | None = None,
    expected_evidence_extraction_policy: str | None = None,
) -> dict[str, Any]:
    """Validate the content-free development assignment and its frozen checksum."""
    assert_development_assignment_path(assignment_path)
    if str(payload.get("partition_id", "")) != expected_partition_id:
        raise ValueError("Unexpected development partition id.")
    if payload.get("split") != DEVELOPMENT_SPLIT:
        raise ValueError("Assignment payload is not the development split.")
    if any(
        bool(payload.get(field))
        for field in (
            "content_included",
            "labels_included",
            "model_signals_included",
        )
    ):
        raise ValueError("Development source assignment must remain content-free.")
    if str(partition_manifest.get("partition_id", "")) != expected_partition_id:
        raise ValueError("Unexpected source-partition manifest id.")
    if partition_manifest.get("source_assignment_frozen") is not True:
        raise ValueError("Source assignment is not frozen.")
    if partition_manifest.get("post_label_source_movement_allowed") is not False:
        raise ValueError("Source movement policy is not frozen.")
    if partition_manifest.get("product_integration_allowed") is not False:
        raise ValueError("Development partition cannot allow product integration.")
    if partition_manifest.get("demo_integration_allowed") is not False:
        raise ValueError("Development partition cannot allow Demo integration.")
    if expected_text_quality_policy is not None:
        if payload.get("source_text_quality_policy") != (
            expected_text_quality_policy
        ):
            raise ValueError("Development assignment has the wrong text policy.")
        quality_manifest = partition_manifest.get("source_text_quality")
        if not isinstance(quality_manifest, Mapping) or quality_manifest.get(
            "policy"
        ) != expected_text_quality_policy:
            raise ValueError("Partition manifest has the wrong text policy.")
    if expected_evidence_extraction_policy is not None:
        if payload.get("evidence_extraction_policy") != (
            expected_evidence_extraction_policy
        ):
            raise ValueError(
                "Development assignment has the wrong evidence extraction policy."
            )
        expected_extraction_manifest = evidence_extraction_manifest(
            expected_evidence_extraction_policy
        )
        if partition_manifest.get("evidence_extraction") != (
            expected_extraction_manifest
        ):
            raise ValueError(
                "Partition manifest has the wrong evidence extraction policy."
            )
    source_revision = str(payload.get("source_dataset_revision", ""))
    if not source_revision or source_revision != str(
        partition_manifest.get("source_dataset_revision", "")
    ):
        raise ValueError("Source dataset revision does not match the partition.")
    policy = tuple(str(value) for value in payload.get("construction_policy", []))
    if policy != SOURCE_POOL_CONSTRUCTION_STRATA:
        raise ValueError("Development construction policy has changed.")
    if policy != tuple(
        str(value)
        for value in partition_manifest.get("construction_policy", [])
    ):
        raise ValueError("Assignment and partition construction policies differ.")
    raw_assignments = payload.get("assignments")
    if not isinstance(raw_assignments, list):
        raise ValueError("Development assignments must be a list.")
    assignments = [
        dict(row) for row in raw_assignments if isinstance(row, Mapping)
    ]
    if len(assignments) != len(raw_assignments):
        raise ValueError("Every development assignment must be an object.")
    expected_checksum = str(
        partition_manifest.get("development_assignment_sha256", "")
    )
    actual_checksum = assignment_sha256(assignments)
    if not expected_checksum or actual_checksum != expected_checksum:
        raise ValueError("Development assignment checksum does not match.")
    resumes_per_family = int(
        partition_manifest.get("development_resumes_per_family", 0)
    )
    contract = _validate_development_rows(
        assignments,
        resumes_per_family=resumes_per_family,
    )
    return {
        **contract,
        "assignment_sha256": actual_checksum,
        "source_dataset_revision": source_revision,
        "random_state": int(payload.get("random_state", 0)),
        "partition_id": expected_partition_id,
    }


def validate_runtime_isolation(
    partition_manifest: Mapping[str, Any],
    *,
    source_dataset_revision: str,
    isolation_summary: Mapping[str, Any],
) -> None:
    """Reject source or consumed-data drift before opening source content."""
    if source_dataset_revision != str(
        partition_manifest.get("source_dataset_revision", "")
    ):
        raise ValueError("Local source revision changed after partitioning.")
    fields = (
        "consumed_job_hashes",
        "consumed_resume_hashes",
        "consumed_requirement_texts",
        "consumed_evidence_texts",
        "isolation_index_sha256",
    )
    if any(
        isolation_summary.get(field) != partition_manifest.get(field)
        for field in fields
    ):
        raise ValueError("Consumed-source isolation state changed after partitioning.")


def materialize_development_tasks(
    assignments: Sequence[Mapping[str, Any]],
    *,
    profiles_by_hash: Mapping[str, Mapping[str, Any]],
    jobs_by_hash: Mapping[str, Mapping[str, Any]],
    source_dataset_revision: str,
    isolation_index: DevelopmentIsolationIndex,
    partition_id: str = PARTITION_ID,
    source_dataset: str = "djinni_successor_development_v4",
) -> list[dict[str, Any]]:
    """Build canonical annotation tasks from development sources only."""
    tasks: list[dict[str, Any]] = []
    for assignment in sorted(
        assignments,
        key=lambda row: str(row["allocation_id"]),
    ):
        resume_hash = str(assignment["source_resume_hash"])
        job_hash = str(assignment["source_job_hash"])
        family = str(assignment["role_family"])
        stratum = str(assignment["construction_stratum"])
        profile = profiles_by_hash.get(resume_hash)
        job = jobs_by_hash.get(job_hash)
        if profile is None or job is None:
            raise ValueError("An assigned development source is unavailable.")
        if str(profile.get("family", "")) != family:
            raise ValueError("Assigned resume role family changed.")
        if str(job.get("family", "")) != family:
            raise ValueError("Assigned job role family changed.")
        task = build_development_task(
            requirement=str(job["requirement"]),
            evidence=[str(value) for value in profile["evidence"]],
            role_family=family,
            source_job_hash=job_hash,
            source_resume_hash=resume_hash,
            source_dataset=source_dataset,
            seed=str(assignment["allocation_id"]),
        )
        task.update(
            {
                "presentation_id": str(task["task_id"]),
                "hidden_repeat_of": None,
                "operational_resume_group": resume_hash,
                "construction_stratum": stratum,
                "taxonomy_reference": list(TAXONOMY_REFERENCE),
                "source_dataset_revision": source_dataset_revision,
                "profile_role_family": family,
                "partition_id": partition_id,
                "source_allocation_id": str(assignment["allocation_id"]),
                "split": DEVELOPMENT_SPLIT,
            }
        )
        if not development_task_isolated(task, isolation_index):
            raise ValueError("Materialized task overlaps consumed source or text.")
        tasks.append(task)
    return tasks


def validate_materialized_development(
    tasks: Sequence[dict[str, Any]],
    assignments: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate one canonical task for every frozen development allocation."""
    checked = validate_queue(tasks)
    if len(checked) != len(assignments):
        raise ValueError("Materialized task count differs from source assignments.")
    if any(set(task) & FORBIDDEN_TASK_FIELDS for task in checked):
        raise ValueError("Development task contains labels or model signals.")
    expected = {
        str(row["allocation_id"]): (
            str(row["source_resume_hash"]),
            str(row["source_job_hash"]),
            str(row["role_family"]),
            str(row["construction_stratum"]),
        )
        for row in assignments
    }
    observed: dict[str, tuple[str, str, str, str]] = {}
    for task in checked:
        allocation_id = str(task.get("source_allocation_id", ""))
        if len(task["candidates"]) != 4:
            raise ValueError("Every development task needs four candidates.")
        if task.get("split") != DEVELOPMENT_SPLIT:
            raise ValueError("Materialized task has the wrong split.")
        observed[allocation_id] = (
            str(task["source_resume_hash"]),
            str(task["source_job_hash"]),
            str(task["role_family"]),
            str(task["construction_stratum"]),
        )
    if observed != expected:
        raise ValueError("Materialized tasks do not match frozen assignments.")
    role_counts = Counter(str(task["role_family"]) for task in checked)
    stratum_counts = Counter(
        str(task["construction_stratum"]) for task in checked
    )
    resume_counts = Counter(
        str(task["source_resume_hash"]) for task in checked
    )
    if set(resume_counts.values()) != {len(SOURCE_POOL_CONSTRUCTION_STRATA)}:
        raise ValueError("Every materialized resume must own exactly six tasks.")
    return {
        "tasks": len(checked),
        "resume_groups": len(resume_counts),
        "tasks_per_resume": len(SOURCE_POOL_CONSTRUCTION_STRATA),
        "candidate_count": sum(len(task["candidates"]) for task in checked),
        "role_counts": dict(role_counts),
        "construction_stratum_counts": dict(stratum_counts),
    }
