"""Contracts for development-only successor packet materialization."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from ml.operational_development_v3 import (
    build_single_reviewer_packet,
    validate_reviewer_packet,
)
from ml.real_development import (
    CONTEXTUAL_EVIDENCE_EXTRACTION,
    ROLE_FAMILIES,
    evidence_extraction_manifest,
)
from ml.real_development_isolation import (
    build_development_isolation_index,
)
from ml.source_pool_partition import (
    DEVELOPMENT_SPLIT,
    SOURCE_POOL_CONSTRUCTION_STRATA,
    assignment_sha256,
    isolation_index_summary,
)
from ml.source_text_quality import (
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    audit_materialized_text_quality,
    source_text_quality_manifest,
)
from ml.successor_development_v4 import (
    DEVELOPMENT_ASSIGNMENT_FILENAME,
    PARTITION_ID,
    assert_development_assignment_path,
    materialize_development_tasks,
    validate_development_assignment_input,
    validate_materialized_development,
    validate_runtime_isolation,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_REVISION = "djinni-manifest-sha256:test-revision"


def _load_materializer_script() -> object:
    path = (
        PROJECT_ROOT
        / "scripts"
        / "ml"
        / "materialize_successor_development.py"
    )
    spec = importlib.util.spec_from_file_location(
        "materialize_successor_development",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _assigned_sources() -> tuple[
    list[dict[str, str]],
    dict[str, dict[str, object]],
    dict[str, dict[str, object]],
]:
    assignments: list[dict[str, str]] = []
    profiles: dict[str, dict[str, object]] = {}
    jobs: dict[str, dict[str, object]] = {}
    for family_index, family in enumerate(ROLE_FAMILIES):
        for group_index in range(6):
            resume_hash = f"resume-{family_index}-{group_index}"
            profiles[resume_hash] = {
                "source_hash": resume_hash,
                "family": family,
                "evidence": [
                    (
                        f"Built unique {family} artifact {candidate_index} "
                        f"for group {group_index} using method "
                        f"method{candidate_index} and measured "
                        f"outcome{candidate_index}."
                    )
                    for candidate_index in range(4)
                ],
            }
            for stratum_index, stratum in enumerate(
                SOURCE_POOL_CONSTRUCTION_STRATA
            ):
                job_hash = (
                    f"job-{family_index}-{group_index}-{stratum_index}"
                )
                allocation_id = (
                    f"allocation-{family_index}-{group_index}-{stratum_index}"
                )
                jobs[job_hash] = {
                    "source_hash": job_hash,
                    "family": family,
                    "requirement": (
                        f"Develop {family} capability for group {group_index} "
                        f"with requirement {stratum_index} and deliver "
                        f"measurable stakeholder results."
                    ),
                }
                assignments.append(
                    {
                        "allocation_id": allocation_id,
                        "split": DEVELOPMENT_SPLIT,
                        "role_family": family,
                        "construction_stratum": stratum,
                        "source_resume_hash": resume_hash,
                        "source_job_hash": job_hash,
                    }
                )
    return assignments, profiles, jobs


def _payload_and_manifest(
    assignments: list[dict[str, str]],
    *,
    partition_id: str = PARTITION_ID,
    text_quality_policy: str | None = None,
    evidence_extraction_policy: str | None = None,
) -> tuple[dict[str, object], dict[str, object]]:
    checksum = assignment_sha256(assignments)
    payload: dict[str, object] = {
        "schema_version": 1,
        "partition_id": partition_id,
        "split": DEVELOPMENT_SPLIT,
        "source_dataset_revision": SOURCE_REVISION,
        "construction_policy": list(SOURCE_POOL_CONSTRUCTION_STRATA),
        "random_state": 41,
        "content_included": False,
        "labels_included": False,
        "model_signals_included": False,
        "assignments": assignments,
    }
    if text_quality_policy is not None:
        payload["source_text_quality_policy"] = text_quality_policy
    if evidence_extraction_policy is not None:
        payload["evidence_extraction_policy"] = evidence_extraction_policy
    isolation = isolation_index_summary(
        build_development_isolation_index([], [])
    )
    manifest: dict[str, object] = {
        "schema_version": 1,
        "partition_id": partition_id,
        "source_dataset_revision": SOURCE_REVISION,
        "construction_policy": list(SOURCE_POOL_CONSTRUCTION_STRATA),
        "development_resumes_per_family": 6,
        "development_assignment_sha256": checksum,
        "source_assignment_frozen": True,
        "post_label_source_movement_allowed": False,
        "product_integration_allowed": False,
        "demo_integration_allowed": False,
        **isolation,
    }
    if text_quality_policy is not None:
        manifest["source_text_quality"] = source_text_quality_manifest(
            text_quality_policy
        )
    if evidence_extraction_policy is not None:
        manifest["evidence_extraction"] = evidence_extraction_manifest(
            evidence_extraction_policy
        )
    return payload, manifest


def test_assignment_input_is_frozen_balanced_and_content_free() -> None:
    assignments, _, _ = _assigned_sources()
    payload, manifest = _payload_and_manifest(assignments)
    contract = validate_development_assignment_input(
        Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
        payload,
        manifest,
    )

    assert contract["tasks"] == 144
    assert contract["resume_groups"] == 24
    assert contract["tasks_per_resume"] == 6
    assert set(contract["role_counts"].values()) == {36}
    assert set(contract["construction_stratum_counts"].values()) == {24}
    assert contract["assignment_sha256"] == assignment_sha256(assignments)


def test_materializer_rejects_every_non_development_assignment_path(
    tmp_path: Path,
) -> None:
    assert_development_assignment_path(
        Path("source_pool") / DEVELOPMENT_ASSIGNMENT_FILENAME
    )
    with pytest.raises(ValueError, match="accepts only"):
        assert_development_assignment_path(Path("other_assignments.json"))
    with pytest.raises(ValueError, match="holdout path"):
        assert_development_assignment_path(
            Path("holdout") / DEVELOPMENT_ASSIGNMENT_FILENAME
        )
    holdout_file = tmp_path / "sealed_holdout_assignments.json"
    holdout_file.write_text("{}", encoding="utf-8")
    disguised = tmp_path / DEVELOPMENT_ASSIGNMENT_FILENAME
    disguised.symlink_to(holdout_file)
    with pytest.raises(ValueError, match="accepts only"):
        assert_development_assignment_path(disguised)


def test_assignment_checksum_revision_and_isolation_drift_are_rejected() -> None:
    assignments, _, _ = _assigned_sources()
    payload, manifest = _payload_and_manifest(assignments)
    changed = deepcopy(payload)
    changed_assignments = deepcopy(assignments)
    changed_assignments[0]["source_job_hash"] = "changed-job"
    changed["assignments"] = changed_assignments
    with pytest.raises(ValueError, match="checksum"):
        validate_development_assignment_input(
            Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
            changed,
            manifest,
        )

    wrong_revision = deepcopy(payload)
    wrong_revision["source_dataset_revision"] = "different-revision"
    with pytest.raises(ValueError, match="revision"):
        validate_development_assignment_input(
            Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
            wrong_revision,
            manifest,
        )

    isolation = isolation_index_summary(
        build_development_isolation_index([], [])
    )
    validate_runtime_isolation(
        manifest,
        source_dataset_revision=SOURCE_REVISION,
        isolation_summary=isolation,
    )
    changed_isolation = dict(isolation)
    changed_isolation["consumed_job_hashes"] = 1
    with pytest.raises(ValueError, match="isolation state changed"):
        validate_runtime_isolation(
            manifest,
            source_dataset_revision=SOURCE_REVISION,
            isolation_summary=changed_isolation,
        )


def test_v5_assignment_requires_frozen_matching_text_policy() -> None:
    assignments, _, _ = _assigned_sources()
    partition_id = "successor_v5_source_pool"
    payload, manifest = _payload_and_manifest(
        assignments,
        partition_id=partition_id,
        text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    )

    contract = validate_development_assignment_input(
        Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
        payload,
        manifest,
        expected_partition_id=partition_id,
        expected_text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    )

    assert contract["partition_id"] == partition_id
    wrong_payload = deepcopy(payload)
    wrong_payload["source_text_quality_policy"] = "legacy"
    with pytest.raises(ValueError, match="wrong text policy"):
        validate_development_assignment_input(
            Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
            wrong_payload,
            manifest,
            expected_partition_id=partition_id,
            expected_text_quality_policy=SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
        )


def test_v6_assignment_requires_canonical_text_policy() -> None:
    assignments, _, _ = _assigned_sources()
    partition_id = "successor_v6_source_pool"
    payload, manifest = _payload_and_manifest(
        assignments,
        partition_id=partition_id,
        text_quality_policy=SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    )

    contract = validate_development_assignment_input(
        Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
        payload,
        manifest,
        expected_partition_id=partition_id,
        expected_text_quality_policy=SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    )

    assert contract["partition_id"] == partition_id
    assert manifest["source_text_quality"] == source_text_quality_manifest(
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY
    )


def test_v7_assignment_requires_sentence_boundary_policy() -> None:
    assignments, _, _ = _assigned_sources()
    partition_id = "successor_v7_source_pool"
    payload, manifest = _payload_and_manifest(
        assignments,
        partition_id=partition_id,
        text_quality_policy=SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    )

    contract = validate_development_assignment_input(
        Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
        payload,
        manifest,
        expected_partition_id=partition_id,
        expected_text_quality_policy=SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    )

    assert contract["partition_id"] == partition_id
    assert manifest["source_text_quality"] == source_text_quality_manifest(
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY
    )


def test_v9_assignment_requires_frozen_evidence_extraction_policy() -> None:
    assignments, _, _ = _assigned_sources()
    partition_id = "successor_v9_source_pool"
    payload, manifest = _payload_and_manifest(
        assignments,
        partition_id=partition_id,
        evidence_extraction_policy=CONTEXTUAL_EVIDENCE_EXTRACTION,
    )

    contract = validate_development_assignment_input(
        Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
        payload,
        manifest,
        expected_partition_id=partition_id,
        expected_evidence_extraction_policy=(
            CONTEXTUAL_EVIDENCE_EXTRACTION
        ),
    )

    assert contract["partition_id"] == partition_id
    wrong_payload = deepcopy(payload)
    wrong_payload["evidence_extraction_policy"] = "action_prefix_v1"
    with pytest.raises(ValueError, match="wrong evidence extraction policy"):
        validate_development_assignment_input(
            Path(DEVELOPMENT_ASSIGNMENT_FILENAME),
            wrong_payload,
            manifest,
            expected_partition_id=partition_id,
            expected_evidence_extraction_policy=(
                CONTEXTUAL_EVIDENCE_EXTRACTION
            ),
        )


def test_materialized_packet_has_144_tasks_and_blind_hidden_repeats() -> None:
    assignments, profiles, jobs = _assigned_sources()
    isolation = build_development_isolation_index([], [])
    tasks = materialize_development_tasks(
        assignments,
        profiles_by_hash=profiles,
        jobs_by_hash=jobs,
        source_dataset_revision=SOURCE_REVISION,
        isolation_index=isolation,
    )
    contract = validate_materialized_development(tasks, assignments)
    reviewer_queue, repeat_map = build_single_reviewer_packet(
        tasks,
        repeat_count=14,
        random_state=41,
        presentation_prefix="successor-v4-presentation",
    )
    validate_reviewer_packet(reviewer_queue)

    assert contract["tasks"] == 144
    assert contract["resume_groups"] == 24
    assert contract["candidate_count"] == 576
    assert len(reviewer_queue) == 158
    assert len(repeat_map) == 14
    assert all(
        row["presentation_id"].startswith("successor-v4-presentation-")
        for row in repeat_map
    )
    repeat_by_id = {
        str(row["presentation_id"]): row for row in reviewer_queue
    }
    original_by_id = {
        str(row["task_id"]): row
        for row in reviewer_queue
        if str(row["task_id"]).startswith("real-dev-")
    }
    for repeat in repeat_map:
        repeated = repeat_by_id[str(repeat["presentation_id"])]
        original = original_by_id[str(repeat["canonical_task_id"])]
        assert [
            candidate["candidate_id"] for candidate in repeated["candidates"]
        ] != [
            candidate["candidate_id"] for candidate in original["candidates"]
        ]
    serialized = json.dumps(reviewer_queue)
    for forbidden in (
        "construction_stratum",
        "taxonomy_reference",
        "source_dataset_revision",
        "source_resume_hash",
        "source_job_hash",
        "source_allocation_id",
        "model_prediction",
    ):
        assert forbidden not in serialized


def test_v5_materialization_records_offline_partition_and_passes_text_audit() -> None:
    assignments, profiles, jobs = _assigned_sources()
    tasks = materialize_development_tasks(
        assignments,
        profiles_by_hash=profiles,
        jobs_by_hash=jobs,
        source_dataset_revision=SOURCE_REVISION,
        isolation_index=build_development_isolation_index([], []),
        partition_id="successor_v5_source_pool",
        source_dataset="djinni_successor_development_v5",
    )

    audit = audit_materialized_text_quality(tasks)

    assert {task["partition_id"] for task in tasks} == {
        "successor_v5_source_pool"
    }
    assert {task["source_dataset"] for task in tasks} == {
        "djinni_successor_development_v5"
    }
    assert audit["automated_quality_gate_passed"] is True
    assert audit["content_included"] is False


def test_materialized_text_audit_reports_ids_without_source_content() -> None:
    assignments, profiles, jobs = _assigned_sources()
    jobs[str(assignments[0]["source_job_hash"])]["requirement"] = (
        "Experience with Azure data services, including"
    )
    tasks = materialize_development_tasks(
        assignments,
        profiles_by_hash=profiles,
        jobs_by_hash=jobs,
        source_dataset_revision=SOURCE_REVISION,
        isolation_index=build_development_isolation_index([], []),
    )

    audit = audit_materialized_text_quality(tasks)
    serialized = json.dumps(audit)

    assert audit["automated_quality_gate_passed"] is False
    assert audit["blocked_tasks"] == 1
    assert "requirement_unfinished_tail" in audit["finding_counts"]
    assert "Azure data services" not in serialized


def test_materialized_text_audit_runs_after_public_text_cleanup() -> None:
    assignments, profiles, jobs = _assigned_sources()
    jobs[str(assignments[0]["source_job_hash"])]["requirement"] = (
        "(Full-time) Experience with REST API integration."
    )
    tasks = materialize_development_tasks(
        assignments,
        profiles_by_hash=profiles,
        jobs_by_hash=jobs,
        source_dataset_revision=SOURCE_REVISION,
        isolation_index=build_development_isolation_index([], []),
    )

    audit = audit_materialized_text_quality(tasks)

    assert audit["automated_quality_gate_passed"] is False
    assert audit["finding_counts"] == {"requirement_word_count": 1}


def test_materializer_has_no_application_import_or_holdout_assignment_read() -> None:
    module_source = (
        PROJECT_ROOT / "src" / "ml" / "successor_development_v4.py"
    ).read_text(encoding="utf-8")
    script_source = (
        PROJECT_ROOT
        / "scripts"
        / "ml"
        / "materialize_successor_development.py"
    ).read_text(encoding="utf-8")
    combined = module_source + script_source

    assert "sealed_holdout_source_assignments.json" not in combined
    assert "src/dashboard" not in combined
    assert "from dashboard" not in combined
    assert "cover_letter" not in combined
    assert '"successor_v5"' in script_source
    assert '"generate_reviewer_packet": False' in script_source
    assert '"successor_v6"' in script_source


def test_v6_materializer_is_canonical_only_and_development_scoped() -> None:
    module = _load_materializer_script()
    config = module.VERSION_CONFIGS["successor_v6"]  # type: ignore[attr-defined]

    assert config["generate_reviewer_packet"] is False
    assert config["assignment_path"].name == (
        "development_source_assignments.json"
    )
    assert "holdout" not in str(config["assignment_path"]).lower()
    assert config["dataset_id"] == "successor_development_v6"


def test_v7_materializer_is_canonical_only_and_development_scoped() -> None:
    module = _load_materializer_script()
    config = module.VERSION_CONFIGS["successor_v7"]  # type: ignore[attr-defined]

    assert config["generate_reviewer_packet"] is False
    assert config["assignment_path"].name == (
        "development_source_assignments.json"
    )
    assert "holdout" not in str(config["assignment_path"]).lower()
    assert config["dataset_id"] == "successor_development_v7"


def test_v8_materializer_is_qa_only_and_development_scoped() -> None:
    module = _load_materializer_script()
    config = module.VERSION_CONFIGS[  # type: ignore[attr-defined]
        "successor_v8"
    ]

    assert config["generate_reviewer_packet"] is False
    assert config["assignment_path"].name == (
        "development_source_assignments.json"
    )
    assert "holdout" not in str(config["assignment_path"]).lower()
    assert config["dataset_id"] == "successor_development_v8"
    assert config["partition_id"] == "successor_v8_development"


def test_v9_materializer_is_qa_only_and_development_scoped() -> None:
    module = _load_materializer_script()
    config = module.VERSION_CONFIGS[  # type: ignore[attr-defined]
        "successor_v9"
    ]

    assert config["generate_reviewer_packet"] is False
    assert config["assignment_path"].name == (
        "development_source_assignments.json"
    )
    assert "holdout" not in str(config["assignment_path"]).lower()
    assert config["dataset_id"] == "successor_development_v9"
    assert config["partition_id"] == "successor_v9_source_pool"
    assert config["evidence_extraction_policy"] == (
        CONTEXTUAL_EVIDENCE_EXTRACTION
    )
