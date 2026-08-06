"""Contracts for source-only development and sealed-holdout partitioning."""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path

import pytest

from ml.real_development import ROLE_FAMILIES
from ml.real_development_isolation import (
    build_development_isolation_index,
)
from ml.source_pool_partition import (
    ASSIGNMENT_FIELDS,
    DEVELOPMENT_SPLIT,
    HOLDOUT_SPLIT,
    SOURCE_POOL_CONSTRUCTION_STRATA,
    assignment_sha256,
    build_source_partition,
    isolation_index_summary,
    source_pool_preflight,
    validate_source_partition,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_partition_script() -> object:
    path = PROJECT_ROOT / "scripts" / "ml" / "build_successor_source_partition.py"
    spec = importlib.util.spec_from_file_location(
        "build_successor_source_partition",
        path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _source_pools() -> tuple[
    dict[str, list[dict[str, object]]],
    dict[str, list[dict[str, object]]],
]:
    profiles: dict[str, list[dict[str, object]]] = {}
    jobs: dict[str, list[dict[str, object]]] = {}
    role_tokens = (
        "rolewordalpha",
        "rolewordbravo",
        "rolewordcharlie",
        "roleworddelta",
    )
    for family_index, family in enumerate(ROLE_FAMILIES):
        profiles[family] = [
            {
                "source_hash": f"resume-{family_index}-{profile_index}",
                "family": family,
                "evidence": [
                    (
                        f"Built {family} profile{profile_index} artifact"
                        f"{evidence_index} using unique method"
                        f"{profile_index}_{evidence_index} and measured outcomes "
                        f"signature{family_index}p{profile_index} "
                        f"evidenceword{evidence_index}."
                    )
                    for evidence_index in range(4)
                ],
            }
            for profile_index in range(5)
        ]
        jobs[family] = [
            {
                "source_hash": f"job-{family_index}-{job_index}",
                "family": family,
                "requirement": (
                    f"Develop {family} {role_tokens[family_index]} capability "
                    f"jobword{chr(97 + job_index // 26)}"
                    f"{chr(97 + job_index % 26)} with method "
                    f"skillword{chr(97 + job_index // 26)}"
                    f"{chr(97 + job_index % 26)} and deliverable "
                    f"outcomeword{chr(97 + job_index // 26)}"
                    f"{chr(97 + job_index % 26)} for stakeholders."
                ),
            }
            for job_index in range(36)
        ]
    return profiles, jobs


def test_partition_is_deterministic_balanced_and_content_free() -> None:
    profiles, jobs = _source_pools()
    isolation = build_development_isolation_index([], [])
    first = build_source_partition(
        profiles,
        jobs,
        isolation_index=isolation,
        development_resumes_per_family=2,
        holdout_resumes_per_family=2,
        random_state=31,
    )
    reversed_profiles = {family: list(reversed(rows)) for family, rows in profiles.items()}
    reversed_jobs = {family: list(reversed(rows)) for family, rows in jobs.items()}
    second = build_source_partition(
        reversed_profiles,
        reversed_jobs,
        isolation_index=isolation,
        development_resumes_per_family=2,
        holdout_resumes_per_family=2,
        random_state=31,
    )

    assert first == second
    development, holdout = first
    report = validate_source_partition(
        development,
        holdout,
        development_resumes_per_family=2,
        holdout_resumes_per_family=2,
    )
    assert report["development_resume_groups"] == 8
    assert report["holdout_resume_groups"] == 8
    assert report["development_assignments"] == 48
    assert report["holdout_assignments"] == 48
    assert report["source_resume_overlap"] == 0
    assert report["source_job_overlap"] == 0
    assert all(set(row) == ASSIGNMENT_FIELDS for row in [*development, *holdout])
    assert all(row["split"] == DEVELOPMENT_SPLIT for row in development)
    assert all(row["split"] == HOLDOUT_SPLIT for row in holdout)
    assert not any(
        key in row
        for row in [*development, *holdout]
        for key in (
            "requirement",
            "evidence",
            "support_label",
            "model_score",
        )
    )


def test_development_only_batch_is_balanced_without_creating_holdout() -> None:
    profiles, jobs = _source_pools()
    development, holdout = build_source_partition(
        profiles,
        jobs,
        isolation_index=build_development_isolation_index([], []),
        development_resumes_per_family=2,
        holdout_resumes_per_family=0,
        random_state=37,
    )

    report = validate_source_partition(
        development,
        holdout,
        development_resumes_per_family=2,
        holdout_resumes_per_family=0,
    )

    assert report["development_assignments"] == 48
    assert report["development_resume_groups"] == 8
    assert report["holdout_assignments"] == 0
    assert report["holdout_resume_groups"] == 0
    assert holdout == []


def test_partition_rejects_cross_split_sources_and_content() -> None:
    profiles, jobs = _source_pools()
    development, holdout = build_source_partition(
        profiles,
        jobs,
        isolation_index=build_development_isolation_index([], []),
        development_resumes_per_family=1,
        holdout_resumes_per_family=1,
        random_state=9,
    )
    overlapping = deepcopy(holdout)
    overlapping[0]["source_resume_hash"] = development[0]["source_resume_hash"]
    with pytest.raises(ValueError, match="resume groups overlap"):
        validate_source_partition(
            development,
            overlapping,
            development_resumes_per_family=1,
            holdout_resumes_per_family=1,
        )

    exposed = deepcopy(development)
    exposed[0]["requirement"] = "hidden content"
    with pytest.raises(ValueError, match="unknown or missing fields"):
        validate_source_partition(
            exposed,
            holdout,
            development_resumes_per_family=1,
            holdout_resumes_per_family=1,
        )


def test_partition_excludes_consumed_source_and_near_text() -> None:
    profiles, jobs = _source_pools()
    consumed_profile = profiles[ROLE_FAMILIES[0]][0]
    consumed_job = jobs[ROLE_FAMILIES[0]][0]
    isolation = build_development_isolation_index(
        [
            {
                "source_resume_hash": consumed_profile["source_hash"],
                "source_job_hash": consumed_job["source_hash"],
                "requirement": consumed_job["requirement"],
                "candidates": [
                    {
                        "candidate_id": str(index),
                        "evidence": evidence,
                    }
                    for index, evidence in enumerate(
                        consumed_profile["evidence"]  # type: ignore[arg-type]
                    )
                ],
            }
        ],
        [],
    )
    development, holdout = build_source_partition(
        profiles,
        jobs,
        isolation_index=isolation,
        development_resumes_per_family=1,
        holdout_resumes_per_family=1,
        random_state=17,
    )
    all_rows = [*development, *holdout]

    assert consumed_profile["source_hash"] not in {row["source_resume_hash"] for row in all_rows}
    assert consumed_job["source_hash"] not in {row["source_job_hash"] for row in all_rows}
    summary = isolation_index_summary(isolation)
    assert summary["consumed_resume_hashes"] == 1
    assert summary["consumed_job_hashes"] == 1
    serialized = json.dumps(summary)
    assert str(consumed_job["requirement"]) not in serialized
    assert str(consumed_profile["evidence"][0]) not in serialized  # type: ignore[index]


def test_preflight_reports_only_aggregate_isolated_capacity() -> None:
    profiles, jobs = _source_pools()
    consumed_profile = profiles[ROLE_FAMILIES[0]][0]
    consumed_job = jobs[ROLE_FAMILIES[0]][0]
    isolation = build_development_isolation_index(
        [
            {
                "source_resume_hash": consumed_profile["source_hash"],
                "source_job_hash": consumed_job["source_hash"],
                "requirement": consumed_job["requirement"],
                "candidates": [
                    {"candidate_id": str(index), "evidence": evidence}
                    for index, evidence in enumerate(
                        consumed_profile["evidence"]  # type: ignore[arg-type]
                    )
                ],
            }
        ],
        [],
    )

    report = source_pool_preflight(
        profiles,
        jobs,
        isolation_index=isolation,
        development_resumes_per_family=2,
        holdout_resumes_per_family=2,
    )

    data = report["role_families"][ROLE_FAMILIES[0]]
    assert data["loaded_profiles"] == 5
    assert data["isolated_profiles_found"] == 4
    assert data["required_profiles"] == 4
    assert data["isolated_requirements_found"] == 24
    assert report["capacity_counts_capped_at_required"] is True
    assert report["preflight_passed"] is True
    serialized = json.dumps(report)
    assert str(consumed_profile["source_hash"]) not in serialized
    assert str(consumed_job["source_hash"]) not in serialized
    assert str(consumed_job["requirement"]) not in serialized


def test_preflight_blocks_before_partition_when_one_family_lacks_profiles() -> None:
    profiles, jobs = _source_pools()
    profiles[ROLE_FAMILIES[0]] = profiles[ROLE_FAMILIES[0]][:1]

    report = source_pool_preflight(
        profiles,
        jobs,
        isolation_index=build_development_isolation_index([], []),
        development_resumes_per_family=2,
        holdout_resumes_per_family=2,
    )

    assert report["preflight_passed"] is False
    row = report["role_families"][ROLE_FAMILIES[0]]
    assert row["isolated_profiles_found"] == 1
    assert row["required_profiles"] == 4
    assert row["necessary_capacity_passed"] is False


def test_assignment_checksum_changes_with_source_assignment() -> None:
    profiles, jobs = _source_pools()
    development, holdout = build_source_partition(
        profiles,
        jobs,
        isolation_index=build_development_isolation_index([], []),
        development_resumes_per_family=1,
        holdout_resumes_per_family=1,
        random_state=5,
    )
    changed = deepcopy(development)
    changed[0]["source_job_hash"] = "different-job"

    assert assignment_sha256(development) == assignment_sha256(list(reversed(development)))
    assert assignment_sha256(development) != assignment_sha256(changed)
    assert assignment_sha256(development) != assignment_sha256(holdout)


def test_partition_code_has_no_application_or_demo_imports() -> None:
    source = (PROJECT_ROOT / "src" / "ml" / "source_pool_partition.py").read_text(encoding="utf-8")
    script = (PROJECT_ROOT / "scripts" / "ml" / "build_successor_source_partition.py").read_text(
        encoding="utf-8"
    )
    combined = source + script

    assert "dashboard" not in combined.lower()
    assert "cover_letter" not in combined
    assert "src/dashboard" not in combined
    assert SOURCE_POOL_CONSTRUCTION_STRATA == (
        "explicit_aligned",
        "semantic_aligned",
        "compound_requirement",
        "same_role_hard_negative",
        "support_absence_probe",
        "lexical_similarity_trap",
    )


def test_v5_retires_both_content_free_v4_source_splits(
    tmp_path: Path,
) -> None:
    module = _load_partition_script()
    partition_dir = tmp_path / "successor_v4"
    partition_dir.mkdir()
    development = [
        {
            "allocation_id": "development-allocation",
            "split": DEVELOPMENT_SPLIT,
            "role_family": "Data",
            "construction_stratum": "explicit_aligned",
            "source_resume_hash": "resume-development",
            "source_job_hash": "job-development",
        }
    ]
    holdout = [
        {
            "allocation_id": "holdout-allocation",
            "split": HOLDOUT_SPLIT,
            "role_family": "Data",
            "construction_stratum": "explicit_aligned",
            "source_resume_hash": "resume-holdout",
            "source_job_hash": "job-holdout",
        }
    ]
    payload_base = {
        "content_included": False,
        "labels_included": False,
        "model_signals_included": False,
    }
    (partition_dir / "development_source_assignments.json").write_text(
        json.dumps({**payload_base, "assignments": development}),
        encoding="utf-8",
    )
    (partition_dir / "sealed_holdout_source_assignments.json").write_text(
        json.dumps({**payload_base, "assignments": holdout}),
        encoding="utf-8",
    )
    (partition_dir / "partition_manifest.json").write_text(
        json.dumps(
            {
                "partition_id": "successor_v4_source_pool",
                "holdout_content_opened": False,
                "holdout_labels_opened": False,
                "development_assignment_sha256": assignment_sha256(development),
                "sealed_holdout_assignment_sha256": assignment_sha256(holdout),
            }
        ),
        encoding="utf-8",
    )

    jobs, resumes, summary = module._retired_source_hashes(  # type: ignore[attr-defined]
        (partition_dir,)
    )

    assert jobs == {"job-development", "job-holdout"}
    assert resumes == {"resume-development", "resume-holdout"}
    assert summary["retired_holdout_assignment_content_opened"] is False
    assert summary["retired_partition_ids"] == ["successor_v4_source_pool"]


def test_v5_uses_one_quality_policy_for_both_source_splits() -> None:
    module = _load_partition_script()
    config = module.VERSION_CONFIGS["successor_v5"]  # type: ignore[attr-defined]
    development = module._assignment_payload(  # type: ignore[attr-defined]
        split=DEVELOPMENT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=7,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )
    holdout = module._assignment_payload(  # type: ignore[attr-defined]
        split=HOLDOUT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=7,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )

    assert development["source_text_quality_policy"] == (holdout["source_text_quality_policy"])
    assert development["partition_id"] == "successor_v5_source_pool"


def test_v6_retires_v4_and_v5_and_freezes_one_canonical_policy() -> None:
    module = _load_partition_script()
    config = module.VERSION_CONFIGS["successor_v6"]  # type: ignore[attr-defined]
    development = module._assignment_payload(  # type: ignore[attr-defined]
        split=DEVELOPMENT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=11,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )
    holdout = module._assignment_payload(  # type: ignore[attr-defined]
        split=HOLDOUT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=11,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )

    assert development["source_text_quality_policy"] == (holdout["source_text_quality_policy"])
    assert development["partition_id"] == "successor_v6_source_pool"
    assert [path.name for path in config["retired_partition_dirs"]] == [
        "successor_v4",
        "successor_v5",
    ]


def test_v7_retires_v4_through_v6_and_freezes_boundary_policy() -> None:
    module = _load_partition_script()
    config = module.VERSION_CONFIGS["successor_v7"]  # type: ignore[attr-defined]
    development = module._assignment_payload(  # type: ignore[attr-defined]
        split=DEVELOPMENT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=13,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )
    holdout = module._assignment_payload(  # type: ignore[attr-defined]
        split=HOLDOUT_SPLIT,
        source_revision="revision",
        assignments=[],
        random_state=13,
        partition_id=str(config["partition_id"]),
        text_quality_policy=str(config["text_quality_policy"]),
    )

    assert development["source_text_quality_policy"] == (holdout["source_text_quality_policy"])
    assert development["partition_id"] == "successor_v7_source_pool"
    assert [path.name for path in config["retired_partition_dirs"]] == [
        "successor_v4",
        "successor_v5",
        "successor_v6",
    ]


def test_v9_opts_into_contextual_extraction_without_changing_prior_versions() -> None:
    module = _load_partition_script()

    assert module.VERSION_CONFIGS["successor_v8"][  # type: ignore[attr-defined]
        "evidence_extraction_policy"
    ] == "action_prefix_v1"
    assert module.VERSION_CONFIGS["successor_v9"][  # type: ignore[attr-defined]
        "evidence_extraction_policy"
    ] == "contextual_action_v2"


def test_v8_batch_reuses_v6_holdout_without_creating_another() -> None:
    module = _load_partition_script()
    config = module.VERSION_CONFIGS[  # type: ignore[attr-defined]
        "successor_v8"
    ]

    assert config["development_resumes_per_family"] == 1
    assert config["holdout_resumes_per_family"] == 0
    assert config["existing_holdout_partition_id"] == ("successor_v6_source_pool")
    assert [path.name for path in config["retired_partition_dirs"]] == [
        "successor_v4",
        "successor_v5",
        "successor_v6",
        "successor_v7",
    ]


def test_v9_freezes_small_development_and_holdout_together() -> None:
    module = _load_partition_script()
    config = module.VERSION_CONFIGS[  # type: ignore[attr-defined]
        "successor_v9"
    ]

    assert config["partition_id"] == "successor_v9_source_pool"
    assert config["development_resumes_per_family"] == 3
    assert config["holdout_resumes_per_family"] == 3
    assert config["existing_holdout_partition_id"] is None
    assert config["include_ats_profiles"] is True
    assert [path.name for path in config["retired_partition_dirs"]] == [
        "successor_v4",
        "successor_v5",
        "successor_v6",
        "successor_v7",
        "successor_v8",
    ]


def test_retired_development_only_partition_excludes_sources_without_holdout(
    tmp_path: Path,
) -> None:
    module = _load_partition_script()
    partition_dir = tmp_path / "successor_v8"
    partition_dir.mkdir()
    development = [
        {
            "allocation_id": "development-only-allocation",
            "split": DEVELOPMENT_SPLIT,
            "role_family": "Data",
            "construction_stratum": "explicit_aligned",
            "source_resume_hash": "resume-development-only",
            "source_job_hash": "job-development-only",
        }
    ]
    payload = {
        "content_included": False,
        "labels_included": False,
        "model_signals_included": False,
        "assignments": development,
    }
    (partition_dir / "development_source_assignments.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (partition_dir / "partition_manifest.json").write_text(
        json.dumps(
            {
                "partition_id": "successor_v8_development",
                "new_holdout_created": False,
                "holdout_content_opened": False,
                "holdout_labels_opened": False,
                "development_assignment_sha256": assignment_sha256(development),
            }
        ),
        encoding="utf-8",
    )

    jobs, resumes, summary = module._retired_source_hashes(  # type: ignore[attr-defined]
        (partition_dir,)
    )

    assert jobs == {"job-development-only"}
    assert resumes == {"resume-development-only"}
    assert summary["retired_partition_ids"] == ["successor_v8_development"]
