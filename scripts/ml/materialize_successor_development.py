"""Materialize the frozen successor-development assignment for blind review."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ml" / "annotations"
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DJINNI_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni"
ATS_PROFILE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "raw" / "resume_ats_score_v1_en"
)
DEFAULT_PARTITION_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "source_partitions"
    / "successor_v4"
)
DEFAULT_ASSIGNMENT_PATH = (
    DEFAULT_PARTITION_DIR / "development_source_assignments.json"
)
DEFAULT_OUTPUT_DIR = ANNOTATION_DIR / "successor_development_v4"
SOURCE_PARTITION_DIR = PROJECT_ROOT / "data" / "ml" / "source_partitions"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.operational_development_v3 import (  # noqa: E402
    build_single_reviewer_packet,
)
from ml.real_development import (  # noqa: E402
    CONTEXTUAL_EVIDENCE_EXTRACTION,
    LEGACY_EVIDENCE_EXTRACTION,
    ROLE_FAMILIES,
    evidence_extraction_manifest,
)
from ml.real_development_isolation import (  # noqa: E402
    build_development_isolation_index,
)
from ml.real_development_sources import (  # noqa: E402
    development_source_revision,
    load_ats_profiles,
    load_djinni_profiles,
    load_djinni_requirements,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402
from ml.source_pool_partition import isolation_index_summary  # noqa: E402
from ml.source_text_quality import (  # noqa: E402
    LEGACY_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    audit_materialized_text_quality,
)
from ml.successor_development_v4 import (  # noqa: E402
    PARTITION_MANIFEST_FILENAME,
    assert_development_assignment_path,
    materialize_development_tasks,
    validate_development_assignment_input,
    validate_materialized_development,
    validate_runtime_isolation,
)


class VersionConfig(TypedDict):
    assignment_path: Path
    output_dir: Path
    dataset_id: str
    partition_id: str
    source_dataset: str
    text_quality_policy: str | None
    evidence_extraction_policy: str
    generate_reviewer_packet: bool
    include_ats_profiles: bool


VERSION_CONFIGS: dict[str, VersionConfig] = {
    "successor_v4": {
        "assignment_path": DEFAULT_ASSIGNMENT_PATH,
        "output_dir": DEFAULT_OUTPUT_DIR,
        "dataset_id": "successor_development_v4",
        "partition_id": "successor_v4_source_pool",
        "source_dataset": "djinni_successor_development_v4",
        "text_quality_policy": None,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": True,
        "include_ats_profiles": False,
    },
    "successor_v5": {
        "assignment_path": (
            SOURCE_PARTITION_DIR
            / "successor_v5"
            / "development_source_assignments.json"
        ),
        "output_dir": ANNOTATION_DIR / "successor_development_v5",
        "dataset_id": "successor_development_v5",
        "partition_id": "successor_v5_source_pool",
        "source_dataset": "djinni_successor_development_v5",
        "text_quality_policy": SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": False,
        "include_ats_profiles": False,
    },
    "successor_v6": {
        "assignment_path": (
            SOURCE_PARTITION_DIR
            / "successor_v6"
            / "development_source_assignments.json"
        ),
        "output_dir": ANNOTATION_DIR / "successor_development_v6",
        "dataset_id": "successor_development_v6",
        "partition_id": "successor_v6_source_pool",
        "source_dataset": "djinni_successor_development_v6",
        "text_quality_policy": SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": False,
        "include_ats_profiles": False,
    },
    "successor_v7": {
        "assignment_path": (
            SOURCE_PARTITION_DIR
            / "successor_v7"
            / "development_source_assignments.json"
        ),
        "output_dir": ANNOTATION_DIR / "successor_development_v7",
        "dataset_id": "successor_development_v7",
        "partition_id": "successor_v7_source_pool",
        "source_dataset": "djinni_successor_development_v7",
        "text_quality_policy": SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": False,
        "include_ats_profiles": False,
    },
    "successor_v8": {
        "assignment_path": (
            SOURCE_PARTITION_DIR
            / "successor_v8"
            / "development_source_assignments.json"
        ),
        "output_dir": ANNOTATION_DIR / "successor_development_v8",
        "dataset_id": "successor_development_v8",
        "partition_id": "successor_v8_development",
        "source_dataset": "djinni_successor_development_v8",
        "text_quality_policy": SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": False,
        "include_ats_profiles": True,
    },
    "successor_v9": {
        "assignment_path": (
            SOURCE_PARTITION_DIR
            / "successor_v9"
            / "development_source_assignments.json"
        ),
        "output_dir": ANNOTATION_DIR / "successor_development_v9",
        "dataset_id": "successor_development_v9",
        "partition_id": "successor_v9_source_pool",
        "source_dataset": "djinni_ats_successor_development_v9",
        "text_quality_policy": SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        "evidence_extraction_policy": CONTEXTUAL_EVIDENCE_EXTRACTION,
        "generate_reviewer_packet": False,
        "include_ats_profiles": True,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=tuple(VERSION_CONFIGS),
        default="successor_v4",
    )
    parser.add_argument(
        "--assignment-path",
        type=Path,
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--hidden-repeats", type=int, default=14)
    return parser.parse_args()


def _consumed_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [
        row
        for path in ANNOTATION_DIR.rglob("*.jsonl")
        for row in load_jsonl(path)
        if row.get("requirement") or row.get("source_job_hash")
    ]
    tasks.extend(load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl"))
    for path in (
        PROJECT_ROOT / "data" / "ml" / "shadow"
    ).glob("real_batch*/[0-9][0-9]_*.json"):
        report = json.loads(path.read_text(encoding="utf-8"))
        tasks.append(
            {
                "source_job_hash": report["source_job_hash"],
                "source_resume_hash": "",
                "requirement": "",
                "candidates": [],
            }
        )
    return tasks, load_jsonl(TRAINING_DIR / "training_pairs.jsonl")


def _flatten_by_hash(
    pools: dict[str, list[dict[str, Any]]],
) -> dict[str, dict[str, Any]]:
    return {
        str(row["source_hash"]): row
        for rows in pools.values()
        for row in rows
    }


def main() -> None:
    args = parse_args()
    version_config = VERSION_CONFIGS[args.version]
    assignment_path = assert_development_assignment_path(
        args.assignment_path or version_config["assignment_path"]
    )
    output_dir = args.output_dir or version_config["output_dir"]
    partition_manifest_path = (
        assignment_path.parent / PARTITION_MANIFEST_FILENAME
    )
    output_paths = {
        "queue": output_dir / "queue.jsonl",
        "manifest": output_dir / "manifest.json",
    }
    if version_config["generate_reviewer_packet"]:
        output_paths.update(
            {
                "reviewer": output_dir / "reviewer_queue.jsonl",
                "repeat_map": output_dir / "repeat_map.json",
            }
        )
    else:
        output_paths["quality_audit"] = (
            output_dir / "automated_quality_audit.json"
        )
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("Development packet exists; refusing overwrite.")

    partition_manifest = json.loads(
        partition_manifest_path.read_text(encoding="utf-8")
    )
    assignment_payload = json.loads(
        assignment_path.read_text(encoding="utf-8")
    )
    assignment_contract = validate_development_assignment_input(
        assignment_path,
        assignment_payload,
        partition_manifest,
        expected_partition_id=version_config["partition_id"],
        expected_text_quality_policy=version_config["text_quality_policy"],
        expected_evidence_extraction_policy=version_config[
            "evidence_extraction_policy"
        ],
    )
    source_manifest_path = DJINNI_DIR / "manifest.json"
    include_ats_profiles = bool(version_config["include_ats_profiles"])
    source_revision = development_source_revision(
        source_manifest_path,
        ats_dataset_dir=ATS_PROFILE_DIR if include_ats_profiles else None,
    )
    excluded_tasks, excluded_pairs = _consumed_data()
    isolation_index = build_development_isolation_index(
        excluded_tasks,
        excluded_pairs,
    )
    isolation_summary = isolation_index_summary(isolation_index)
    validate_runtime_isolation(
        partition_manifest,
        source_dataset_revision=source_revision,
        isolation_summary=isolation_summary,
    )

    random_state = int(assignment_contract["random_state"])
    assignments = [
        dict(row) for row in assignment_payload["assignments"]
    ]
    assigned_job_hashes = {
        str(row["source_job_hash"]) for row in assignments
    }
    assigned_resume_hashes = {
        str(row["source_resume_hash"]) for row in assignments
    }
    jobs = load_djinni_requirements(
        DJINNI_DIR / "raw" / "djinni_jobs.parquet",
        set(isolation_index.job_hashes),
        random_state=random_state,
        family_limit=800,
        text_quality_policy=(
            version_config["text_quality_policy"]
            or LEGACY_SOURCE_TEXT_QUALITY
        ),
        included_job_hashes=assigned_job_hashes,
    )
    profiles = load_djinni_profiles(
        DJINNI_DIR / "raw" / "djinni_profiles.parquet",
        set(isolation_index.resume_hashes),
        random_state=random_state,
        family_limit=800,
        include_supplemental_profile_text=True,
        prefer_specialized_role=True,
        minimum_evidence=4,
        text_quality_policy=(
            version_config["text_quality_policy"]
            or LEGACY_SOURCE_TEXT_QUALITY
        ),
        included_resume_hashes=assigned_resume_hashes,
        evidence_extraction_policy=version_config[
            "evidence_extraction_policy"
        ],
    )
    if include_ats_profiles:
        ats_profiles = load_ats_profiles(
            ATS_PROFILE_DIR,
            set(isolation_index.resume_hashes),
            random_state=random_state,
            family_limit=800,
            minimum_evidence=4,
            text_quality_policy=(
                version_config["text_quality_policy"]
                or LEGACY_SOURCE_TEXT_QUALITY
            ),
            included_resume_hashes=assigned_resume_hashes,
            evidence_extraction_policy=version_config[
                "evidence_extraction_policy"
            ],
        )
        profiles = {
            family: [
                *profiles.get(family, []),
                *ats_profiles.get(family, []),
            ]
            for family in ROLE_FAMILIES
        }
    tasks = materialize_development_tasks(
        assignments,
        profiles_by_hash=_flatten_by_hash(profiles),
        jobs_by_hash=_flatten_by_hash(jobs),
        source_dataset_revision=source_revision,
        isolation_index=isolation_index,
        partition_id=version_config["partition_id"],
        source_dataset=version_config["source_dataset"],
    )
    task_contract = validate_materialized_development(tasks, assignments)
    quality_audit = audit_materialized_text_quality(
        tasks,
        policy=(
            version_config["text_quality_policy"]
            or LEGACY_SOURCE_TEXT_QUALITY
        ),
    )
    reviewer_queue: list[dict[str, Any]] = []
    repeat_map: list[dict[str, str]] = []
    if version_config["generate_reviewer_packet"]:
        reviewer_queue, repeat_map = build_single_reviewer_packet(
            tasks,
            repeat_count=args.hidden_repeats,
            random_state=random_state,
            presentation_prefix="successor-v4-presentation",
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    write_queue(tasks, output_paths["queue"])
    if version_config["generate_reviewer_packet"]:
        write_queue(reviewer_queue, output_paths["reviewer"])
        output_paths["repeat_map"].write_text(
            json.dumps(repeat_map, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    else:
        output_paths["quality_audit"].write_text(
            json.dumps(quality_audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    manifest = {
        "schema_version": 1,
        "dataset_id": version_config["dataset_id"],
        "dataset_role": "successor_training_and_nested_development_only",
        "partition_id": assignment_contract["partition_id"],
        "source_dataset_revision": source_revision,
        "development_assignment_sha256": assignment_contract[
            "assignment_sha256"
        ],
        **task_contract,
        "review_presentations": len(reviewer_queue),
        "hidden_repeats": len(repeat_map),
        "task_source_counts": dict(
            Counter(str(task["source_dataset"]) for task in tasks)
        ),
        "task_sha256": sha256_bytes(jsonl_bytes(tasks)),
        "reviewer_packet_sha256": (
            sha256_bytes(jsonl_bytes(reviewer_queue))
            if reviewer_queue
            else None
        ),
        "repeat_map_sha256": (
            sha256_bytes(
                json.dumps(repeat_map, sort_keys=True).encode()
            )
            if repeat_map
            else None
        ),
        "automated_quality_gate_passed": quality_audit[
            "automated_quality_gate_passed"
        ],
        "automated_quality_audit_sha256": sha256_bytes(
            json.dumps(quality_audit, sort_keys=True).encode()
        ),
        "source_text_quality_policy": (
            version_config["text_quality_policy"]
            or LEGACY_SOURCE_TEXT_QUALITY
        ),
        "source_text_quality_policy_sha256": (
            partition_manifest.get("source_text_quality", {}).get(
                "policy_sha256"
            )
        ),
        "evidence_extraction_policy": version_config[
            "evidence_extraction_policy"
        ],
        "evidence_extraction_policy_sha256": evidence_extraction_manifest(
            version_config["evidence_extraction_policy"]
        )["policy_sha256"],
        "manual_quality_gate_passed": False,
        **isolation_summary,
        "source_assignment_frozen": True,
        "labels_included": False,
        "candidate_model_predictions_used": False,
        "model_disagreements_used_for_sampling": False,
        "frozen_reserve_labels_read": False,
        "shadow_diagnostics_used_for_sampling": False,
        "holdout_assignment_opened": False,
        "holdout_content_opened": False,
        "holdout_labels_opened": False,
        "holdout_reviewer_packet_generated": False,
        "product_integration_allowed": False,
        "demo_integration_allowed": False,
        "post_label_source_movement_allowed": False,
        "annotation_allowed": False,
        "reviewer_packet_generated": bool(reviewer_queue),
        "status": (
            "blocked_by_automated_text_quality_audit"
            if not quality_audit["automated_quality_gate_passed"]
            else (
                "awaiting_pre_review_quality_audit"
                if reviewer_queue
                else "awaiting_manual_stratified_quality_audit"
            )
        ),
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in output_paths.values():
        path.chmod(0o600)
    print(f"Development tasks: {manifest['tasks']}")
    print(f"Resume groups: {manifest['resume_groups']}")
    print(f"Review presentations: {manifest['review_presentations']}")
    print(
        "Automated quality gate passed: "
        f"{manifest['automated_quality_gate_passed']}"
    )
    print(
        "Reviewer packet generated: "
        f"{manifest['reviewer_packet_generated']}"
    )
    print("Holdout assignment opened: False")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
