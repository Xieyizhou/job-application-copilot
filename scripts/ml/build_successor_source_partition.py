"""Partition new sources into development and a sealed operational holdout."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
from typing import Any, TypedDict


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ml" / "annotations"
TRAINING_DIR = PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v4_batch4"
DJINNI_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni"
ATS_PROFILE_DIR = PROJECT_ROOT / "data" / "ml" / "raw" / "resume_ats_score_v1_en"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ml" / "source_partitions" / "successor_v4"
SOURCE_PARTITION_DIR = PROJECT_ROOT / "data" / "ml" / "source_partitions"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.real_development import ROLE_FAMILIES  # noqa: E402
from ml.real_development import (  # noqa: E402
    CONTEXTUAL_EVIDENCE_EXTRACTION,
    LEGACY_EVIDENCE_EXTRACTION,
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
from ml.source_pool_partition import (  # noqa: E402
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
from ml.source_text_quality import (  # noqa: E402
    LEGACY_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    source_text_quality_manifest,
)


class VersionConfig(TypedDict):
    partition_id: str
    output_dir: Path
    random_state: int
    text_quality_policy: str
    retired_partition_dirs: tuple[Path, ...]
    development_resumes_per_family: int
    holdout_resumes_per_family: int
    existing_holdout_partition_id: str | None
    include_ats_profiles: bool
    evidence_extraction_policy: str


VERSION_CONFIGS: dict[str, VersionConfig] = {
    "successor_v4": {
        "partition_id": "successor_v4_source_pool",
        "output_dir": DEFAULT_OUTPUT_DIR,
        "random_state": 20260801,
        "text_quality_policy": LEGACY_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (),
        "development_resumes_per_family": 6,
        "holdout_resumes_per_family": 6,
        "existing_holdout_partition_id": None,
        "include_ats_profiles": False,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
    },
    "successor_v5": {
        "partition_id": "successor_v5_source_pool",
        "output_dir": SOURCE_PARTITION_DIR / "successor_v5",
        "random_state": 20260802,
        "text_quality_policy": SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (SOURCE_PARTITION_DIR / "successor_v4",),
        "development_resumes_per_family": 6,
        "holdout_resumes_per_family": 6,
        "existing_holdout_partition_id": None,
        "include_ats_profiles": False,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
    },
    "successor_v6": {
        "partition_id": "successor_v6_source_pool",
        "output_dir": SOURCE_PARTITION_DIR / "successor_v6",
        "random_state": 20260803,
        "text_quality_policy": SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (
            SOURCE_PARTITION_DIR / "successor_v4",
            SOURCE_PARTITION_DIR / "successor_v5",
        ),
        "development_resumes_per_family": 6,
        "holdout_resumes_per_family": 6,
        "existing_holdout_partition_id": None,
        "include_ats_profiles": False,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
    },
    "successor_v7": {
        "partition_id": "successor_v7_source_pool",
        "output_dir": SOURCE_PARTITION_DIR / "successor_v7",
        "random_state": 20260804,
        "text_quality_policy": SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (
            SOURCE_PARTITION_DIR / "successor_v4",
            SOURCE_PARTITION_DIR / "successor_v5",
            SOURCE_PARTITION_DIR / "successor_v6",
        ),
        "development_resumes_per_family": 6,
        "holdout_resumes_per_family": 6,
        "existing_holdout_partition_id": None,
        "include_ats_profiles": False,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
    },
    "successor_v8": {
        "partition_id": "successor_v8_development",
        "output_dir": SOURCE_PARTITION_DIR / "successor_v8",
        "random_state": 20260805,
        "text_quality_policy": SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (
            SOURCE_PARTITION_DIR / "successor_v4",
            SOURCE_PARTITION_DIR / "successor_v5",
            SOURCE_PARTITION_DIR / "successor_v6",
            SOURCE_PARTITION_DIR / "successor_v7",
        ),
        "development_resumes_per_family": 1,
        "holdout_resumes_per_family": 0,
        "existing_holdout_partition_id": "successor_v6_source_pool",
        "include_ats_profiles": True,
        "evidence_extraction_policy": LEGACY_EVIDENCE_EXTRACTION,
    },
    "successor_v9": {
        "partition_id": "successor_v9_source_pool",
        "output_dir": SOURCE_PARTITION_DIR / "successor_v9",
        "random_state": 20260806,
        "text_quality_policy": SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        "retired_partition_dirs": (
            SOURCE_PARTITION_DIR / "successor_v4",
            SOURCE_PARTITION_DIR / "successor_v5",
            SOURCE_PARTITION_DIR / "successor_v6",
            SOURCE_PARTITION_DIR / "successor_v7",
            SOURCE_PARTITION_DIR / "successor_v8",
        ),
        "development_resumes_per_family": 3,
        "holdout_resumes_per_family": 3,
        "existing_holdout_partition_id": None,
        "include_ats_profiles": True,
        "evidence_extraction_policy": CONTEXTUAL_EVIDENCE_EXTRACTION,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version",
        choices=tuple(VERSION_CONFIGS),
        default="successor_v4",
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--development-resumes-per-family",
        type=int,
    )
    parser.add_argument(
        "--holdout-resumes-per-family",
        type=int,
    )
    parser.add_argument("--random-state", type=int)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Write the content-free capacity report without creating assignments.",
    )
    return parser.parse_args()


def _consumed_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = [
        row
        for path in ANNOTATION_DIR.rglob("*.jsonl")
        for row in load_jsonl(path)
        if row.get("requirement") or row.get("source_job_hash")
    ]
    tasks.extend(load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl"))
    for path in (PROJECT_ROOT / "data" / "ml" / "shadow").glob("real_batch*/[0-9][0-9]_*.json"):
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


def _assignment_payload(
    *,
    split: str,
    source_revision: str,
    assignments: list[dict[str, str]],
    random_state: int,
    partition_id: str,
    text_quality_policy: str,
    evidence_extraction_policy: str = LEGACY_EVIDENCE_EXTRACTION,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "partition_id": partition_id,
        "split": split,
        "source_dataset_revision": source_revision,
        "construction_policy": list(SOURCE_POOL_CONSTRUCTION_STRATA),
        "random_state": random_state,
        "source_text_quality_policy": text_quality_policy,
        "evidence_extraction_policy": evidence_extraction_policy,
        "content_included": False,
        "labels_included": False,
        "model_signals_included": False,
        "assignments": assignments,
    }


def _retired_source_hashes(
    partition_dirs: tuple[Path, ...],
) -> tuple[set[str], set[str], dict[str, Any]]:
    job_hashes: set[str] = set()
    resume_hashes: set[str] = set()
    partition_ids: list[str] = []
    assignment_files = (
        (
            "development_source_assignments.json",
            "development_assignment_sha256",
        ),
        (
            "sealed_holdout_source_assignments.json",
            "sealed_holdout_assignment_sha256",
        ),
    )
    for directory in partition_dirs:
        manifest = json.loads((directory / "partition_manifest.json").read_text(encoding="utf-8"))
        partition_ids.append(str(manifest["partition_id"]))
        if (
            manifest.get("holdout_content_opened") is not False
            or manifest.get("holdout_labels_opened") is not False
        ):
            raise ValueError("A retired partition has opened holdout content.")
        files_to_check = list(assignment_files)
        if manifest.get("new_holdout_created") is False:
            files_to_check = files_to_check[:1]
        for filename, checksum_field in files_to_check:
            payload = json.loads((directory / filename).read_text(encoding="utf-8"))
            if any(
                bool(payload.get(field))
                for field in (
                    "content_included",
                    "labels_included",
                    "model_signals_included",
                )
            ):
                raise ValueError("Retired source assignment exposes restricted data.")
            assignments = payload.get("assignments")
            if not isinstance(assignments, list) or any(
                not isinstance(row, dict) or set(row) != ASSIGNMENT_FIELDS for row in assignments
            ):
                raise ValueError("Retired source assignment is malformed.")
            if assignment_sha256(assignments) != str(manifest.get(checksum_field, "")):
                raise ValueError("Retired source assignment checksum changed.")
            job_hashes.update(str(row["source_job_hash"]) for row in assignments)
            resume_hashes.update(str(row["source_resume_hash"]) for row in assignments)
    return (
        job_hashes,
        resume_hashes,
        {
            "retired_partition_ids": partition_ids,
            "retired_job_hashes_excluded": len(job_hashes),
            "retired_resume_hashes_excluded": len(resume_hashes),
            "retired_holdout_assignment_content_opened": False,
            "retired_holdout_labels_opened": False,
        },
    )


def _write_preflight_report(
    *,
    version: str,
    partition_id: str,
    source_revision: str,
    profile_sources: list[str],
    preflight: dict[str, Any],
    isolation_summary: dict[str, Any],
    retired_summary: dict[str, Any],
    evidence_extraction_policy: str,
) -> Path:
    report_dir = PROJECT_ROOT / "reports" / "ml" / "generated"
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / f"source_pool_preflight_{version}.json"
    payload = {
        "schema_version": 1,
        "version": version,
        "partition_id": partition_id,
        "source_dataset_revision": source_revision,
        "profile_sources": profile_sources,
        "evidence_extraction": evidence_extraction_manifest(
            evidence_extraction_policy
        ),
        "assignments_created": False,
        "reviewer_packets_generated": False,
        "product_integration_allowed": False,
        "demo_integration_allowed": False,
        "preflight": preflight,
        "isolation": isolation_summary,
        "retired_sources": retired_summary,
    }
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def main() -> None:
    args = parse_args()
    version_config = VERSION_CONFIGS[args.version]
    output_dir = args.output_dir or Path(version_config["output_dir"])
    random_state = (
        args.random_state if args.random_state is not None else int(version_config["random_state"])
    )
    partition_id = str(version_config["partition_id"])
    text_quality_policy = str(version_config["text_quality_policy"])
    evidence_extraction_policy = str(
        version_config["evidence_extraction_policy"]
    )
    retired_partition_dirs = tuple(Path(path) for path in version_config["retired_partition_dirs"])
    development_resumes_per_family = (
        args.development_resumes_per_family
        if args.development_resumes_per_family is not None
        else int(version_config["development_resumes_per_family"])
    )
    holdout_resumes_per_family = (
        args.holdout_resumes_per_family
        if args.holdout_resumes_per_family is not None
        else int(version_config["holdout_resumes_per_family"])
    )
    development_only = holdout_resumes_per_family == 0
    paths = {
        "development": (output_dir / "development_source_assignments.json"),
        "manifest": output_dir / "partition_manifest.json",
    }
    if not development_only:
        paths["holdout"] = output_dir / "sealed_holdout_source_assignments.json"
    if any(path.exists() for path in paths.values()):
        raise SystemExit("Source partition exists; refusing overwrite.")
    source_manifest = DJINNI_DIR / "manifest.json"
    include_ats_profiles = bool(version_config["include_ats_profiles"])
    source_revision = development_source_revision(
        source_manifest,
        ats_dataset_dir=ATS_PROFILE_DIR if include_ats_profiles else None,
    )
    excluded_tasks, excluded_pairs = _consumed_data()
    isolation_index = build_development_isolation_index(
        excluded_tasks,
        excluded_pairs,
    )
    retired_jobs, retired_resumes, retired_summary = _retired_source_hashes(retired_partition_dirs)
    excluded_jobs = set(isolation_index.job_hashes) | retired_jobs
    excluded_resumes = set(isolation_index.resume_hashes) | retired_resumes
    jobs = load_djinni_requirements(
        DJINNI_DIR / "raw" / "djinni_jobs.parquet",
        excluded_jobs,
        random_state=random_state,
        family_limit=800,
        text_quality_policy=text_quality_policy,
    )
    profiles = load_djinni_profiles(
        DJINNI_DIR / "raw" / "djinni_profiles.parquet",
        excluded_resumes,
        random_state=random_state,
        family_limit=800,
        include_supplemental_profile_text=True,
        prefer_specialized_role=True,
        minimum_evidence=4,
        text_quality_policy=text_quality_policy,
        evidence_extraction_policy=evidence_extraction_policy,
    )
    if include_ats_profiles:
        ats_profiles = load_ats_profiles(
            ATS_PROFILE_DIR,
            excluded_resumes,
            random_state=random_state,
            family_limit=800,
            minimum_evidence=4,
            text_quality_policy=text_quality_policy,
            evidence_extraction_policy=evidence_extraction_policy,
        )
        profiles = {
            family: [
                *profiles.get(family, []),
                *ats_profiles.get(family, []),
            ]
            for family in ROLE_FAMILIES
        }
    profile_sources = (
        ["djinni_profile", "resume_ats_profile"]
        if include_ats_profiles
        else ["djinni_profile"]
    )
    preflight = source_pool_preflight(
        profiles,
        jobs,
        isolation_index=isolation_index,
        development_resumes_per_family=development_resumes_per_family,
        holdout_resumes_per_family=holdout_resumes_per_family,
    )
    preflight_path = _write_preflight_report(
        version=args.version,
        partition_id=partition_id,
        source_revision=source_revision,
        profile_sources=profile_sources,
        preflight=preflight,
        isolation_summary=isolation_index_summary(isolation_index),
        retired_summary=retired_summary,
        evidence_extraction_policy=evidence_extraction_policy,
    )
    if not bool(preflight["preflight_passed"]):
        shortages = [
            (
                f"{family} profiles "
                f"{row['isolated_profiles_found']}/"
                f"{row['required_profiles']}"
            )
            for family, row in preflight["role_families"].items()
            if not bool(row["necessary_capacity_passed"])
        ]
        raise SystemExit(
            "Source-pool preflight failed before assignment: "
            + ", ".join(shortages)
            + f". Content-free report: {preflight_path}"
        )
    if args.preflight_only:
        print(f"Source-pool preflight passed: {preflight_path}")
        return
    development, holdout = build_source_partition(
        profiles,
        jobs,
        isolation_index=isolation_index,
        development_resumes_per_family=development_resumes_per_family,
        holdout_resumes_per_family=holdout_resumes_per_family,
        random_state=random_state,
    )
    contract = validate_source_partition(
        development,
        holdout,
        development_resumes_per_family=development_resumes_per_family,
        holdout_resumes_per_family=holdout_resumes_per_family,
    )
    development_payload = _assignment_payload(
        split=DEVELOPMENT_SPLIT,
        source_revision=source_revision,
        assignments=development,
        random_state=random_state,
        partition_id=partition_id,
        text_quality_policy=text_quality_policy,
        evidence_extraction_policy=evidence_extraction_policy,
    )
    holdout_payload = (
        {
            **_assignment_payload(
                split=HOLDOUT_SPLIT,
                source_revision=source_revision,
                assignments=holdout,
                random_state=random_state,
                partition_id=partition_id,
                text_quality_policy=text_quality_policy,
                evidence_extraction_policy=evidence_extraction_policy,
            ),
            "sealed": True,
            "open_before_model_freeze": False,
            "evaluation_runs_allowed": 1,
        }
        if holdout
        else None
    )
    development_checksum = assignment_sha256(development)
    holdout_checksum = assignment_sha256(holdout) if holdout else None
    manifest = {
        "schema_version": 1,
        "partition_id": partition_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dataset_revision": source_revision,
        "source_scope": (
            "unused public Djinni jobs plus Djinni and ATS profiles"
            if include_ats_profiles
            else "unused public Djinni profiles and jobs"
        ),
        "profile_sources": profile_sources,
        "role_families": list(ROLE_FAMILIES),
        "construction_policy": list(SOURCE_POOL_CONSTRUCTION_STRATA),
        "source_text_quality": source_text_quality_manifest(text_quality_policy),
        "evidence_extraction": evidence_extraction_manifest(
            evidence_extraction_policy
        ),
        "development_resumes_per_family": development_resumes_per_family,
        "holdout_resumes_per_family": holdout_resumes_per_family,
        "development_assignment_sha256": development_checksum,
        "sealed_holdout_assignment_sha256": holdout_checksum,
        "new_holdout_created": not development_only,
        "existing_holdout_partition_id": version_config["existing_holdout_partition_id"],
        "holdout_content_opened": False,
        "holdout_labels_opened": False,
        "holdout_diagnostics_opened": False,
        "holdout_evaluation_runs": 0,
        "holdout_evaluation_runs_allowed": 1,
        "source_assignment_frozen": True,
        "post_label_source_movement_allowed": False,
        "candidate_model_predictions_used": False,
        "model_disagreements_used_for_sampling": False,
        "frozen_reserve_labels_read": False,
        "shadow_diagnostics_used_for_sampling": False,
        "product_integration_allowed": False,
        "demo_integration_allowed": False,
        "reviewer_packets_generated": False,
        "development_label_minimums_per_role": {
            "Direct": 4,
            "Partial": 8,
            "No Support": 6,
        },
        **contract,
        **isolation_index_summary(isolation_index),
        **retired_summary,
        "status": (
            "development_assignment_frozen_before_packet_generation"
            if development_only
            else "source_partition_sealed_before_packet_generation"
        ),
    }
    all_rows = [*development, *holdout]
    if any(
        str(row["source_job_hash"]) in retired_jobs
        or str(row["source_resume_hash"]) in retired_resumes
        for row in all_rows
    ):
        raise ValueError("New source partition reuses a retired source.")
    output_dir.mkdir(parents=True, exist_ok=False)
    paths["development"].write_text(
        json.dumps(development_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if holdout_payload is not None:
        paths["holdout"].write_text(
            json.dumps(holdout_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for path in paths.values():
        path.chmod(0o600)
    print(f"Development resume groups: {manifest['development_resume_groups']}")
    print(f"Sealed holdout resume groups: {manifest['holdout_resume_groups']}")
    print(
        "Assignments: "
        f"{manifest['development_assignments']} development, "
        f"{manifest['holdout_assignments']} sealed holdout"
    )
    print("Cross-split source/text overlap: 0")
    print("Reviewer packets generated: False")
    print(f"Text quality policy: {text_quality_policy}")
    print(f"Retired source partitions excluded: {len(retired_partition_dirs)}")
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
