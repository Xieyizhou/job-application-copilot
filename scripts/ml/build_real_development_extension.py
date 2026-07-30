"""Build a source-isolated real-text development extension and blind packets."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DJINNI_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni" / "raw"
SKILLSPAN_DIR = (
    PROJECT_ROOT / "data" / "ml" / "external" / "skillspan" / "data" / "json"
)
ATS_PROFILE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "raw" / "resume_ats_score_v1_en"
)
OLD_REAL_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
TRAINING_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v2"
)
DEVELOPMENT_V2_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v2"
)
DEVELOPMENT_V3_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v3"
)
ACCEPTANCE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "acceptance_supplement_v1"
)
RESERVE_V2_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v2"
)
RESERVE_V3_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.real_development_isolation import (  # noqa: E402
    assert_development_isolated,
    build_development_isolation_index,
)
from ml.real_development_sampling import (  # noqa: E402
    balanced_reviewer_queue,
    build_stratified_tasks,
)
from ml.real_development_sources import (  # noqa: E402
    load_ats_profiles,
    load_djinni_profiles,
    load_djinni_requirements,
    load_skillspan_requirements,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402
from ml.real_review import (  # noqa: E402
    reviewer_packet_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--dataset-version",
        type=int,
        choices=(2, 3, 4),
        default=2,
    )
    parser.add_argument(
        "--dataset-role",
        choices=("development", "reserve"),
        default="development",
    )
    parser.add_argument("--tasks-per-family", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=20260724)
    parser.add_argument(
        "--candidate-path",
        type=Path,
        help="Optional precommitted candidate artifact; only its byte hash is read.",
    )
    return parser.parse_args()


def _load_excluded(
    *,
    dataset_version: int,
    dataset_role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    for name in (
        "real_holdout_gold_v1.jsonl",
        "real_validation_gold_v1.jsonl",
        "real_reserve_gold_v1.jsonl",
    ):
        tasks.extend(load_jsonl(OLD_REAL_DIR / name))
    if dataset_version >= 3:
        development_gold = DEVELOPMENT_V2_DIR / "real_development_gold_v2.jsonl"
        development_queue = (
            DEVELOPMENT_V2_DIR / "real_development_queue_v2.jsonl"
        )
        tasks.extend(
            load_jsonl(
                development_gold
                if development_gold.exists()
                else development_queue
            )
        )
    if dataset_role == "reserve":
        for directory, gold_name, queue_name in (
            (
                DEVELOPMENT_V2_DIR,
                "real_development_gold_v2.jsonl",
                "real_development_queue_v2.jsonl",
            ),
            (
                DEVELOPMENT_V3_DIR,
                "real_development_gold_v3.jsonl",
                "real_development_queue_v3.jsonl",
            ),
            (
                ACCEPTANCE_DIR,
                "acceptance_supplement_gold_v1.jsonl",
                "acceptance_supplement_queue_v1.jsonl",
            ),
            (
                RESERVE_V2_DIR,
                "real_reserve_gold_v2.jsonl",
                "real_reserve_queue_v2.jsonl",
            ),
            (
                RESERVE_V3_DIR,
                "real_reserve_gold_v3.jsonl",
                "real_reserve_queue_v3.jsonl",
            ),
        ):
            gold = directory / gold_name
            queue = directory / queue_name
            tasks.extend(load_jsonl(gold if gold.exists() else queue))
    return tasks, pairs


def _write_outputs(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    construction: list[dict[str, str]],
    overlap_checks: dict[str, int],
    *,
    dataset_version: int,
    dataset_role: str,
    random_state: int,
    candidate_path: Path | None,
) -> None:
    version = f"v{dataset_version}"
    prefix = "real_development" if dataset_role == "development" else "real_reserve"
    paths = {
        "source": output_dir / f"{prefix}_queue_{version}.jsonl",
        "reviewer_a": (
            output_dir / f"{prefix}_reviewer_a_queue_{version}.jsonl"
        ),
        "reviewer_b": (
            output_dir / f"{prefix}_reviewer_b_queue_{version}.jsonl"
        ),
        "manifest": output_dir / f"{prefix}_manifest_{version}.json",
        "construction": output_dir / f"{prefix}_construction_{version}.json",
    }
    if any(path.exists() for path in paths.values()):
        raise SystemExit("Development extension output exists; refusing overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    reviewer_queues = {
        "reviewer_a": balanced_reviewer_queue(
            tasks,
            reviewer_id=f"{dataset_role}_{version}_reviewer_a",
            random_state=random_state,
        ),
        "reviewer_b": balanced_reviewer_queue(
            tasks,
            reviewer_id=f"{dataset_role}_{version}_reviewer_b",
            random_state=random_state,
        ),
    }
    write_queue(tasks, paths["source"])
    write_queue(reviewer_queues["reviewer_a"], paths["reviewer_a"])
    write_queue(reviewer_queues["reviewer_b"], paths["reviewer_b"])
    packet = reviewer_packet_manifest(tasks, reviewer_queues)
    if candidate_path is not None and not candidate_path.is_file():
        raise SystemExit("Precommitted candidate artifact is missing.")
    candidate_sha256 = (
        hashlib.sha256(candidate_path.read_bytes()).hexdigest()
        if candidate_path is not None
        else None
    )
    manifest = {
        **packet,
        "dataset_id": f"real_{dataset_role}_{version}",
        "dataset_version": dataset_version,
        "dataset_role": (
            "development_validation_extension"
            if dataset_role == "development"
            else "fresh_source_isolated_reserve"
        ),
        "task_sha256": sha256_bytes(jsonl_bytes(tasks)),
        "role_counts": dict(Counter(task["role_family"] for task in tasks)),
        "source_counts": dict(
            Counter(task["source_dataset"] for task in tasks)
        ),
        "overlap_checks": overlap_checks,
        "construction_uses_model_predictions": False,
        "labels_included": False,
        "old_holdout_or_reserve_used_for_selection": False,
        "construction_reads_candidate_model": False,
        "precommitted_candidate_file": (
            candidate_path.name if candidate_path is not None else None
        ),
        "precommitted_candidate_sha256": candidate_sha256,
        "prior_datasets_used_for_exclusion_only": True,
        "product_integration_allowed": False,
        "status": "ready_for_blind_review",
    }
    paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    paths["construction"].write_text(
        json.dumps(
            {
                "schema_version": 1,
                "stratum_counts": dict(
                    Counter(item["stratum"] for item in construction)
                ),
                "requirement_source_counts": dict(
                    Counter(
                        item["requirement_source"] for item in construction
                    )
                ),
                "tasks": construction,
                "reviewer_visibility": "hidden",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Tasks: {len(tasks)}")
    print(f"Roles: {manifest['role_counts']}")
    print(f"Sources: {manifest['source_counts']}")
    print(f"Content identity verified: {packet['content_identity_verified']}")
    print(f"Output: {output_dir}")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or (
        DEFAULT_OUTPUT
        if args.dataset_role == "development" and args.dataset_version == 2
        else PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / f"real_{args.dataset_role}_v{args.dataset_version}"
    )
    excluded_tasks, excluded_pairs = _load_excluded(
        dataset_version=args.dataset_version,
        dataset_role=args.dataset_role,
    )
    excluded_job_hashes = {
        str(task.get("source_job_hash", "")) for task in excluded_tasks
    }
    excluded_resume_hashes = {
        str(task.get("source_resume_hash", "")) for task in excluded_tasks
    }
    jobs = load_djinni_requirements(
        DJINNI_DIR / "djinni_jobs.parquet",
        excluded_job_hashes,
        random_state=args.random_state,
    )
    skillspan = load_skillspan_requirements(
        SKILLSPAN_DIR,
        excluded_job_hashes,
        random_state=args.random_state,
    )
    profiles = load_djinni_profiles(
        DJINNI_DIR / "djinni_profiles.parquet",
        excluded_resume_hashes,
        random_state=args.random_state,
    )
    if args.dataset_role == "reserve":
        ats_profiles = load_ats_profiles(
            ATS_PROFILE_DIR,
            excluded_resume_hashes,
            random_state=args.random_state,
        )
        profiles = {
            family: [*profiles[family], *ats_profiles[family]]
            for family in profiles
        }
    isolation_index = build_development_isolation_index(
        excluded_tasks,
        excluded_pairs,
    )
    tasks, construction = build_stratified_tasks(
        jobs,
        skillspan,
        profiles,
        tasks_per_family=args.tasks_per_family,
        random_state=args.random_state,
        isolation_index=isolation_index,
        allow_stratum_rebalance=args.dataset_role == "reserve",
    )
    overlap_checks = assert_development_isolated(
        tasks,
        excluded_tasks,
        excluded_pairs,
    )
    _write_outputs(
        output_dir,
        tasks,
        construction,
        overlap_checks,
        dataset_version=args.dataset_version,
        dataset_role=args.dataset_role,
        random_state=args.random_state,
        candidate_path=args.candidate_path,
    )


if __name__ == "__main__":
    main()
