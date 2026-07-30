"""Build a source-isolated active-learning acceptance annotation supplement."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DJINNI_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni" / "raw"
SKILLSPAN_DIR = (
    PROJECT_ROOT / "data" / "ml" / "external" / "skillspan" / "data" / "json"
)
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
OLD_REAL_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
DEVELOPMENT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_development_v2"
)
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "annotations"
    / "acceptance_supplement_v1"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.acceptance_sampling import (  # noqa: E402
    build_acceptance_candidate_pool,
    score_acceptance_candidates,
    select_balanced_acceptance_tasks,
)
from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.evidence_multiclass import MulticlassEvidenceReranker  # noqa: E402
from ml.evidence_two_stage import reviewed_multiclass_rows  # noqa: E402
from ml.real_development_isolation import (  # noqa: E402
    assert_development_isolated,
    build_development_isolation_index,
    development_task_isolated,
)
from ml.real_development_sampling import balanced_reviewer_queue  # noqa: E402
from ml.real_development_sources import (  # noqa: E402
    load_djinni_profiles,
    load_djinni_requirements,
    load_skillspan_requirements,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402
from ml.real_review import reviewer_packet_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--tasks-per-family", type=int, default=20)
    parser.add_argument("--random-state", type=int, default=20260727)
    return parser.parse_args()


def _load_excluded() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    tasks = load_jsonl(TRAINING_DIR / "annotated_tasks.jsonl")
    pairs = load_jsonl(TRAINING_DIR / "training_pairs.jsonl")
    for path in (
        OLD_REAL_DIR / "real_holdout_gold_v1.jsonl",
        OLD_REAL_DIR / "real_validation_gold_v1.jsonl",
        OLD_REAL_DIR / "real_reserve_gold_v1.jsonl",
        DEVELOPMENT_DIR / "real_development_gold_v2.jsonl",
    ):
        tasks.extend(load_jsonl(path))
    return tasks, pairs


def _fit_sampling_model(
    base_pairs: list[dict[str, Any]],
) -> MulticlassEvidenceReranker:
    development = load_jsonl(
        DEVELOPMENT_DIR / "real_development_gold_v2.jsonl"
    )
    reviewed = reviewed_multiclass_rows(development)
    requirements = [
        *[str(pair["requirement"]) for pair in base_pairs],
        *[row[0] for row in reviewed],
    ]
    evidence = [
        *[str(pair["evidence"]) for pair in base_pairs],
        *[row[1] for row in reviewed],
    ]
    labels = [
        *[str(pair["support_label"]) for pair in base_pairs],
        *[row[2] for row in reviewed],
    ]
    return MulticlassEvidenceReranker(random_state=42).fit(
        requirements,
        evidence,
        labels,
    )


def _filter_isolated_pool(
    tasks: list[dict[str, Any]],
    excluded_tasks: list[dict[str, Any]],
    excluded_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    index = build_development_isolation_index(
        excluded_tasks,
        excluded_pairs,
    )
    return [
        task
        for task in tasks
        if development_task_isolated(task, index)
    ]


def _write_outputs(
    output_dir: Path,
    tasks: list[dict[str, Any]],
    construction: list[dict[str, Any]],
    overlap_checks: dict[str, int],
    *,
    random_state: int,
    candidate_pool_size: int,
) -> None:
    paths = {
        "source": output_dir / "acceptance_supplement_queue_v1.jsonl",
        "reviewer_a": (
            output_dir / "acceptance_supplement_reviewer_a_queue_v1.jsonl"
        ),
        "reviewer_b": (
            output_dir / "acceptance_supplement_reviewer_b_queue_v1.jsonl"
        ),
        "manifest": output_dir / "acceptance_supplement_manifest_v1.json",
        "construction": (
            output_dir / "acceptance_supplement_construction_v1.json"
        ),
    }
    if any(path.exists() for path in paths.values()):
        raise SystemExit("Acceptance supplement output exists; refusing overwrite.")
    output_dir.mkdir(parents=True, exist_ok=True)
    random.Random(random_state).shuffle(tasks)
    reviewer_queues = {
        "reviewer_a": balanced_reviewer_queue(
            tasks,
            reviewer_id="acceptance_supplement_v1_reviewer_a",
            random_state=random_state,
        ),
        "reviewer_b": balanced_reviewer_queue(
            tasks,
            reviewer_id="acceptance_supplement_v1_reviewer_b",
            random_state=random_state,
        ),
    }
    write_queue(tasks, paths["source"])
    write_queue(reviewer_queues["reviewer_a"], paths["reviewer_a"])
    write_queue(reviewer_queues["reviewer_b"], paths["reviewer_b"])
    packet = reviewer_packet_manifest(tasks, reviewer_queues)
    manifest = {
        **packet,
        "dataset_id": "acceptance_supplement_v1",
        "dataset_role": "active_learning_training_supplement",
        "tasks": len(tasks),
        "candidate_pool_size": candidate_pool_size,
        "task_sha256": sha256_bytes(jsonl_bytes(tasks)),
        "role_counts": dict(
            Counter(str(task["role_family"]) for task in tasks)
        ),
        "stratum_counts": dict(
            Counter(str(item["stratum"]) for item in construction)
        ),
        "source_counts": dict(
            Counter(str(task["source_dataset"]) for task in tasks)
        ),
        "overlap_checks": overlap_checks,
        "construction_uses_model_predictions": True,
        "labels_included": False,
        "valid_for_model_selection_or_final_evaluation": False,
        "reviewer_visibility_of_construction": "hidden",
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
    print(f"Candidate pool: {candidate_pool_size}")
    print(f"Roles: {manifest['role_counts']}")
    print(f"Strata: {manifest['stratum_counts']}")
    print(f"Content identity verified: {packet['content_identity_verified']}")
    print(f"Output: {output_dir}")


def main() -> None:
    args = parse_args()
    excluded_tasks, excluded_pairs = _load_excluded()
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
        family_limit=500,
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
        family_limit=500,
    )
    requirements = {
        family: [*skillspan[family], *jobs[family]]
        for family in jobs
    }
    pool = build_acceptance_candidate_pool(
        requirements,
        profiles,
        random_state=args.random_state,
    )
    pool = _filter_isolated_pool(
        pool,
        excluded_tasks,
        excluded_pairs,
    )
    model = _fit_sampling_model(excluded_pairs)
    scored = score_acceptance_candidates(pool, model)
    tasks, construction = select_balanced_acceptance_tasks(
        scored,
        tasks_per_family=args.tasks_per_family,
    )
    overlap_checks = assert_development_isolated(
        tasks,
        excluded_tasks,
        excluded_pairs,
    )
    _write_outputs(
        args.output_dir,
        tasks,
        construction,
        overlap_checks,
        random_state=args.random_state,
        candidate_pool_size=len(pool),
    )


if __name__ == "__main__":
    main()
