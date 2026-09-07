"""Build the blind, source-isolated 120-task operational v3 packet."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any


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
DEFAULT_OUTPUT = ANNOTATION_DIR / "operational_development_v3"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.annotation_generation import normalize_text  # noqa: E402
from ml.evidence_text import useful_tokens  # noqa: E402
from ml.operational_development_v3 import (  # noqa: E402
    CONSTRUCTION_STRATA,
    TAXONOMY_REFERENCE,
    build_single_reviewer_packet,
    ordered_jobs_for_stratum,
    validate_v3_contract,
)
from ml.real_development import (  # noqa: E402
    DevelopmentDataError,
    ROLE_FAMILIES,
    build_development_task,
)
from ml.real_development_isolation import (  # noqa: E402
    build_development_isolation_index,
    development_task_isolated,
)
from ml.real_development_sources import (  # noqa: E402
    load_djinni_profiles,
    load_djinni_requirements,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resumes-per-family", type=int, default=6)
    parser.add_argument("--tasks-per-resume", type=int, default=5)
    parser.add_argument("--hidden-repeats", type=int, default=12)
    parser.add_argument("--random-state", type=int, default=20260731)
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


def _near_requirement(first: str, second: str) -> bool:
    left = useful_tokens(first)
    right = useful_tokens(second)
    union = left | right
    return bool(union) and (
        normalize_text(first) == normalize_text(second)
        or len(left & right) / len(union) >= 0.82
    )


def _build_profile_tasks(
    profile: dict[str, Any],
    *,
    family: str,
    jobs: list[dict[str, Any]],
    used_jobs: set[str],
    used_requirements: list[str],
    isolation_index: Any,
    source_revision: str,
    random_state: int,
) -> list[dict[str, Any]]:
    tentative: list[dict[str, Any]] = []
    tentative_jobs: set[str] = set()
    tentative_requirements: list[str] = []
    for stratum in CONSTRUCTION_STRATA:
        selected: dict[str, Any] | None = None
        ordered = ordered_jobs_for_stratum(
            jobs,
            profile["evidence"],
            stratum=stratum,
            seed=f"{random_state}:{profile['source_hash']}:{stratum}",
        )
        for job in ordered:
            job_hash = str(job["source_hash"])
            requirement = str(job["requirement"])
            if (
                job_hash in used_jobs
                or job_hash in tentative_jobs
                or any(
                    _near_requirement(requirement, prior)
                    for prior in [*used_requirements, *tentative_requirements]
                )
            ):
                continue
            try:
                task = build_development_task(
                    requirement=requirement,
                    evidence=profile["evidence"],
                    role_family=family,
                    source_job_hash=job_hash,
                    source_resume_hash=str(profile["source_hash"]),
                    source_dataset=(
                        "djinni_job_plus_djinni_profile_operational_v3"
                    ),
                    seed=(
                        f"{random_state}:{profile['source_hash']}:"
                        f"{job_hash}:{stratum}"
                    ),
                )
            except DevelopmentDataError:
                continue
            task.update(
                {
                    "presentation_id": str(task["task_id"]),
                    "hidden_repeat_of": None,
                    "operational_resume_group": str(
                        profile["source_hash"]
                    ),
                    "construction_stratum": stratum,
                    "taxonomy_reference": list(TAXONOMY_REFERENCE),
                    "source_dataset_revision": source_revision,
                    "profile_role_family": family,
                    "pairing_type": (
                        "aligned"
                        if stratum
                        in {
                            "explicit_aligned",
                            "semantic_aligned",
                            "compound_requirement",
                        }
                        else (
                            "same_role_adjacent"
                            if stratum == "same_role_adjacent"
                            else "same_role_hard_negative"
                        )
                    ),
                }
            )
            if not development_task_isolated(task, isolation_index):
                continue
            selected = task
            break
        if selected is None:
            return []
        tentative.append(selected)
        tentative_jobs.add(str(selected["source_job_hash"]))
        tentative_requirements.append(str(selected["requirement"]))
    return tentative


def _build_tasks(
    *,
    resumes_per_family: int,
    random_state: int,
    excluded_tasks: list[dict[str, Any]],
    excluded_pairs: list[dict[str, Any]],
    source_revision: str,
) -> tuple[list[dict[str, Any]], Any]:
    excluded_jobs = {
        str(task.get("source_job_hash", "")) for task in excluded_tasks
    }
    excluded_resumes = {
        str(task.get("source_resume_hash", "")) for task in excluded_tasks
    }
    jobs = load_djinni_requirements(
        DJINNI_DIR / "raw" / "djinni_jobs.parquet",
        excluded_jobs,
        random_state=random_state,
        family_limit=500,
    )
    profiles = load_djinni_profiles(
        DJINNI_DIR / "raw" / "djinni_profiles.parquet",
        excluded_resumes,
        random_state=random_state,
        family_limit=400,
        include_supplemental_profile_text=True,
        prefer_specialized_role=True,
        minimum_evidence=4,
    )
    isolation_index = build_development_isolation_index(
        excluded_tasks,
        excluded_pairs,
    )
    tasks: list[dict[str, Any]] = []
    used_jobs: set[str] = set()
    used_resumes: set[str] = set()
    used_requirements: list[str] = []
    for family in ROLE_FAMILIES:
        selected_profiles = 0
        for profile in profiles[family]:
            resume_hash = str(profile["source_hash"])
            if resume_hash in used_resumes:
                continue
            profile_tasks = _build_profile_tasks(
                profile,
                family=family,
                jobs=jobs[family],
                used_jobs=used_jobs,
                used_requirements=used_requirements,
                isolation_index=isolation_index,
                source_revision=source_revision,
                random_state=random_state,
            )
            if len(profile_tasks) != len(CONSTRUCTION_STRATA):
                continue
            tasks.extend(profile_tasks)
            used_resumes.add(resume_hash)
            used_jobs.update(
                str(task["source_job_hash"]) for task in profile_tasks
            )
            used_requirements.extend(
                str(task["requirement"]) for task in profile_tasks
            )
            selected_profiles += 1
            if selected_profiles == resumes_per_family:
                break
        if selected_profiles != resumes_per_family:
            raise SystemExit(
                f"Only selected {selected_profiles}/{resumes_per_family} "
                f"{family} Djinni profiles."
            )
    random.Random(random_state).shuffle(tasks)
    return tasks, isolation_index


def main() -> None:
    args = parse_args()
    if args.tasks_per_resume != len(CONSTRUCTION_STRATA):
        raise SystemExit("Operational v3 requires exactly five tasks per resume.")
    output_paths = {
        "queue": args.output_dir / "queue.jsonl",
        "reviewer": args.output_dir / "reviewer_queue.jsonl",
        "repeat_map": args.output_dir / "repeat_map.json",
        "manifest": args.output_dir / "manifest.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("Operational v3 output exists; refusing overwrite.")
    djinni_manifest_path = DJINNI_DIR / "manifest.json"
    source_revision = "djinni-manifest-sha256:" + hashlib.sha256(
        djinni_manifest_path.read_bytes()
    ).hexdigest()
    excluded_tasks, excluded_pairs = _consumed_data()
    tasks, isolation_index = _build_tasks(
        resumes_per_family=args.resumes_per_family,
        random_state=args.random_state,
        excluded_tasks=excluded_tasks,
        excluded_pairs=excluded_pairs,
        source_revision=source_revision,
    )
    contract = validate_v3_contract(
        tasks,
        isolation_index=isolation_index,
        resumes_per_family=args.resumes_per_family,
        tasks_per_resume=args.tasks_per_resume,
    )
    reviewer_queue, repeat_map = build_single_reviewer_packet(
        tasks,
        repeat_count=args.hidden_repeats,
        random_state=args.random_state,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_queue(tasks, output_paths["queue"])
    write_queue(reviewer_queue, output_paths["reviewer"])
    output_paths["repeat_map"].write_text(
        json.dumps(repeat_map, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "schema_version": 1,
        "dataset_id": "operational_development_v3",
        "dataset_role": "successor_training_and_nested_development_only",
        **contract,
        "review_presentations": len(reviewer_queue),
        "hidden_repeats": len(repeat_map),
        "resume_source": "unused public Djinni candidate profiles",
        "job_source": "unused public Djinni job descriptions",
        "source_dataset_revision": source_revision,
        "taxonomy_reference": list(TAXONOMY_REFERENCE),
        "taxonomy_is_label_authority": False,
        "contains_real_candidate_profiles": True,
        "eligible_for_final_evaluation": False,
        "product_integration_allowed": False,
        "task_source_counts": dict(
            Counter(str(task["source_dataset"]) for task in tasks)
        ),
        "task_sha256": sha256_bytes(jsonl_bytes(tasks)),
        "reviewer_packet_sha256": sha256_bytes(
            jsonl_bytes(reviewer_queue)
        ),
        "repeat_map_sha256": sha256_bytes(
            json.dumps(repeat_map, sort_keys=True).encode()
        ),
        "candidate_model_predictions_used": False,
        "model_disagreements_used_for_sampling": False,
        "frozen_reserve_labels_read": False,
        "shadow_failures_used_for_sampling": False,
        "post_label_selection_allowed": False,
        "split_policy": "keep_all_five_tasks_grouped_by_source_resume_hash",
        "labels_included": False,
        "status": "ready_for_single_blind_review",
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Tasks: {manifest['tasks']}")
    print(f"Resume groups: {manifest['resume_groups']}")
    print(f"Role counts: {manifest['role_counts']}")
    print(f"Construction strata: {manifest['construction_stratum_counts']}")
    print(f"Candidates: {manifest['candidate_count']}")
    print(f"Review presentations: {manifest['review_presentations']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
