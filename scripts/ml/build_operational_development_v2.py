"""Build source-isolated one-resume-to-many-jobs development packets."""

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
ANNOTATION_DIR = PROJECT_ROOT / "data" / "ml" / "annotations"
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed"
    / "reviewed_evidence_training_v4_batch4"
)
DJINNI_DIR = PROJECT_ROOT / "data" / "ml" / "external" / "djinni" / "raw"
ATS_PROFILE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "raw" / "resume_ats_score_v1_en"
)
TALENTCLEF_TASK_A_DIR = (
    PROJECT_ROOT / "data" / "ml" / "external" / "talentclef_2026"
    / "extracted" / "TaskA"
)
DEFAULT_OUTPUT = ANNOTATION_DIR / "operational_development_v2"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.evidence import useful_tokens  # noqa: E402
from ml.operational_development import (  # noqa: E402
    CROSS_ROLE,
    assert_operational_development_contract,
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
from ml.real_development_sampling import balanced_reviewer_queue  # noqa: E402
from ml.real_development_sources import (  # noqa: E402
    load_ats_profiles,
    load_djinni_profiles,
    load_djinni_requirements,
    load_talentclef_profiles,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402
from ml.real_review import reviewer_packet_manifest  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resumes-per-family", type=int, default=4)
    parser.add_argument("--tasks-per-resume", type=int, default=4)
    parser.add_argument("--random-state", type=int, default=20260730)
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
    return bool(union) and len(left & right) / len(union) >= 0.82


def _build_profile_tasks(
    profile: dict[str, Any],
    *,
    profile_family: str,
    jobs: dict[str, list[dict[str, Any]]],
    used_jobs: set[str],
    used_requirements: list[str],
    isolation_index: Any,
    tasks_per_resume: int,
    random_state: int,
) -> list[dict[str, Any]]:
    if tasks_per_resume < 2:
        raise ValueError("Operational development needs multiple jobs per resume.")
    target_families = [
        *([profile_family] * (tasks_per_resume - 1)),
        CROSS_ROLE[profile_family],
    ]
    tentative: list[dict[str, Any]] = []
    tentative_jobs: set[str] = set()
    tentative_requirements: list[str] = []
    for task_index, target_family in enumerate(target_families):
        selected: dict[str, Any] | None = None
        for job in jobs[target_family]:
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
                    role_family=target_family,
                    source_job_hash=job_hash,
                    source_resume_hash=str(profile["source_hash"]),
                    source_dataset=(
                        "djinni_job_plus_"
                        f"{profile.get('source_dataset', 'unknown_profile')}_"
                        "operational"
                    ),
                    seed=(
                        f"{random_state}:{profile['source_hash']}:"
                        f"{job_hash}:{task_index}"
                    ),
                )
            except DevelopmentDataError:
                continue
            task.update(
                {
                    "profile_role_family": profile_family,
                    "pairing_type": (
                        "aligned"
                        if target_family == profile_family
                        else "cross_role_hard_negative"
                    ),
                    "operational_resume_group": str(profile["source_hash"]),
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
    tasks_per_resume: int,
    random_state: int,
    excluded_tasks: list[dict[str, Any]],
    excluded_pairs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    excluded_jobs = {
        str(task.get("source_job_hash", "")) for task in excluded_tasks
    }
    excluded_resumes = {
        str(task.get("source_resume_hash", "")) for task in excluded_tasks
    }
    jobs = load_djinni_requirements(
        DJINNI_DIR / "djinni_jobs.parquet",
        excluded_jobs,
        random_state=random_state,
        family_limit=300,
    )
    talentclef_profiles = load_talentclef_profiles(
        TALENTCLEF_TASK_A_DIR,
        excluded_resumes,
        random_state=random_state,
        family_limit=120,
    )
    needs_fallback = any(
        len(talentclef_profiles[family]) < resumes_per_family
        for family in ROLE_FAMILIES
    )
    djinni_profiles = (
        load_djinni_profiles(
            DJINNI_DIR / "djinni_profiles.parquet",
            excluded_resumes,
            random_state=random_state,
            family_limit=300,
        )
        if needs_fallback
        else {family: [] for family in ROLE_FAMILIES}
    )
    ats_profiles = (
        load_ats_profiles(
            ATS_PROFILE_DIR,
            excluded_resumes,
            random_state=random_state,
            family_limit=120,
        )
        if needs_fallback
        else {family: [] for family in ROLE_FAMILIES}
    )
    profiles = {
        family: [
            *talentclef_profiles[family],
            *djinni_profiles[family],
            *ats_profiles[family],
        ]
        for family in ROLE_FAMILIES
    }
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
                profile_family=family,
                jobs=jobs,
                used_jobs=used_jobs,
                used_requirements=used_requirements,
                isolation_index=isolation_index,
                tasks_per_resume=tasks_per_resume,
                random_state=random_state,
            )
            if len(profile_tasks) != tasks_per_resume:
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
                f"{family} profiles."
            )
    random.Random(random_state).shuffle(tasks)
    return tasks


def main() -> None:
    args = parse_args()
    output_paths = {
        "queue": args.output_dir / "queue.jsonl",
        "reviewer_a": args.output_dir / "reviewer_a_queue.jsonl",
        "reviewer_b": args.output_dir / "reviewer_b_queue.jsonl",
        "manifest": args.output_dir / "manifest.json",
    }
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("Operational development output exists; refusing overwrite.")
    excluded_tasks, excluded_pairs = _consumed_data()
    tasks = _build_tasks(
        resumes_per_family=args.resumes_per_family,
        tasks_per_resume=args.tasks_per_resume,
        random_state=args.random_state,
        excluded_tasks=excluded_tasks,
        excluded_pairs=excluded_pairs,
    )
    contract = assert_operational_development_contract(
        tasks,
        excluded_tasks,
        excluded_pairs,
        tasks_per_resume=args.tasks_per_resume,
        resumes_per_profile_family=args.resumes_per_family,
    )
    reviewers = {
        "reviewer_a": balanced_reviewer_queue(
            tasks,
            reviewer_id="operational_v2_reviewer_a",
            random_state=args.random_state,
        ),
        "reviewer_b": balanced_reviewer_queue(
            tasks,
            reviewer_id="operational_v2_reviewer_b",
            random_state=args.random_state,
        ),
    }
    packet = reviewer_packet_manifest(tasks, reviewers)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_queue(tasks, output_paths["queue"])
    write_queue(reviewers["reviewer_a"], output_paths["reviewer_a"])
    write_queue(reviewers["reviewer_b"], output_paths["reviewer_b"])
    manifest = {
        **packet,
        **contract,
        "dataset_id": "operational_development_v2",
        "dataset_role": "successor_model_development_only",
        "resume_source": (
            "TalentCLEF 2026 privacy-protecting synthetic profiles "
            "preferred; public Djinni and local ATS fallbacks"
        ),
        "job_source": "public Djinni job descriptions",
        "talentclef_record": "10.5281/zenodo.19652670",
        "talentclef_license": "CC-BY-4.0",
        "contains_real_candidate_profiles": False,
        "eligible_for_final_real_evaluation": False,
        "task_source_counts": dict(
            Counter(str(task["source_dataset"]) for task in tasks)
        ),
        "task_sha256": sha256_bytes(jsonl_bytes(tasks)),
        "candidate_count": sum(len(task["candidates"]) for task in tasks),
        "candidate_model_predictions_used": False,
        "shadow_v1_holdout_used_for_construction": False,
        "split_policy": "group_by_source_resume_hash_then_source_job_hash",
        "labels_included": False,
        "status": "ready_for_blind_review",
    }
    output_paths["manifest"].write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Tasks: {manifest['tasks']}")
    print(f"Resume groups: {manifest['resume_groups']}")
    print(f"Requirement roles: {manifest['requirement_role_counts']}")
    print(f"Pairing: {manifest['pairing_counts']}")
    print(f"Candidates: {manifest['candidate_count']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
