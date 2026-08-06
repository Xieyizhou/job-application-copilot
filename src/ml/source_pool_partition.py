"""Content-free source partitioning for successor development and sealed holdout."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
import hashlib
import json
import random
from typing import Any

from ml.annotation_generation import normalize_text
from ml.evidence import useful_tokens
from ml.operational_development_v3 import construction_features
from ml.real_development import ROLE_FAMILIES
from ml.real_development_isolation import (
    DevelopmentIsolationIndex,
    development_task_isolated,
)
from ml.real_holdout import sha256_bytes


DEVELOPMENT_SPLIT = "successor_development"
HOLDOUT_SPLIT = "sealed_operational_holdout"
SOURCE_POOL_CONSTRUCTION_STRATA = (
    "explicit_aligned",
    "semantic_aligned",
    "compound_requirement",
    "same_role_hard_negative",
    "support_absence_probe",
    "lexical_similarity_trap",
)
ASSIGNMENT_FIELDS = {
    "allocation_id",
    "split",
    "role_family",
    "construction_stratum",
    "source_resume_hash",
    "source_job_hash",
}
FORBIDDEN_ASSIGNMENT_FIELDS = {
    "requirement",
    "evidence",
    "candidates",
    "support_label",
    "selected_candidate_id",
    "candidate_labels",
    "model_score",
    "model_prediction",
    "similarity",
}
SOURCE_POOL_PREFLIGHT_SCHEMA_VERSION = 1


def _near_text(first: str, second: str, *, threshold: float) -> bool:
    if normalize_text(first) == normalize_text(second):
        return True
    left = useful_tokens(first)
    right = useful_tokens(second)
    union = left | right
    return bool(union) and len(left & right) / len(union) >= threshold


def _profile_isolated(
    profile: Mapping[str, Any],
    *,
    index: DevelopmentIsolationIndex,
    prior_evidence: Sequence[str],
) -> bool:
    evidence = [str(value) for value in profile["evidence"]]
    if any(
        _near_text(candidate, prior, threshold=0.88)
        for candidate in evidence
        for prior in prior_evidence
    ):
        return False
    probe = {
        "source_job_hash": "source-partition-probe",
        "source_resume_hash": str(profile["source_hash"]),
        "requirement": "source partition isolation probe",
        "candidates": [
            {"candidate_id": str(candidate_index), "evidence": candidate}
            for candidate_index, candidate in enumerate(evidence)
        ],
    }
    return development_task_isolated(probe, index)


def _requirement_isolated(
    job: Mapping[str, Any],
    *,
    index: DevelopmentIsolationIndex,
) -> bool:
    job_hash = str(job["source_hash"])
    requirement = str(job["requirement"])
    if job_hash in index.job_hashes:
        return False
    normalized = normalize_text(requirement)
    if normalized in index.requirement_normalized:
        return False
    tokens = frozenset(useful_tokens(requirement))
    return not any(
        bool(tokens | prior)
        and len(tokens & prior) / len(tokens | prior) >= 0.88
        for prior in index.requirement_tokens
    )


def source_pool_preflight(
    profiles: Mapping[str, Sequence[dict[str, Any]]],
    jobs: Mapping[str, Sequence[dict[str, Any]]],
    *,
    isolation_index: DevelopmentIsolationIndex,
    development_resumes_per_family: int,
    holdout_resumes_per_family: int,
) -> dict[str, Any]:
    """Report content-free necessary capacity before partition search.

    Counts are computed against consumed source and near-text isolation. They
    stop at the required capacity, intentionally do not expose source ids or
    text, and do not claim that a mutually isolated six-job allocation is
    possible.
    """
    if development_resumes_per_family <= 0 or holdout_resumes_per_family < 0:
        raise ValueError(
            "Development needs resume groups and holdout groups cannot be negative."
        )
    required_profiles = (
        development_resumes_per_family + holdout_resumes_per_family
    )
    required_requirements = required_profiles * len(
        SOURCE_POOL_CONSTRUCTION_STRATA
    )
    families: dict[str, dict[str, int | bool]] = {}
    for family in ROLE_FAMILIES:
        family_profiles = list(profiles.get(family, ()))
        family_jobs = list(jobs.get(family, ()))
        isolated_profiles = 0
        for profile in family_profiles:
            isolated_profiles += int(
                _profile_isolated(
                    profile,
                    index=isolation_index,
                    prior_evidence=(),
                )
            )
            if isolated_profiles >= required_profiles:
                break
        isolated_requirements = 0
        for job in family_jobs:
            isolated_requirements += int(
                _requirement_isolated(job, index=isolation_index)
            )
            if isolated_requirements >= required_requirements:
                break
        families[family] = {
            "loaded_profiles": len(family_profiles),
            "isolated_profiles_found": isolated_profiles,
            "required_profiles": required_profiles,
            "loaded_requirements": len(family_jobs),
            "isolated_requirements_found": isolated_requirements,
            "required_requirements": required_requirements,
            "necessary_capacity_passed": (
                isolated_profiles >= required_profiles
                and isolated_requirements >= required_requirements
            ),
        }
    return {
        "schema_version": SOURCE_POOL_PREFLIGHT_SCHEMA_VERSION,
        "content_included": False,
        "source_ids_included": False,
        "labels_included": False,
        "model_signals_included": False,
        "necessary_capacity_only": True,
        "capacity_counts_capped_at_required": True,
        "development_resumes_per_family": development_resumes_per_family,
        "holdout_resumes_per_family": holdout_resumes_per_family,
        "role_families": families,
        "preflight_passed": all(
            bool(row["necessary_capacity_passed"])
            for row in families.values()
        ),
    }


def _job_isolated(
    job: Mapping[str, Any],
    profile: Mapping[str, Any],
    *,
    index: DevelopmentIsolationIndex,
    used_job_hashes: set[str],
    prior_requirements: Sequence[str],
) -> bool:
    job_hash = str(job["source_hash"])
    requirement = str(job["requirement"])
    if job_hash in used_job_hashes or any(
        _near_text(requirement, prior, threshold=0.82)
        for prior in prior_requirements
    ):
        return False
    probe = {
        "source_job_hash": job_hash,
        "source_resume_hash": str(profile["source_hash"]),
        "requirement": requirement,
        "candidates": [
            {"candidate_id": str(candidate_index), "evidence": str(evidence)}
            for candidate_index, evidence in enumerate(profile["evidence"])
        ],
    }
    return development_task_isolated(probe, index)


def _ordered_jobs(
    jobs: Sequence[dict[str, Any]],
    evidence: Sequence[str],
    *,
    stratum: str,
    seed: str,
) -> list[dict[str, Any]]:
    ranked: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
    evidence_key = tuple(str(value) for value in evidence)
    for job in jobs:
        lexical, taxonomy, compound, tokens = _cached_construction_features(
            str(job["requirement"]),
            evidence_key,
        )
        key: tuple[float, ...]
        if stratum == "explicit_aligned":
            key = (lexical, taxonomy, -abs(tokens - 12.0))
        elif stratum == "semantic_aligned":
            key = (taxonomy, -lexical, -abs(tokens - 15.0))
        elif stratum == "compound_requirement":
            key = (compound, taxonomy, lexical, tokens)
        elif stratum == "same_role_hard_negative":
            key = (taxonomy, -abs(lexical - 0.10), -abs(tokens - 14.0))
        elif stratum == "support_absence_probe":
            key = (-lexical, -taxonomy, -compound, -tokens)
        elif stratum == "lexical_similarity_trap":
            key = (lexical, -taxonomy, -compound, -abs(tokens - 14.0))
        else:
            raise ValueError(f"Unknown source-pool stratum: {stratum}")
        tie_break = hashlib.sha256(
            f"{seed}:{job['source_hash']}".encode()
        ).hexdigest()
        ranked.append((key, tie_break, job))
    ranked.sort(key=lambda row: (row[0], row[1]), reverse=True)
    return [job for _, _, job in ranked]


@lru_cache(maxsize=65_536)
def _cached_construction_features(
    requirement: str,
    evidence: tuple[str, ...],
) -> tuple[float, float, float, float]:
    """Cache content features across the six strata for one profile/job pair."""
    features = construction_features(requirement, evidence)
    return (
        float(features["lexical"]),
        float(features["taxonomy_overlap"]),
        float(features["compound"]),
        float(features["token_count"]),
    )


def _allocation_row(
    *,
    split: str,
    family: str,
    stratum: str,
    resume_hash: str,
    job_hash: str,
) -> dict[str, str]:
    allocation_id = hashlib.sha256(
        f"{split}:{family}:{stratum}:{resume_hash}:{job_hash}".encode()
    ).hexdigest()[:24]
    return {
        "allocation_id": f"source-allocation-{allocation_id}",
        "split": split,
        "role_family": family,
        "construction_stratum": stratum,
        "source_resume_hash": resume_hash,
        "source_job_hash": job_hash,
    }


def _select_family_groups(
    ordered_profiles: Sequence[dict[str, Any]],
    family_jobs: Sequence[dict[str, Any]],
    *,
    family: str,
    split_schedule: Sequence[str],
    isolation_index: DevelopmentIsolationIndex,
    used_jobs: set[str],
    used_resumes: set[str],
    used_requirements: Sequence[str],
    used_evidence: Sequence[str],
    random_state: int,
) -> list[tuple[str, dict[str, Any], list[tuple[str, dict[str, Any]]]]]:
    """Find a complete family allocation without accepting a greedy dead end."""

    def search(
        start: int,
        selected_groups: list[
            tuple[str, dict[str, Any], list[tuple[str, dict[str, Any]]]]
        ],
        local_jobs: set[str],
        local_requirements: list[str],
        local_evidence: list[str],
    ) -> list[
        tuple[str, dict[str, Any], list[tuple[str, dict[str, Any]]]]
    ] | None:
        if len(selected_groups) == len(split_schedule):
            return selected_groups
        for profile_index in range(start, len(ordered_profiles)):
            profile = ordered_profiles[profile_index]
            resume_hash = str(profile["source_hash"])
            if resume_hash in used_resumes or not _profile_isolated(
                profile,
                index=isolation_index,
                prior_evidence=[*used_evidence, *local_evidence],
            ):
                continue
            tentative_jobs: set[str] = set()
            tentative_requirements: list[str] = []
            selected_jobs: list[tuple[str, dict[str, Any]]] = []
            for stratum in SOURCE_POOL_CONSTRUCTION_STRATA:
                selected = next(
                    (
                        job
                        for job in _ordered_jobs(
                            family_jobs,
                            [str(value) for value in profile["evidence"]],
                            stratum=stratum,
                            seed=(
                                f"{random_state}:{family}:{resume_hash}:"
                                f"{stratum}"
                            ),
                        )
                        if _job_isolated(
                            job,
                            profile,
                            index=isolation_index,
                            used_job_hashes=(
                                used_jobs | local_jobs | tentative_jobs
                            ),
                            prior_requirements=[
                                *used_requirements,
                                *local_requirements,
                                *tentative_requirements,
                            ],
                        )
                    ),
                    None,
                )
                if selected is None:
                    selected_jobs = []
                    break
                selected_jobs.append((stratum, selected))
                tentative_jobs.add(str(selected["source_hash"]))
                tentative_requirements.append(str(selected["requirement"]))
            if len(selected_jobs) != len(SOURCE_POOL_CONSTRUCTION_STRATA):
                continue
            result = search(
                profile_index + 1,
                [
                    *selected_groups,
                    (
                        split_schedule[len(selected_groups)],
                        profile,
                        selected_jobs,
                    ),
                ],
                local_jobs | tentative_jobs,
                [*local_requirements, *tentative_requirements],
                [
                    *local_evidence,
                    *(str(value) for value in profile["evidence"]),
                ],
            )
            if result is not None:
                return result
        return None

    return search(0, [], set(), [], []) or []


def build_source_partition(
    profiles: Mapping[str, Sequence[dict[str, Any]]],
    jobs: Mapping[str, Sequence[dict[str, Any]]],
    *,
    isolation_index: DevelopmentIsolationIndex,
    development_resumes_per_family: int,
    holdout_resumes_per_family: int,
    random_state: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Assign sources without emitting requirement, evidence, label, or model data."""
    if development_resumes_per_family <= 0 or holdout_resumes_per_family < 0:
        raise ValueError(
            "Development needs resume groups and holdout groups cannot be negative."
        )
    used_jobs: set[str] = set()
    used_resumes: set[str] = set()
    used_requirements: list[str] = []
    used_evidence: list[str] = []
    assignments: dict[str, list[dict[str, str]]] = {
        DEVELOPMENT_SPLIT: [],
        HOLDOUT_SPLIT: [],
    }
    for family in ROLE_FAMILIES:
        split_schedule = [
            *[DEVELOPMENT_SPLIT] * development_resumes_per_family,
            *[HOLDOUT_SPLIT] * holdout_resumes_per_family,
        ]
        random.Random(f"{random_state}:{family}:split-schedule").shuffle(
            split_schedule
        )
        ordered_profiles = sorted(
            profiles[family],
            key=lambda profile: hashlib.sha256(
                f"{random_state}:{family}:{profile['source_hash']}".encode()
            ).hexdigest(),
        )
        family_groups = _select_family_groups(
            ordered_profiles,
            jobs[family],
            family=family,
            split_schedule=split_schedule,
            isolation_index=isolation_index,
            used_jobs=used_jobs,
            used_resumes=used_resumes,
            used_requirements=used_requirements,
            used_evidence=used_evidence,
            random_state=random_state,
        )
        if len(family_groups) != len(split_schedule):
            raise ValueError(
                f"Only selected {len(family_groups)}/{len(split_schedule)} "
                f"isolated {family} resume groups."
            )
        for split, profile, selected_jobs in family_groups:
            resume_hash = str(profile["source_hash"])
            assignments[split].extend(
                _allocation_row(
                    split=split,
                    family=family,
                    stratum=stratum,
                    resume_hash=resume_hash,
                    job_hash=str(job["source_hash"]),
                )
                for stratum, job in selected_jobs
            )
            tentative_jobs = {
                str(job["source_hash"]) for _, job in selected_jobs
            }
            tentative_requirements = [
                str(job["requirement"]) for _, job in selected_jobs
            ]
            used_resumes.add(resume_hash)
            used_jobs.update(tentative_jobs)
            used_requirements.extend(tentative_requirements)
            used_evidence.extend(str(value) for value in profile["evidence"])
    development = sorted(
        assignments[DEVELOPMENT_SPLIT],
        key=lambda row: row["allocation_id"],
    )
    holdout = sorted(
        assignments[HOLDOUT_SPLIT],
        key=lambda row: row["allocation_id"],
    )
    validate_source_partition(
        development,
        holdout,
        development_resumes_per_family=development_resumes_per_family,
        holdout_resumes_per_family=holdout_resumes_per_family,
    )
    return development, holdout


def validate_source_partition(
    development: Sequence[Mapping[str, Any]],
    holdout: Sequence[Mapping[str, Any]],
    *,
    development_resumes_per_family: int,
    holdout_resumes_per_family: int,
) -> dict[str, Any]:
    """Validate counts, content-free rows, and complete cross-split isolation."""
    all_rows = [*development, *holdout]
    for row in all_rows:
        if set(row) != ASSIGNMENT_FIELDS:
            raise ValueError("Source assignments contain unknown or missing fields.")
        if set(row) & FORBIDDEN_ASSIGNMENT_FIELDS:
            raise ValueError("Source assignments expose content, labels, or scores.")
    if any(row["split"] != DEVELOPMENT_SPLIT for row in development):
        raise ValueError("Development source assignment has the wrong split.")
    if any(row["split"] != HOLDOUT_SPLIT for row in holdout):
        raise ValueError("Holdout source assignment has the wrong split.")
    if len({str(row["allocation_id"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Source allocation ids must be unique.")
    if len({str(row["source_job_hash"]) for row in all_rows}) != len(all_rows):
        raise ValueError("Every source-pool job must be unique.")
    development_resumes = {
        str(row["source_resume_hash"]) for row in development
    }
    holdout_resumes = {
        str(row["source_resume_hash"]) for row in holdout
    }
    if development_resumes & holdout_resumes:
        raise ValueError("Development and holdout resume groups overlap.")
    expected_strata = set(SOURCE_POOL_CONSTRUCTION_STRATA)
    for split, rows, per_family in (
        (
            DEVELOPMENT_SPLIT,
            development,
            development_resumes_per_family,
        ),
        (HOLDOUT_SPLIT, holdout, holdout_resumes_per_family),
    ):
        role_counts = Counter(str(row["role_family"]) for row in rows)
        expected_tasks = per_family * len(expected_strata)
        if any(role_counts[family] != expected_tasks for family in ROLE_FAMILIES):
            raise ValueError(f"{split} role-family assignments are imbalanced.")
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row["source_resume_hash"]), []).append(row)
        if len(groups) != per_family * len(ROLE_FAMILIES):
            raise ValueError(f"{split} has the wrong number of resume groups.")
        for group in groups.values():
            if {str(row["construction_stratum"]) for row in group} != (
                expected_strata
            ):
                raise ValueError("Every resume group must contain every stratum.")
            if len({str(row["role_family"]) for row in group}) != 1:
                raise ValueError("One resume group cannot cross role families.")
    return {
        "development_assignments": len(development),
        "holdout_assignments": len(holdout),
        "development_resume_groups": len(development_resumes),
        "holdout_resume_groups": len(holdout_resumes),
        "source_resume_overlap": 0,
        "source_job_overlap": 0,
        "exact_text_overlap": 0,
        "near_text_overlap": 0,
        "content_included": False,
        "labels_included": False,
        "model_signals_included": False,
    }


def assignment_sha256(assignments: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable checksum for content-free source assignments."""
    ordered = sorted(
        (dict(row) for row in assignments),
        key=lambda row: str(row["allocation_id"]),
    )
    payload = json.dumps(
        ordered,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return sha256_bytes(payload)


def isolation_index_summary(
    index: DevelopmentIsolationIndex,
) -> dict[str, int | str]:
    """Commit to consumed isolation state without exposing normalized text."""
    payload = {
        "job_hashes": sorted(index.job_hashes),
        "resume_hashes": sorted(index.resume_hashes),
        "requirement_text_hashes": sorted(
            hashlib.sha256(value.encode()).hexdigest()
            for value in index.requirement_normalized
        ),
        "evidence_text_hashes": sorted(
            hashlib.sha256(value.encode()).hexdigest()
            for value in index.evidence_normalized
        ),
    }
    return {
        "consumed_job_hashes": len(index.job_hashes),
        "consumed_resume_hashes": len(index.resume_hashes),
        "consumed_requirement_texts": len(index.requirement_normalized),
        "consumed_evidence_texts": len(index.evidence_normalized),
        "isolation_index_sha256": sha256_bytes(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ),
    }
