"""Sample unlabeled construction strata without loading model predictions."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import random
from typing import Any

from ml.annotation import validate_queue
from ml.annotation_generation import normalize_text
from ml.evidence import concept_tags, useful_tokens
from ml.evidence_constraints import named_terms, term_families
from ml.real_development import (
    ROLE_FAMILIES,
    build_development_task,
    evidence_overlap,
)
from ml.real_development_isolation import (
    DevelopmentIsolationIndex,
    development_task_isolated,
)


STRATA = ("supportive", "adjacent", "low")
PROFILE_SEARCH_LIMIT = 100
SUPPORTIVE_AFFINITY_MIN = 0.20
ADJACENT_AFFINITY_MIN = 0.08


def construction_affinity(requirement: str, evidence: str) -> float:
    """Combine lexical and deterministic taxonomy overlap for sampling only."""
    lexical = evidence_overlap(requirement, evidence)
    shared_terms = named_terms(requirement) & named_terms(evidence)
    shared_families = term_families(requirement) & term_families(evidence)
    shared_concepts = concept_tags(requirement) & concept_tags(evidence)
    return max(
        lexical,
        0.22 if shared_terms else 0.0,
        0.20 if shared_concepts else 0.0,
        0.18 if shared_families else 0.0,
    )


def affinity_stratum(score: float) -> str:
    """Map construction affinity into mutually exclusive sampling strata."""
    if score >= SUPPORTIVE_AFFINITY_MIN:
        return "supportive"
    if score >= ADJACENT_AFFINITY_MIN:
        return "adjacent"
    return "low"


def build_stratified_tasks(
    jobs: dict[str, list[dict[str, Any]]],
    skillspan: dict[str, list[dict[str, Any]]],
    profiles: dict[str, list[dict[str, Any]]],
    *,
    tasks_per_family: int,
    random_state: int,
    isolation_index: DevelopmentIsolationIndex | None = None,
    allow_stratum_rebalance: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Build balanced unlabeled tasks and hidden construction metadata."""
    if tasks_per_family % len(STRATA):
        raise ValueError("tasks_per_family must be divisible by three.")
    tasks: list[dict[str, Any]] = []
    construction: list[dict[str, str]] = []
    used_jobs: set[str] = set()
    used_resumes: set[str] = set()
    used_requirement_normalized: set[str] = set()
    used_requirement_tokens: list[set[str]] = []
    used_evidence_normalized: set[str] = set()
    used_evidence_tokens: list[set[str]] = []
    per_stratum = tasks_per_family // len(STRATA)
    for family in ROLE_FAMILIES:
        family_start = len(tasks)
        for stratum in STRATA:
            records = [*skillspan[family], *jobs[family]]
            created = _build_one_stratum(
                records,
                profiles,
                family=family,
                stratum=stratum,
                target=per_stratum,
                random_state=random_state,
                used_jobs=used_jobs,
                used_resumes=used_resumes,
                tasks=tasks,
                construction=construction,
                isolation_index=isolation_index,
                used_requirement_normalized=used_requirement_normalized,
                used_requirement_tokens=used_requirement_tokens,
                used_evidence_normalized=used_evidence_normalized,
                used_evidence_tokens=used_evidence_tokens,
            )
            if created != per_stratum and not allow_stratum_rebalance:
                raise ValueError(
                    f"Only built {created}/{per_stratum} "
                    f"{family} {stratum} tasks."
                )
        remaining = tasks_per_family - (len(tasks) - family_start)
        if allow_stratum_rebalance:
            for fallback_stratum in STRATA:
                if not remaining:
                    break
                added = _build_one_stratum(
                    [*skillspan[family], *jobs[family]],
                    profiles,
                    family=family,
                    stratum=fallback_stratum,
                    target=remaining,
                    random_state=random_state,
                    used_jobs=used_jobs,
                    used_resumes=used_resumes,
                    tasks=tasks,
                    construction=construction,
                    isolation_index=isolation_index,
                    used_requirement_normalized=used_requirement_normalized,
                    used_requirement_tokens=used_requirement_tokens,
                    used_evidence_normalized=used_evidence_normalized,
                    used_evidence_tokens=used_evidence_tokens,
                )
                remaining -= added
        if remaining:
            raise ValueError(
                f"Only built {tasks_per_family - remaining}/"
                f"{tasks_per_family} {family} tasks after stratum rebalance."
            )
    random.Random(random_state).shuffle(tasks)
    return tasks, construction


def balanced_reviewer_queue(
    tasks: list[dict[str, Any]],
    *,
    reviewer_id: str,
    random_state: int,
) -> list[dict[str, Any]]:
    """Balance the lexical anchor across A/B/C/D without storing that cue."""
    checked = validate_queue(tasks)
    queue = deepcopy(checked)
    random.Random(f"{random_state}:{reviewer_id}:tasks").shuffle(queue)
    offset = int(
        hashlib.sha256(reviewer_id.encode()).hexdigest()[:8],
        16,
    ) % 4
    unique_anchor_index = 0
    for task in queue:
        candidates = list(task["candidates"])
        scores = [
            evidence_overlap(
                str(task["requirement"]),
                str(candidate["evidence"]),
            )
            for candidate in candidates
        ]
        best_score = max(scores)
        tied = [
            candidate_index
            for candidate_index, score in enumerate(scores)
            if score == best_score
        ]
        if len(tied) > 1:
            tie_seed = f"{random_state}:{reviewer_id}:{task['task_id']}:all"
            random.Random(tie_seed).shuffle(candidates)
            task["candidates"] = candidates
            continue
        tie_seed = f"{random_state}:{reviewer_id}:{task['task_id']}:tie"
        anchor_index = random.Random(tie_seed).choice(tied)
        anchor = candidates.pop(anchor_index)
        rest_seed = f"{random_state}:{reviewer_id}:{task['task_id']}:rest"
        random.Random(rest_seed).shuffle(candidates)
        target_position = (unique_anchor_index + offset) % 4
        unique_anchor_index += 1
        candidates.insert(target_position, anchor)
        task["candidates"] = candidates
    return validate_queue(queue)


def _build_one_stratum(
    records: list[dict[str, Any]],
    profiles: dict[str, list[dict[str, Any]]],
    *,
    family: str,
    stratum: str,
    target: int,
    random_state: int,
    used_jobs: set[str],
    used_resumes: set[str],
    tasks: list[dict[str, Any]],
    construction: list[dict[str, str]],
    isolation_index: DevelopmentIsolationIndex | None,
    used_requirement_normalized: set[str],
    used_requirement_tokens: list[set[str]],
    used_evidence_normalized: set[str],
    used_evidence_tokens: list[set[str]],
) -> int:
    created = 0
    skillspan_count = 0
    for record in records:
        if created >= target:
            break
        is_skillspan = record["source_dataset"] == "skillspan"
        if record["source_hash"] in used_jobs:
            continue
        if is_skillspan and skillspan_count >= 1:
            continue
        ranked_profiles = _rank_profiles(
            str(record["requirement"]),
            family,
            stratum,
            profiles,
            used_resumes,
        )
        selected: tuple[dict[str, Any], dict[str, Any]] | None = None
        for profile in ranked_profiles[:PROFILE_SEARCH_LIMIT]:
            seed = (
                f"{random_state}:{family}:{stratum}:"
                f"{record['source_hash']}:{profile['source_hash']}"
            )
            try:
                task = build_development_task(
                    requirement=str(record["requirement"]),
                    evidence=profile["evidence"],
                    role_family=family,
                    source_job_hash=str(record["source_hash"]),
                    source_resume_hash=str(profile["source_hash"]),
                    source_dataset=(
                        f"{record['source_dataset']}_"
                        f"{profile.get('source_dataset', 'djinni_profile')}"
                    ),
                    seed=seed,
                )
            except ValueError:
                continue
            if (
                isolation_index is not None
                and not development_task_isolated(task, isolation_index)
            ):
                continue
            requirement = str(task["requirement"])
            evidence = [
                str(candidate["evidence"])
                for candidate in task["candidates"]
            ]
            if _batch_text_overlap(
                requirement,
                used_requirement_normalized,
                used_requirement_tokens,
            ) or any(
                _batch_text_overlap(
                    item,
                    used_evidence_normalized,
                    used_evidence_tokens,
                )
                for item in evidence
            ):
                continue
            selected = profile, task
            break
        if selected is None:
            continue
        profile, task = selected
        tasks.append(task)
        construction.append(
            {
                "task_id": str(task["task_id"]),
                "stratum": stratum,
                "requirement_source": str(record["source_dataset"]),
                "profile_family": str(profile["family"]),
            }
        )
        used_jobs.add(str(record["source_hash"]))
        used_resumes.add(str(profile["source_hash"]))
        _record_batch_text(
            str(task["requirement"]),
            used_requirement_normalized,
            used_requirement_tokens,
        )
        for candidate in task["candidates"]:
            _record_batch_text(
                str(candidate["evidence"]),
                used_evidence_normalized,
                used_evidence_tokens,
            )
        created += 1
        skillspan_count += int(is_skillspan)
    return created


def _rank_profiles(
    requirement: str,
    family: str,
    stratum: str,
    profiles: dict[str, list[dict[str, Any]]],
    used_resumes: set[str],
) -> list[dict[str, Any]]:
    candidate_families = (
        [family]
        if stratum != "low"
        else [item for item in ROLE_FAMILIES if item != family]
    )
    candidates = [
        profile
        for candidate_family in candidate_families
        for profile in profiles[candidate_family]
        if profile["source_hash"] not in used_resumes
    ]
    scored = [
        (
            max(
                construction_affinity(requirement, item)
                for item in profile["evidence"]
            ),
            profile,
        )
        for profile in candidates[:500]
    ]
    if stratum == "supportive":
        eligible = [
            item for item in scored if affinity_stratum(item[0]) == stratum
        ]
        return [
            profile
            for _, profile in sorted(
                eligible,
                key=lambda item: (
                    item[0],
                    str(item[1]["source_hash"]),
                ),
                reverse=True,
            )
        ]
    if stratum == "adjacent":
        eligible = [
            item for item in scored if affinity_stratum(item[0]) == stratum
        ]
        return [
            profile
            for _, profile in sorted(
                eligible,
                key=lambda item: (
                    abs(item[0] - 0.10),
                    str(item[1]["source_hash"]),
                ),
            )
        ]
    eligible = [
        item for item in scored if affinity_stratum(item[0]) == stratum
    ]
    return [
        profile
        for _, profile in sorted(
            eligible,
            key=lambda item: (
                item[0],
                str(item[1]["source_hash"]),
            ),
        )
    ]


def _batch_text_overlap(
    text: str,
    normalized: set[str],
    token_sets: list[set[str]],
) -> bool:
    normalized_text = normalize_text(text)
    if normalized_text in normalized:
        return True
    tokens = useful_tokens(text)
    return any(
        bool(tokens | other)
        and len(tokens & other) / len(tokens | other) >= 0.88
        for other in token_sets
    )


def _record_batch_text(
    text: str,
    normalized: set[str],
    token_sets: list[set[str]],
) -> None:
    normalized.add(normalize_text(text))
    token_sets.append(useful_tokens(text))
