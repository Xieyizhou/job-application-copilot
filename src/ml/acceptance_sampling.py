"""Active-learning sampling for real-text evidence acceptance labels."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ml.evidence_multiclass import SUPPORT_CLASSES, MulticlassEvidenceReranker
from ml.real_development import (
    ROLE_FAMILIES,
    build_development_task,
    evidence_overlap,
)


ACCEPTANCE_STRATA = (
    "semantic_lexical_gap",
    "high_similarity_low_support",
    "rank_disagreement",
    "support_boundary",
)


def build_acceptance_candidate_pool(
    requirements: dict[str, list[dict[str, Any]]],
    profiles: dict[str, list[dict[str, Any]]],
    *,
    records_per_family: int = 140,
    profiles_considered: int = 160,
    random_state: int,
) -> list[dict[str, Any]]:
    """Build diverse task candidates before model-disagreement selection."""
    tasks: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    for family in ROLE_FAMILIES:
        records = requirements[family][:records_per_family]
        same_family = profiles[family][:profiles_considered]
        other_family = [
            profile
            for name in ROLE_FAMILIES
            if name != family
            for profile in profiles[name][: max(profiles_considered // 3, 1)]
        ]
        for record in records:
            requirement = str(record["requirement"])
            profile_choices = _profile_choices(
                requirement,
                same_family,
                other_family,
            )
            for profile in profile_choices:
                seed = (
                    f"{random_state}:{record['source_hash']}:"
                    f"{profile['source_hash']}"
                )
                try:
                    task = build_development_task(
                        requirement=requirement,
                        evidence=profile["evidence"],
                        role_family=family,
                        source_job_hash=str(record["source_hash"]),
                        source_resume_hash=str(profile["source_hash"]),
                        source_dataset=(
                            f"{record['source_dataset']}_djinni_profile"
                        ),
                        seed=seed,
                    )
                except ValueError:
                    continue
                task_id = str(task["task_id"])
                if task_id not in seen_task_ids:
                    tasks.append(task)
                    seen_task_ids.add(task_id)
    return tasks


def score_acceptance_candidates(
    tasks: Sequence[dict[str, Any]],
    model: MulticlassEvidenceReranker,
) -> list[dict[str, Any]]:
    """Attach hidden disagreement affinities without assigning gold labels."""
    scored: list[dict[str, Any]] = []
    direct_index = SUPPORT_CLASSES.index("Direct")
    partial_index = SUPPORT_CLASSES.index("Partial")
    no_support_index = SUPPORT_CLASSES.index("No Support")
    for task in tasks:
        candidates = list(task["candidates"])
        requirements = [str(task["requirement"])] * len(candidates)
        evidence = [str(candidate["evidence"]) for candidate in candidates]
        tfidf = np.asarray(
            model.word.score(requirements, evidence),
            dtype=np.float64,
        )
        lsa = np.asarray(
            model.embedding.score(requirements, evidence),
            dtype=np.float64,
        )
        probabilities = model.predict_class_proba(requirements, evidence)
        lsa_top = int(np.argmax(lsa))
        tfidf_top = int(np.argmax(tfidf))
        support = float(
            probabilities[lsa_top, direct_index]
            + probabilities[lsa_top, partial_index]
        )
        no_support = float(probabilities[lsa_top, no_support_index])
        lsa_score = float(lsa[lsa_top])
        lexical_score = float(tfidf[lsa_top])
        affinities = {
            "semantic_lexical_gap": (
                lsa_score - lexical_score + 0.4 * support
            ),
            "high_similarity_low_support": (
                lsa_score + lexical_score + no_support
            ),
            "rank_disagreement": (
                2.0 * float(lsa_top != tfidf_top)
                + abs(float(lsa[tfidf_top]) - lsa_score)
                + abs(float(tfidf[lsa_top]) - float(tfidf[tfidf_top]))
            ),
            "support_boundary": (
                1.0 - 2.0 * abs(support - 0.5) + 0.2 * lsa_score
            ),
        }
        scored.append(
            {
                "task": task,
                "affinities": affinities,
                "signals": {
                    "lsa_top_candidate_id": str(
                        candidates[lsa_top]["candidate_id"]
                    ),
                    "tfidf_top_candidate_id": str(
                        candidates[tfidf_top]["candidate_id"]
                    ),
                    "lsa_top_score": lsa_score,
                    "tfidf_at_lsa_top": lexical_score,
                    "support_probability_at_lsa_top": support,
                    "no_support_probability_at_lsa_top": no_support,
                },
            }
        )
    return scored


def select_balanced_acceptance_tasks(
    scored: Sequence[dict[str, Any]],
    *,
    tasks_per_family: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select equal role/stratum counts with unique job and resume sources."""
    if tasks_per_family % len(ACCEPTANCE_STRATA):
        raise ValueError("tasks_per_family must be divisible by four.")
    per_stratum = tasks_per_family // len(ACCEPTANCE_STRATA)
    selected: list[dict[str, Any]] = []
    construction: list[dict[str, Any]] = []
    used_tasks: set[str] = set()
    used_jobs: set[str] = set()
    used_resumes: set[str] = set()
    for family in ROLE_FAMILIES:
        family_rows = [
            row
            for row in scored
            if str(row["task"]["role_family"]) == family
        ]
        for _ in range(per_stratum):
            for stratum in ACCEPTANCE_STRATA:
                eligible = [
                    row
                    for row in family_rows
                    if str(row["task"]["task_id"]) not in used_tasks
                    and str(row["task"]["source_job_hash"]) not in used_jobs
                    and str(row["task"]["source_resume_hash"])
                    not in used_resumes
                ]
                if not eligible:
                    raise ValueError(
                        f"Insufficient unique {family} candidates for {stratum}."
                    )
                chosen = max(
                    eligible,
                    key=lambda row: (
                        float(row["affinities"][stratum]),
                        str(row["task"]["task_id"]),
                    ),
                )
                task = chosen["task"]
                selected.append(task)
                construction.append(
                    {
                        "task_id": str(task["task_id"]),
                        "stratum": stratum,
                        "role_family": family,
                        "selection_affinity": float(
                            chosen["affinities"][stratum]
                        ),
                        **chosen["signals"],
                    }
                )
                used_tasks.add(str(task["task_id"]))
                used_jobs.add(str(task["source_job_hash"]))
                used_resumes.add(str(task["source_resume_hash"]))
    return selected, construction


def _profile_choices(
    requirement: str,
    same_family: Sequence[dict[str, Any]],
    other_family: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    def scored(
        rows: Sequence[dict[str, Any]],
    ) -> list[tuple[float, dict[str, Any]]]:
        return [
            (
                max(
                    evidence_overlap(requirement, item)
                    for item in profile["evidence"]
                ),
                profile,
            )
            for profile in rows
        ]

    same = scored(same_family)
    cross = scored(other_family)
    choices = [
        max(same, key=lambda item: item[0], default=(0.0, None))[1],
        min(
            same,
            key=lambda item: abs(item[0] - 0.10),
            default=(0.0, None),
        )[1],
        min(
            same,
            key=lambda item: abs(item[0] - 0.03),
            default=(0.0, None),
        )[1],
        max(cross, key=lambda item: item[0], default=(0.0, None))[1],
    ]
    unique: dict[str, dict[str, Any]] = {}
    for profile in choices:
        if profile is not None:
            unique[str(profile["source_hash"])] = profile
    return list(unique.values())
