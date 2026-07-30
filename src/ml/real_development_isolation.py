"""Leakage checks for real-text development extensions."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from ml.annotation_generation import normalize_text
from ml.evidence import useful_tokens
from ml.real_development import DevelopmentDataError


@dataclass(frozen=True)
class DevelopmentIsolationIndex:
    """Precomputed consumed sources and texts for repeated pool screening."""

    job_hashes: frozenset[str]
    resume_hashes: frozenset[str]
    requirement_normalized: frozenset[str]
    requirement_tokens: tuple[frozenset[str], ...]
    evidence_normalized: frozenset[str]
    evidence_tokens: tuple[frozenset[str], ...]


def build_development_isolation_index(
    excluded_tasks: Iterable[dict[str, Any]],
    excluded_pairs: Iterable[dict[str, Any]],
) -> DevelopmentIsolationIndex:
    """Index consumed data once without retaining any additional source fields."""
    excluded = list(excluded_tasks)
    pairs = list(excluded_pairs)
    requirements = [
        str(task.get("requirement", "")) for task in excluded
    ]
    evidence = [
        str(candidate.get("evidence", ""))
        for task in excluded
        for candidate in task.get("candidates", [])
    ]
    for pair in pairs:
        requirements.append(str(pair.get("requirement", "")))
        evidence.append(str(pair.get("evidence", "")))
    requirement_normalized, requirement_tokens = _index_texts(requirements)
    evidence_normalized, evidence_tokens = _index_texts(evidence)
    return DevelopmentIsolationIndex(
        job_hashes=frozenset(
            str(task.get("source_job_hash", "")) for task in excluded
        ),
        resume_hashes=frozenset(
            str(task.get("source_resume_hash", "")) for task in excluded
        ),
        requirement_normalized=requirement_normalized,
        requirement_tokens=requirement_tokens,
        evidence_normalized=evidence_normalized,
        evidence_tokens=evidence_tokens,
    )


def development_task_isolated(
    task: dict[str, Any],
    index: DevelopmentIsolationIndex,
) -> bool:
    """Return whether one task is source- and text-isolated from consumed data."""
    if (
        str(task["source_job_hash"]) in index.job_hashes
        or str(task["source_resume_hash"]) in index.resume_hashes
        or _overlaps_index(
            str(task["requirement"]),
            index.requirement_normalized,
            index.requirement_tokens,
        )
    ):
        return False
    return all(
        not _overlaps_index(
            str(candidate["evidence"]),
            index.evidence_normalized,
            index.evidence_tokens,
        )
        for candidate in task["candidates"]
    )


def assert_development_isolated(
    tasks: Sequence[dict[str, Any]],
    excluded_tasks: Iterable[dict[str, Any]],
    excluded_pairs: Iterable[dict[str, Any]],
) -> dict[str, int]:
    """Reject source, exact-text, or near-text overlap with consumed data."""
    excluded = list(excluded_tasks)
    pairs = list(excluded_pairs)
    index = build_development_isolation_index(excluded, pairs)

    seen_jobs: set[str] = set()
    seen_resumes: set[str] = set()
    for task in tasks:
        job_hash = str(task["source_job_hash"])
        resume_hash = str(task["source_resume_hash"])
        if job_hash in index.job_hashes or resume_hash in index.resume_hashes:
            raise DevelopmentDataError("Development source overlaps consumed data.")
        if job_hash in seen_jobs or resume_hash in seen_resumes:
            raise DevelopmentDataError(
                "Each development task must use a distinct job and resume."
            )
        seen_jobs.add(job_hash)
        seen_resumes.add(resume_hash)
        if _overlaps_index(
            str(task["requirement"]),
            index.requirement_normalized,
            index.requirement_tokens,
        ):
            raise DevelopmentDataError(
                "Development requirement overlaps consumed text."
            )
        for candidate in task["candidates"]:
            if _overlaps_index(
                str(candidate["evidence"]),
                index.evidence_normalized,
                index.evidence_tokens,
            ):
                raise DevelopmentDataError(
                    "Development evidence overlaps consumed text."
                )
    return {
        "source_job_overlap": 0,
        "source_resume_overlap": 0,
        "requirement_overlap": 0,
        "evidence_overlap": 0,
    }


def _index_texts(
    texts: Sequence[str],
) -> tuple[frozenset[str], tuple[frozenset[str], ...]]:
    usable = [text for text in texts if text]
    return (
        frozenset(normalize_text(text) for text in usable),
        tuple(frozenset(useful_tokens(text)) for text in usable),
    )


def _overlaps_index(
    text: str,
    normalized: frozenset[str],
    token_sets: Sequence[frozenset[str]],
) -> bool:
    if normalize_text(text) in normalized:
        return True
    tokens = frozenset(useful_tokens(text))
    return any(
        bool(tokens | other)
        and len(tokens & other) / len(tokens | other) >= 0.88
        for other in token_sets
    )
