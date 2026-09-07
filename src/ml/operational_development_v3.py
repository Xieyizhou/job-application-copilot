"""Contracts for blind, source-isolated operational development v3 data."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import math
import random
from typing import Any

from ml.annotation import validate_queue
from document_text import normalize_comparison_text as normalize_text
from ml.candidate_judgments import (
    normalized_candidate_labels,
    validate_complete_candidate_labels,
)
from ml.evidence_text import concept_tags, useful_tokens
from ml.evidence_constraints import named_terms, term_families
from ml.real_development import ROLE_FAMILIES, evidence_overlap
from ml.real_development_isolation import (
    DevelopmentIsolationIndex,
    development_task_isolated,
)


CONSTRUCTION_STRATA = (
    "explicit_aligned",
    "semantic_aligned",
    "compound_requirement",
    "same_role_adjacent",
    "same_role_action_mismatch",
)
TAXONOMY_REFERENCE = ("O*NET 30.3", "ESCO 1.2.1")
REVIEWER_VISIBLE_FIELDS = {
    "schema_version",
    "task_id",
    "presentation_id",
    "role_family",
    "requirement",
    "candidates",
}
REVIEWER_FORBIDDEN_FIELDS = {
    "construction_stratum",
    "taxonomy_reference",
    "source_dataset",
    "source_dataset_revision",
    "source_resume_hash",
    "source_job_hash",
    "operational_resume_group",
    "hidden_repeat_of",
    "blind_duplicate_of",
    "pairing_type",
    "model_prediction",
    "retrieval_score",
    "similarity",
}


def construction_features(
    requirement: str,
    evidence: Sequence[str],
) -> dict[str, float | int]:
    """Return transparent sampling features; these never become labels."""
    lexical = max(
        (evidence_overlap(requirement, candidate) for candidate in evidence),
        default=0.0,
    )
    requirement_terms = named_terms(requirement)
    requirement_families = term_families(requirement)
    requirement_concepts = concept_tags(requirement)
    taxonomy_overlap = 0
    for candidate in evidence:
        taxonomy_overlap = max(
            taxonomy_overlap,
            int(bool(requirement_terms & named_terms(candidate)))
            + int(bool(requirement_families & term_families(candidate)))
            + int(bool(requirement_concepts & concept_tags(candidate))),
        )
    normalized = normalize_text(requirement)
    compound = int(
        any(token in normalized for token in (" and ", " or ", ","))
        or len(requirement_terms) >= 2
        or len(requirement_families) >= 2
    )
    return {
        "lexical": lexical,
        "taxonomy_overlap": taxonomy_overlap,
        "compound": compound,
        "token_count": len(useful_tokens(requirement)),
    }


def ordered_jobs_for_stratum(
    jobs: Sequence[dict[str, Any]],
    evidence: Sequence[str],
    *,
    stratum: str,
    seed: str,
) -> list[dict[str, Any]]:
    """Order same-role jobs using auditable heuristics, never model scores."""
    if stratum not in CONSTRUCTION_STRATA:
        raise ValueError(f"Unknown construction stratum: {stratum}")
    ranked: list[tuple[tuple[float, ...], str, dict[str, Any]]] = []
    for job in jobs:
        requirement = str(job["requirement"])
        features = construction_features(requirement, evidence)
        lexical = float(features["lexical"])
        taxonomy = float(features["taxonomy_overlap"])
        compound = float(features["compound"])
        tokens = float(features["token_count"])
        key: tuple[float, ...]
        if stratum == "explicit_aligned":
            key = (lexical, taxonomy, -abs(tokens - 12.0))
        elif stratum == "semantic_aligned":
            key = (taxonomy, -lexical, -abs(tokens - 15.0))
        elif stratum == "compound_requirement":
            key = (compound, taxonomy, lexical, tokens)
        elif stratum == "same_role_adjacent":
            key = (
                taxonomy,
                -abs(lexical - 0.10),
                -abs(tokens - 14.0),
            )
        else:
            key = (-lexical, -taxonomy, -compound, -tokens)
        tie_break = hashlib.sha256(
            f"{seed}:{job['source_hash']}".encode()
        ).hexdigest()
        ranked.append((key, tie_break, job))
    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [job for _, _, job in ranked]


def validate_v3_contract(
    tasks: Sequence[dict[str, Any]],
    *,
    isolation_index: DevelopmentIsolationIndex,
    resumes_per_family: int = 6,
    tasks_per_resume: int = 5,
) -> dict[str, Any]:
    """Validate the fixed 24-resume, 120-task v3 construction contract."""
    expected_groups = resumes_per_family * len(ROLE_FAMILIES)
    expected_tasks = expected_groups * tasks_per_resume
    if len(tasks) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} tasks, found {len(tasks)}.")
    task_ids = [str(task["task_id"]) for task in tasks]
    job_hashes = [str(task["source_job_hash"]) for task in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("Operational v3 task ids must be unique.")
    if len(job_hashes) != len(set(job_hashes)):
        raise ValueError("Operational v3 job hashes must be unique.")
    if any(not development_task_isolated(task, isolation_index) for task in tasks):
        raise ValueError("Operational v3 overlaps consumed source or text.")
    resume_counts = Counter(str(task["source_resume_hash"]) for task in tasks)
    if len(resume_counts) != expected_groups or set(resume_counts.values()) != {
        tasks_per_resume
    }:
        raise ValueError("Every operational v3 resume must own exactly five tasks.")
    role_counts = Counter(str(task["role_family"]) for task in tasks)
    expected_role_tasks = resumes_per_family * tasks_per_resume
    if any(role_counts[family] != expected_role_tasks for family in ROLE_FAMILIES):
        raise ValueError("Operational v3 role families are not balanced.")
    group_roles: dict[str, set[str]] = {}
    group_strata: dict[str, set[str]] = {}
    for task in tasks:
        group = str(task["operational_resume_group"])
        group_roles.setdefault(group, set()).add(str(task["role_family"]))
        group_strata.setdefault(group, set()).add(
            str(task["construction_stratum"])
        )
        if len(task["candidates"]) != 4:
            raise ValueError("Every operational v3 task needs four candidates.")
    if any(len(roles) != 1 for roles in group_roles.values()):
        raise ValueError("A resume group cannot cross role families.")
    expected_strata = set(CONSTRUCTION_STRATA)
    if any(strata != expected_strata for strata in group_strata.values()):
        raise ValueError("Every resume group must contain all five strata.")
    return {
        "tasks": len(tasks),
        "resume_groups": len(resume_counts),
        "tasks_per_resume": tasks_per_resume,
        "candidate_count": sum(len(task["candidates"]) for task in tasks),
        "role_counts": dict(role_counts),
        "construction_stratum_counts": dict(
            Counter(str(task["construction_stratum"]) for task in tasks)
        ),
        "source_job_overlap": 0,
        "source_resume_overlap": 0,
        "content_overlap": 0,
        "resume_group_split_required": True,
    }


def _reviewer_task(
    task: Mapping[str, Any],
    *,
    presentation_id: str,
    random_state: int,
) -> dict[str, Any]:
    visible = {
        field: deepcopy(task[field])
        for field in REVIEWER_VISIBLE_FIELDS
        if field in task and field not in {"task_id", "presentation_id"}
    }
    visible["task_id"] = presentation_id
    visible["presentation_id"] = presentation_id
    candidates = list(visible["candidates"])
    random.Random(
        f"{random_state}:{presentation_id}:candidate-order"
    ).shuffle(candidates)
    source_order = [
        str(candidate["candidate_id"]) for candidate in task["candidates"]
    ]
    reviewer_order = [
        str(candidate["candidate_id"]) for candidate in candidates
    ]
    if reviewer_order == source_order and len(candidates) > 1:
        candidates = [*candidates[1:], candidates[0]]
    visible["candidates"] = candidates
    return visible


def build_single_reviewer_packet(
    tasks: Sequence[dict[str, Any]],
    *,
    repeat_count: int = 12,
    random_state: int,
    presentation_prefix: str = "op-v3-presentation",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Create a blind packet with hidden repeats and a private repeat map."""
    checked = validate_queue(tasks)
    if repeat_count <= 0 or repeat_count >= len(checked):
        raise ValueError("Repeat count must be positive and smaller than the dataset.")
    originals = [
        _reviewer_task(
            task,
            presentation_id=str(task["task_id"]),
            random_state=random_state,
        )
        for task in checked
    ]
    chosen = random.Random(f"{random_state}:repeat-sample").sample(
        checked,
        repeat_count,
    )
    repeats: list[dict[str, Any]] = []
    repeat_map: list[dict[str, str]] = []
    for index, task in enumerate(chosen, start=1):
        canonical_id = str(task["task_id"])
        presentation_id = (
            f"{presentation_prefix}-"
            + hashlib.sha256(
                f"{random_state}:{canonical_id}:{index}".encode()
            ).hexdigest()[:16]
        )
        repeated = _reviewer_task(
            task,
            presentation_id=presentation_id,
            random_state=random_state + 10_000,
        )
        original = next(
            row for row in originals if row["task_id"] == canonical_id
        )
        repeated_ids = [
            str(candidate["candidate_id"])
            for candidate in repeated["candidates"]
        ]
        original_ids = [
            str(candidate["candidate_id"])
            for candidate in original["candidates"]
        ]
        if repeated_ids == original_ids:
            repeated["candidates"] = [
                *repeated["candidates"][1:],
                repeated["candidates"][0],
            ]
        repeats.append(repeated)
        repeat_map.append(
            {
                "presentation_id": presentation_id,
                "original_presentation_id": canonical_id,
                "canonical_task_id": canonical_id,
                "hidden_repeat_of": canonical_id,
            }
        )
    packet = [*originals, *repeats]
    random.Random(f"{random_state}:reviewer-task-order").shuffle(packet)
    validate_reviewer_packet(packet)
    return packet, sorted(repeat_map, key=lambda row: row["presentation_id"])


def build_blind_reviewer_queue(
    tasks: Sequence[dict[str, Any]],
    *,
    reviewer_id: str,
    random_state: int,
) -> list[dict[str, Any]]:
    """Create one metadata-free reviewer queue without reading other decisions."""
    if not reviewer_id.strip():
        raise ValueError("Reviewer id is required.")
    checked = validate_queue(tasks)
    packet = [
        _reviewer_task(
            task,
            presentation_id=str(task["task_id"]),
            random_state=int(
                hashlib.sha256(
                    f"{random_state}:{reviewer_id}".encode()
                ).hexdigest()[:8],
                16,
            ),
        )
        for task in checked
    ]
    random.Random(
        f"{random_state}:{reviewer_id}:reviewer-task-order"
    ).shuffle(packet)
    validate_reviewer_packet(packet)
    return packet


def validate_reviewer_packet(tasks: Sequence[dict[str, Any]]) -> None:
    """Reject source, construction, taxonomy, repeat, or model metadata."""
    validate_queue(tasks)
    for task in tasks:
        forbidden = set(task) & REVIEWER_FORBIDDEN_FIELDS
        if forbidden:
            raise ValueError(
                "Reviewer packet exposes forbidden fields: "
                + ", ".join(sorted(forbidden))
            )
        if set(task) - REVIEWER_VISIBLE_FIELDS:
            raise ValueError("Reviewer packet contains an unknown visible field.")
        for candidate in task["candidates"]:
            if set(candidate) != {"candidate_id", "evidence"}:
                raise ValueError("Reviewer candidates expose hidden metadata.")


def _decision_signature(state: Mapping[str, Any]) -> tuple[object, ...]:
    labels = tuple(sorted(normalized_candidate_labels(
        state.get("candidate_labels")
    ).items()))
    return (
        str(state.get("support_label", "")),
        state.get("selected_candidate_id"),
        labels,
    )


def cohen_kappa(left: Sequence[str], right: Sequence[str]) -> float:
    labels = sorted(set(left) | set(right))
    if not left or len(left) != len(right):
        raise ValueError("Cohen kappa needs aligned non-empty labels.")
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        left_counts[label] / len(left) * right_counts[label] / len(right)
        for label in labels
    )
    if math.isclose(expected, 1.0):
        return 1.0 if math.isclose(observed, 1.0) else 0.0
    return (observed - expected) / (1.0 - expected)


def repeat_agreement_report(
    source_tasks: Sequence[dict[str, Any]],
    repeat_map: Sequence[dict[str, str]],
    states: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Freeze pre-adjudication repeat consistency without exposing text."""
    source_by_id = {str(task["task_id"]): task for task in source_tasks}
    left_labels: list[str] = []
    right_labels: list[str] = []
    exact = 0
    disagreements: list[str] = []
    for repeat in repeat_map:
        canonical_id = str(repeat["canonical_task_id"])
        original_id = str(repeat["original_presentation_id"])
        repeated_id = str(repeat["presentation_id"])
        if canonical_id not in source_by_id:
            raise ValueError("Repeat map references an unknown canonical task.")
        if original_id not in states or repeated_id not in states:
            raise ValueError("Every hidden repeat presentation must be labeled.")
        original = states[original_id]
        repeated = states[repeated_id]
        task = source_by_id[canonical_id]
        original_candidate_labels = normalized_candidate_labels(
            original.get("candidate_labels")
        )
        repeated_candidate_labels = normalized_candidate_labels(
            repeated.get("candidate_labels")
        )
        validate_complete_candidate_labels(
            task,
            original_candidate_labels,
            (
                str(original["selected_candidate_id"])
                if original.get("selected_candidate_id") is not None
                else None
            ),
        )
        validate_complete_candidate_labels(
            task,
            repeated_candidate_labels,
            (
                str(repeated["selected_candidate_id"])
                if repeated.get("selected_candidate_id") is not None
                else None
            ),
        )
        for candidate in task["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            left_labels.append(original_candidate_labels[candidate_id])
            right_labels.append(repeated_candidate_labels[candidate_id])
        agrees = _decision_signature(original) == _decision_signature(repeated)
        exact += int(agrees)
        if not agrees:
            disagreements.append(canonical_id)
    task_rate = exact / len(repeat_map)
    kappa = cohen_kappa(left_labels, right_labels)
    return {
        "schema_version": 1,
        "repeat_pairs": len(repeat_map),
        "candidate_judgments_compared": len(left_labels),
        "candidate_cohen_kappa": kappa,
        "task_exact_agreements": exact,
        "task_exact_agreement_rate": task_rate,
        "disagreement_tasks": len(disagreements),
        "disagreement_task_ids": sorted(disagreements),
        "candidate_kappa_gate": kappa >= 0.70,
        "task_exact_gate": task_rate >= 0.80,
        "agreement_gate_passed": kappa >= 0.70 and task_rate >= 0.80,
        "pre_adjudication_frozen": True,
    }


def build_self_adjudication_queue(
    source_tasks: Sequence[dict[str, Any]],
    disagreement_task_ids: Sequence[str],
    *,
    random_state: int,
) -> list[dict[str, Any]]:
    """Create a content-only queue without showing either prior decision."""
    by_id = {str(task["task_id"]): task for task in source_tasks}
    requested = {str(task_id) for task_id in disagreement_task_ids}
    if requested - set(by_id):
        raise ValueError("Adjudication references an unknown canonical task.")
    queue = [
        _reviewer_task(
            by_id[task_id],
            presentation_id=task_id,
            random_state=random_state,
        )
        for task_id in sorted(requested)
    ]
    random.Random(f"{random_state}:self-adjudication-order").shuffle(queue)
    validate_reviewer_packet(queue)
    return queue


def resolved_single_reviewer_states(
    source_tasks: Sequence[dict[str, Any]],
    original_states: Mapping[str, Mapping[str, Any]],
    adjudication_states: Mapping[str, Mapping[str, Any]],
    repeat_report: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Use original decisions except where frozen repeat conflicts were adjudicated."""
    disagreements = {
        str(task_id)
        for task_id in repeat_report.get("disagreement_task_ids", [])
    }
    resolved: dict[str, dict[str, Any]] = {}
    for task in source_tasks:
        task_id = str(task["task_id"])
        source = (
            adjudication_states.get(task_id)
            if task_id in disagreements
            else original_states.get(task_id)
        )
        if source is None:
            raise ValueError(f"Task {task_id} has no final human decision.")
        resolved[task_id] = dict(source)
    if disagreements - set(adjudication_states):
        raise ValueError("Every repeat disagreement requires self-adjudication.")
    return resolved


def sufficiency_report(tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Evaluate natural label coverage without selecting or dropping rows."""
    label_counts = Counter(str(task["support_label"]) for task in tasks)
    role_labels: dict[str, Counter[str]] = {
        family: Counter() for family in ROLE_FAMILIES
    }
    for task in tasks:
        role_labels[str(task["role_family"])][
            str(task["support_label"])
        ] += 1
    supported = label_counts["Direct"] + label_counts["Partial"]
    checks = {
        "supported_tasks_at_least_48": supported >= 48,
        "direct_tasks_at_least_15": label_counts["Direct"] >= 15,
        "partial_tasks_at_least_20": label_counts["Partial"] >= 20,
        "no_support_tasks_at_least_30": label_counts["No Support"] >= 30,
        "every_role_has_10_supported": all(
            counts["Direct"] + counts["Partial"] >= 10
            for counts in role_labels.values()
        ),
        "every_role_has_6_no_support": all(
            counts["No Support"] >= 6 for counts in role_labels.values()
        ),
    }
    return {
        "task_label_counts": dict(label_counts),
        "supported_tasks": supported,
        "role_label_counts": {
            family: dict(counts) for family, counts in role_labels.items()
        },
        "checks": checks,
        "dataset_sufficiency_gate_passed": all(checks.values()),
        "post_label_selection_performed": False,
    }
