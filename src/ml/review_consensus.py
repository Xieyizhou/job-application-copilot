"""Blind-queue preparation and consensus routing for barrier datasets."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import random
from typing import Any, Iterable

from ml.annotation import SCHEMA_VERSION
from ml.barrier_contracts import (
    BARRIER_SCHEMA_VERSION,
    BarrierContractError,
    validate_generated_case,
    validate_review,
)
from ml.annotation_generation import normalize_text


REJECT_FLAGS = {"privacy_risk", "near_duplicate"}
HUMAN_REVIEW_FLAGS = {
    "ambiguous_requirement",
    "keyword_copy_leakage",
    "multiple_plausible_candidates",
    "surface_style_leakage",
    "unsupported_inference",
    "rubric_conflict",
    "constraint_mismatch",
}


def merge_generated_cases(records: Iterable[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate records and reject content duplicates across producers."""
    merged: list[dict[str, Any]] = []
    rejected: list[str] = []
    seen_case_ids: set[str] = set()
    seen_requirements: set[str] = set()
    for raw in records:
        try:
            case = validate_generated_case(raw)
        except BarrierContractError as error:
            rejected.append(str(error))
            continue
        requirement_key = normalize_text(str(case["requirement"]))
        if case["case_id"] in seen_case_ids or requirement_key in seen_requirements:
            rejected.append(f"Duplicate generated case: {case['case_id']}")
            continue
        seen_case_ids.add(str(case["case_id"]))
        seen_requirements.add(requirement_key)
        merged.append(case)
    return merged, rejected


def build_blind_queue(
    cases: Iterable[dict[str, Any]],
    *,
    reviewer_id: str,
    random_state: int = 42,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build allow-listed blind tasks plus a separate private id mapping."""
    queue: list[dict[str, Any]] = []
    mappings: list[dict[str, Any]] = []
    for case in cases:
        seed_material = f"{random_state}:{reviewer_id}:{case['case_id']}"
        seed = int(hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16], 16)
        review_task_id = "review-" + hashlib.sha256(seed_material.encode("utf-8")).hexdigest()[:16]
        candidates: list[dict[str, str]] = []
        candidate_map: dict[str, str] = {}
        for index, candidate in enumerate(case["candidates"]):
            blind_material = f"{seed_material}:{candidate['candidate_id']}:{index}"
            blind_id = "option-" + hashlib.sha256(
                blind_material.encode("utf-8")
            ).hexdigest()[:12]
            candidates.append({"candidate_id": blind_id, "evidence": candidate["evidence"]})
            candidate_map[blind_id] = str(candidate["candidate_id"])
        random.Random(seed).shuffle(candidates)
        queue.append(
            {
                "record_type": "blind_review_task",
                "schema_version": BARRIER_SCHEMA_VERSION,
                "review_task_id": review_task_id,
                "requirement": case["requirement"],
                "candidates": candidates,
            }
        )
        mappings.append(
            {
                "schema_version": BARRIER_SCHEMA_VERSION,
                "review_task_id": review_task_id,
                "case_id": case["case_id"],
                "reviewer_id": reviewer_id,
                "candidate_id_map": candidate_map,
            }
        )
    random.Random(f"{random_state}:{reviewer_id}").shuffle(queue)
    return queue, mappings


def _decision(review: dict[str, Any]) -> tuple[str, str | None]:
    return str(review["support_label"]), review.get("selected_candidate_id")


def fleiss_kappa(label_groups: Iterable[Iterable[str]]) -> float | None:
    """Return Fleiss' kappa for subjects with at least two categorical ratings."""
    groups = [list(labels) for labels in label_groups]
    groups = [labels for labels in groups if len(labels) >= 2]
    if not groups:
        return None
    total_ratings = sum(len(labels) for labels in groups)
    label_totals = Counter(label for labels in groups for label in labels)
    observed_agreement = sum(
        (
            sum(count**2 for count in Counter(labels).values()) - len(labels)
        )
        / (len(labels) * (len(labels) - 1))
        for labels in groups
    ) / len(groups)
    expected_agreement = sum(
        (count / total_ratings) ** 2
        for count in label_totals.values()
    )
    if expected_agreement >= 1.0:
        return 1.0
    return (observed_agreement - expected_agreement) / (1.0 - expected_agreement)


def unblind_review(
    blind_review: dict[str, Any],
    mapping: dict[str, Any],
    case: dict[str, Any],
) -> dict[str, Any]:
    """Map one reviewer packet back to canonical ids before validation."""
    if blind_review.get("review_task_id") != mapping.get("review_task_id"):
        raise BarrierContractError("Review task does not match its private mapping.")
    candidate_map = dict(mapping["candidate_id_map"])
    selected_blind = blind_review.get("selected_candidate_id")
    selected_canonical = (
        candidate_map.get(str(selected_blind))
        if selected_blind is not None
        else None
    )
    raw_judgments = blind_review.get("candidate_judgments", [])
    if isinstance(raw_judgments, dict):
        judgment_items = [
            {"candidate_id": candidate_id, "support_label": support_label}
            for candidate_id, support_label in raw_judgments.items()
        ]
    elif isinstance(raw_judgments, list):
        judgment_items = raw_judgments
    else:
        raise BarrierContractError("Candidate judgments must be a list or id-label object.")
    judgments = [
        {
            "candidate_id": candidate_map.get(str(judgment.get("candidate_id", "")), ""),
            "support_label": judgment.get("support_label"),
        }
        for judgment in judgment_items
        if isinstance(judgment, dict)
    ]
    return validate_review(
        {
            "reviewer_id": mapping["reviewer_id"],
            "support_label": blind_review.get("support_label"),
            "selected_candidate_id": selected_canonical,
            "candidate_judgments": judgments,
            "flags": blind_review.get("flags", []),
        },
        case,
    )


def promote_consensus_gold(
    cases: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
    raw_reviews: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Promote only unanimously accepted blind-review decisions into gold tasks."""
    case_by_id = {str(case["case_id"]): case for case in cases}
    accepted_by_id = {
        str(decision["case_id"]): decision
        for decision in decisions
        if decision.get("status") == "accepted"
    }
    reviews_by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for raw_review in raw_reviews:
        case = case_by_id.get(str(raw_review.get("case_id", "")))
        if case is None:
            continue
        try:
            review = validate_review(raw_review, case)
        except BarrierContractError:
            continue
        reviews_by_case[str(case["case_id"])].append(review)

    gold_tasks: list[dict[str, Any]] = []
    for case_id, decision in accepted_by_id.items():
        case = case_by_id[case_id]
        reviews = reviews_by_case[case_id]
        if not reviews:
            continue
        candidate_labels: dict[str, str] = {}
        for candidate in case["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            labels = {
                str(judgment["support_label"])
                for review in reviews
                for judgment in review["candidate_judgments"]
                if judgment["candidate_id"] == candidate_id
            }
            if len(labels) != 1:
                raise BarrierContractError(
                    f"Accepted case {case_id} lacks unanimous candidate judgments."
                )
            candidate_labels[candidate_id] = labels.pop()
        gold_material = f"{case_id}:blind_consensus:{len(reviews)}"
        gold_tasks.append(
            {
                "record_type": "gold_task",
                "schema_version": BARRIER_SCHEMA_VERSION,
                "gold_id": "gold-"
                + hashlib.sha256(gold_material.encode("utf-8")).hexdigest()[:16],
                "case_id": case_id,
                "role_family": case["role_family"],
                "requirement": case["requirement"],
                "candidates": [
                    {
                        "candidate_id": candidate["candidate_id"],
                        "evidence": candidate["evidence"],
                        "support_label": candidate_labels[str(candidate["candidate_id"])],
                    }
                    for candidate in case["candidates"]
                ],
                "best_candidate_id": decision["consensus_candidate_id"],
                "support_label": decision["consensus_label"],
                "decision_source": "blind_consensus",
                "reviewer_count": len(reviews),
                "semantic_case_group_id": case["semantic_case_group_id"],
                "source_kind": case["source_kind"],
            }
        )
    return gold_tasks


def adjudicate_reviews(
    cases: list[dict[str, Any]],
    raw_reviews: Iterable[dict[str, Any]],
    *,
    minimum_reviewers: int = 3,
    minimum_exact_agreement: float = 1.0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Route cases to accepted, human-review, or rejected without generator voting."""
    case_by_id = {str(case["case_id"]): case for case in cases}
    reviews_by_case: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_reviewers: set[tuple[str, str]] = set()
    invalid_reviews: list[str] = []
    for raw_review in raw_reviews:
        case_id = str(raw_review.get("case_id", ""))
        case = case_by_id.get(case_id)
        if case is None:
            invalid_reviews.append(f"Review references unknown case: {case_id or 'missing'}")
            continue
        try:
            review = validate_review(raw_review, case)
        except BarrierContractError as error:
            invalid_reviews.append(str(error))
            continue
        reviewer_key = (case_id, str(review["reviewer_id"]))
        if reviewer_key in seen_reviewers:
            invalid_reviews.append(f"Duplicate reviewer decision: {reviewer_key}")
            continue
        seen_reviewers.add(reviewer_key)
        reviews_by_case[case_id].append(review)

    decisions: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    for case in cases:
        case_id = str(case["case_id"])
        reviews = reviews_by_case[case_id]
        decision_counts = Counter(_decision(review) for review in reviews)
        label_counts = Counter(str(review["support_label"]) for review in reviews)
        winning_decision, winning_count = decision_counts.most_common(1)[0] if reviews else (("", None), 0)
        exact_agreement = winning_count / len(reviews) if reviews else 0.0
        label_agreement = max(label_counts.values(), default=0) / len(reviews) if reviews else 0.0
        candidate_decisions: defaultdict[str, Counter[str]] = defaultdict(Counter)
        for review in reviews:
            for judgment in review["candidate_judgments"]:
                candidate_decisions[str(judgment["candidate_id"])][
                    str(judgment["support_label"])
                ] += 1
        candidate_label_agreement = (
            sum(max(counts.values()) / len(reviews) for counts in candidate_decisions.values())
            / len(candidate_decisions)
            if reviews and candidate_decisions
            else 0.0
        )
        flags = sorted({flag for review in reviews for flag in review["flags"]})
        if set(flags) & REJECT_FLAGS:
            status = "rejected"
        elif (
            len(reviews) < minimum_reviewers
            or exact_agreement < minimum_exact_agreement
            or candidate_label_agreement < 1.0
            or set(flags) & HUMAN_REVIEW_FLAGS
        ):
            status = "human_review"
        else:
            status = "accepted"
        status_counts[status] += 1
        support_label, selected_candidate_id = winning_decision
        decisions.append(
            {
                "record_type": "consensus_decision",
                "schema_version": BARRIER_SCHEMA_VERSION,
                "case_id": case_id,
                "status": status,
                "review_count": len(reviews),
                "exact_agreement": exact_agreement,
                "label_agreement": label_agreement,
                "candidate_label_agreement": candidate_label_agreement,
                "reviewer_flags": flags,
                "consensus_label": support_label or None,
                "consensus_candidate_id": selected_candidate_id,
                "barrier_id": case["barrier_id"],
                "role_family": case["role_family"],
            }
        )
    report = {
        "schema_version": BARRIER_SCHEMA_VERSION,
        "cases": len(cases),
        "status_counts": dict(status_counts),
        "review_count": sum(len(items) for items in reviews_by_case.values()),
        "invalid_reviews": invalid_reviews,
        "barrier_counts": dict(Counter(str(case["barrier_id"]) for case in cases)),
        "producer_counts": dict(Counter(str(case["producer_id"]) for case in cases)),
    }
    overall_groups = [
        [str(review["support_label"]) for review in reviews_by_case[str(case["case_id"])]]
        for case in cases
    ]
    candidate_groups: list[list[str]] = []
    for case in cases:
        reviews = reviews_by_case[str(case["case_id"])]
        for candidate in case["candidates"]:
            candidate_id = str(candidate["candidate_id"])
            candidate_groups.append(
                [
                    str(judgment["support_label"])
                    for review in reviews
                    for judgment in review["candidate_judgments"]
                    if judgment["candidate_id"] == candidate_id
                ]
            )
    overall_kappa = fleiss_kappa(overall_groups)
    candidate_kappa = fleiss_kappa(candidate_groups)
    report.update(
        {
            "overall_fleiss_kappa": (
                round(overall_kappa, 4)
                if overall_kappa is not None
                else None
            ),
            "candidate_fleiss_kappa": (
                round(candidate_kappa, 4)
                if candidate_kappa is not None
                else None
            ),
            "unanimous_overall_cases": sum(
                len(set(labels)) == 1
                for labels in overall_groups
                if labels
            ),
            "unanimous_candidate_subjects": sum(
                len(set(labels)) == 1
                for labels in candidate_groups
                if labels
            ),
        }
    )
    return decisions, report


def human_review_cases(
    cases: list[dict[str, Any]],
    decisions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return content-only cases that require independent human adjudication."""
    review_ids = {
        str(decision["case_id"])
        for decision in decisions
        if decision["status"] == "human_review"
    }
    return [
        {
            "schema_version": SCHEMA_VERSION,
            "task_id": case["case_id"],
            "role_family": case["role_family"],
            "requirement": case["requirement"],
            "candidates": case["candidates"],
            "source_resume_hash": "",
            "source_job_hash": "",
            "source_dataset": "barrier_human_review",
            "blind_duplicate_of": None,
        }
        for case in cases
        if case["case_id"] in review_ids
    ]
