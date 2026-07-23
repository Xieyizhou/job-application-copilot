"""Quality diagnostics for synthetic evidence proposals and blind packets."""

from __future__ import annotations

from collections import Counter, defaultdict
import math
import re
from typing import Any, Hashable, Iterable, TypeVar

from ml.annotation_generation import normalize_text


BLIND_TASK_FIELDS = {
    "record_type",
    "schema_version",
    "review_task_id",
    "requirement",
    "candidates",
}
BLIND_CANDIDATE_FIELDS = {"candidate_id", "evidence"}
STYLE_LEAKAGE_WARNING_THRESHOLD = 0.15
POSITION_IMBALANCE_WARNING_THRESHOLD = 0.35
NEAR_DUPLICATE_THRESHOLD = 0.85
ACTION_VERBS = {
    "built",
    "created",
    "delivered",
    "designed",
    "developed",
    "implemented",
    "launched",
    "led",
    "managed",
    "migrated",
    "owned",
    "reduced",
    "shipped",
}
HashableItem = TypeVar("HashableItem", bound=Hashable)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9+#.]+", normalize_text(text))


def _trigrams(text: str) -> set[tuple[str, str, str]]:
    tokens = _tokens(text)
    return {
        (tokens[index], tokens[index + 1], tokens[index + 2])
        for index in range(max(0, len(tokens) - 2))
    }


def _jaccard(left: set[HashableItem], right: set[HashableItem]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _style_signature(requirement: str) -> str:
    tokens = _tokens(requirement)
    length_bucket = min(len(tokens) // 4, 5)
    first = tokens[0] if tokens else ""
    start_style = "action" if first in ACTION_VERBS else "other"
    compound = "compound" if {"and", "both", "plus"} & set(tokens) else "single"
    quantifier = "quantity" if re.search(r"\d|at least|minimum", requirement.lower()) else "none"
    punctuation = "punctuated" if re.search(r"[:;()/]", requirement) else "plain"
    return f"{length_bucket}:{start_style}:{compound}:{quantifier}:{punctuation}"


def _candidate_style_features(evidence: str) -> dict[str, str]:
    tokens = _tokens(evidence)
    first = tokens[0] if tokens else ""
    return {
        "length_bucket": str(min(len(tokens) // 4, 5)),
        "action_opening": "action" if first in ACTION_VERBS else "other",
        "contains_number": "number" if re.search(r"\d", evidence) else "none",
        "marked_punctuation": (
            "marked" if re.search(r"[:;()/]", evidence) else "plain"
        ),
    }


def cramers_v(rows: Iterable[tuple[str, str]]) -> float:
    """Return bias-corrected Cramér's V for two categorical variables."""
    pairs = list(rows)
    if not pairs:
        return 0.0
    left_values = sorted({left for left, _ in pairs})
    right_values = sorted({right for _, right in pairs})
    if len(left_values) < 2 or len(right_values) < 2:
        return 0.0
    left_counts = Counter(left for left, _ in pairs)
    right_counts = Counter(right for _, right in pairs)
    joint_counts = Counter(pairs)
    total = len(pairs)
    chi_square = 0.0
    for left in left_values:
        for right in right_values:
            expected = left_counts[left] * right_counts[right] / total
            if expected:
                observed = joint_counts[(left, right)]
                chi_square += (observed - expected) ** 2 / expected
    phi_square = chi_square / total
    rows_count = len(left_values)
    columns_count = len(right_values)
    correction = ((columns_count - 1) * (rows_count - 1)) / max(total - 1, 1)
    corrected_phi = max(0.0, phi_square - correction)
    corrected_rows = rows_count - ((rows_count - 1) ** 2) / max(total - 1, 1)
    corrected_columns = columns_count - (
        (columns_count - 1) ** 2
    ) / max(total - 1, 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return math.sqrt(corrected_phi / denominator) if denominator > 0 else 0.0


def blind_packet_leaks(tasks: Iterable[dict[str, Any]]) -> list[str]:
    """Return schema violations that could reveal producer intent to a reviewer."""
    leaks: list[str] = []
    for task in tasks:
        task_id = str(task.get("review_task_id", "missing"))
        extra_fields = set(task) - BLIND_TASK_FIELDS
        missing_fields = BLIND_TASK_FIELDS - set(task)
        if extra_fields:
            leaks.append(f"{task_id} has disallowed fields: {sorted(extra_fields)}")
        if missing_fields:
            leaks.append(f"{task_id} is missing fields: {sorted(missing_fields)}")
        for candidate in task.get("candidates", []):
            candidate_extra = set(candidate) - BLIND_CANDIDATE_FIELDS
            candidate_missing = BLIND_CANDIDATE_FIELDS - set(candidate)
            if candidate_extra or candidate_missing:
                leaks.append(f"{task_id} has invalid candidate fields.")
    return leaks


def _near_duplicate_pairs(cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    documents: list[tuple[str, str, set[tuple[str, str, str]]]] = []
    for case in cases:
        case_id = str(case["case_id"])
        documents.append((case_id, "requirement", _trigrams(str(case["requirement"]))))
        for candidate in case["candidates"]:
            documents.append(
                (
                    case_id,
                    str(candidate["candidate_id"]),
                    _trigrams(str(candidate["evidence"])),
                )
            )
    matches: list[dict[str, Any]] = []
    for index, left in enumerate(documents):
        for right in documents[index + 1 :]:
            if left[0] == right[0]:
                continue
            similarity = _jaccard(left[2], right[2])
            if similarity >= NEAR_DUPLICATE_THRESHOLD:
                matches.append(
                    {
                        "left_case_id": left[0],
                        "left_field": left[1],
                        "right_case_id": right[0],
                        "right_field": right[1],
                        "similarity": round(similarity, 4),
                    }
                )
    return matches


def audit_proposal_pool(
    cases: list[dict[str, Any]],
    *,
    rejected_records: list[str] | None = None,
) -> dict[str, Any]:
    """Return aggregate diagnostics without treating producer intent as truth."""
    positions: Counter[str] = Counter()
    for case in cases:
        intended_id = case.get("intended_candidate_id")
        if intended_id is None:
            positions["None"] += 1
            continue
        for index, candidate in enumerate(case["candidates"]):
            if candidate["candidate_id"] == intended_id:
                positions["ABCD"[index]] += 1
                break
    supported_total = sum(positions[position] for position in "ABCD")
    max_supported_position_share = (
        max((positions[position] for position in "ABCD"), default=0) / supported_total
        if supported_total
        else 0.0
    )
    style_v = cramers_v(
        (
            _style_signature(str(case["requirement"])),
            str(case["intended_label"]),
        )
        for case in cases
    )
    candidate_style_rows: list[tuple[str, dict[str, str]]] = []
    for case in cases:
        intended_id = case.get("intended_candidate_id")
        for candidate in case["candidates"]:
            candidate_relation = (
                "intended_support"
                if intended_id is not None
                and candidate["candidate_id"] == intended_id
                else "unselected"
            )
            candidate_style_rows.append(
                (
                    candidate_relation,
                    _candidate_style_features(str(candidate["evidence"])),
                )
            )
    candidate_style_values = {
        feature: cramers_v(
            (features[feature], relation)
            for relation, features in candidate_style_rows
        )
        for feature in (
            "length_bucket",
            "action_opening",
            "contains_number",
            "marked_punctuation",
        )
    }
    max_candidate_style_v = max(candidate_style_values.values(), default=0.0)
    near_duplicates = _near_duplicate_pairs(cases)
    warnings: list[str] = []
    if style_v > STYLE_LEAKAGE_WARNING_THRESHOLD:
        warnings.append(
            "Requirement surface style is associated with producer intent above the "
            f"{STYLE_LEAKAGE_WARNING_THRESHOLD:.2f} diagnostic threshold."
        )
    if max_candidate_style_v > STYLE_LEAKAGE_WARNING_THRESHOLD:
        warnings.append(
            "Candidate surface features are associated with producer-selected evidence "
            f"above the {STYLE_LEAKAGE_WARNING_THRESHOLD:.2f} diagnostic threshold."
        )
    if max_supported_position_share > POSITION_IMBALANCE_WARNING_THRESHOLD:
        warnings.append(
            "Supported evidence position exceeds the "
            f"{POSITION_IMBALANCE_WARNING_THRESHOLD:.0%} balance threshold."
        )
    if near_duplicates:
        warnings.append("Near-duplicate content requires clustering before dataset splitting.")

    producer_barriers: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for case in cases:
        producer_barriers[str(case["producer_id"])][str(case["barrier_id"])] += 1
    return {
        "schema_version": 1,
        "case_count": len(cases),
        "rejected_record_count": len(rejected_records or []),
        "rejected_records": rejected_records or [],
        "producer_counts": dict(Counter(str(case["producer_id"]) for case in cases)),
        "producer_barriers": {
            producer: dict(counts)
            for producer, counts in producer_barriers.items()
        },
        "barrier_counts": dict(Counter(str(case["barrier_id"]) for case in cases)),
        "role_counts": dict(Counter(str(case["role_family"]) for case in cases)),
        "producer_intent_counts": dict(
            Counter(str(case["intended_label"]) for case in cases)
        ),
        "intended_position_counts": dict(positions),
        "max_supported_position_share": round(max_supported_position_share, 4),
        "requirement_style_intent_cramers_v": round(style_v, 4),
        "candidate_style_intent_cramers_v": {
            feature: round(value, 4)
            for feature, value in candidate_style_values.items()
        },
        "max_candidate_style_intent_cramers_v": round(max_candidate_style_v, 4),
        "near_duplicate_pairs": near_duplicates,
        "warnings": warnings,
        "gold_policy": (
            "Producer intent is diagnostic metadata only; blind reviewer consensus or "
            "human adjudication is required for gold promotion."
        ),
    }
