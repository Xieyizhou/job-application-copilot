"""Provider-neutral contracts for barrier-based evidence dataset production."""

from __future__ import annotations

import hashlib
import re
from typing import Any, TypedDict

from ml.annotation_generation import normalize_text, sanitize_snippet


BARRIER_SCHEMA_VERSION = 1
SUPPORT_LABELS = {"Direct", "Partial", "No Support"}
ROLE_FAMILIES = {"Data", "ML", "Software", "Business"}
REVIEW_FLAGS = {
    "multiple_plausible_candidates",
    "surface_style_leakage",
    "unsupported_inference",
    "privacy_risk",
    "near_duplicate",
    "rubric_conflict",
    "constraint_mismatch",
    "ambiguous_requirement",
    "keyword_copy_leakage",
}
class BarrierDefinition(TypedDict):
    """Static metadata for one generation difficulty barrier."""

    difficulty: int
    allowed_labels: set[str]
    description: str


BARRIER_DEFINITIONS: dict[str, BarrierDefinition] = {
    "B1_EXACT_DIRECT": {
        "difficulty": 1,
        "allowed_labels": {"Direct"},
        "description": "Exact capability with independently verifiable hands-on delivery.",
    },
    "B2_SEMANTIC_DIRECT": {
        "difficulty": 2,
        "allowed_labels": {"Direct"},
        "description": "Direct support expressed without repeating the requirement terminology.",
    },
    "B3_ADJACENT_PARTIAL": {
        "difficulty": 3,
        "allowed_labels": {"Partial"},
        "description": "Adjacent capability or incomplete responsibility coverage.",
    },
    "B4_SAME_DOMAIN_NEGATIVE": {
        "difficulty": 4,
        "allowed_labels": {"No Support"},
        "description": "Topically similar work that does not establish the required capability.",
    },
    "B5_CROSS_DOMAIN_NEGATIVE": {
        "difficulty": 5,
        "allowed_labels": {"No Support"},
        "description": "Clear negative evidence from a different work domain.",
    },
    "B6_NON_PRACTICAL_MENTION": {
        "difficulty": 6,
        "allowed_labels": {"No Support"},
        "description": "Observation, planning, reading, or documentation without practical work.",
    },
    "B7_COMPOUND_PARTIAL": {
        "difficulty": 7,
        "allowed_labels": {"Partial"},
        "description": "Evidence covers only part of a compound requirement.",
    },
    "B8_CONSTRAINT_MISMATCH": {
        "difficulty": 8,
        "allowed_labels": {"Partial", "No Support"},
        "description": "Related evidence fails an ownership, scale, duration, or deployment constraint.",
    },
}
CONTACT_PATTERN = re.compile(
    r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.\w+|\+?\d[\d\s().-]{7,}\d)",
    re.IGNORECASE,
)


class BarrierContractError(ValueError):
    """Raised when a generated or reviewed record violates the local contract."""


def stable_case_id(case: dict[str, Any]) -> str:
    """Return a content-derived identifier independent of producer metadata."""
    candidate_text = "\0".join(
        normalize_text(str(candidate.get("evidence", "")))
        for candidate in case.get("candidates", [])
    )
    material = "\0".join(
        [
            normalize_text(str(case.get("requirement", ""))),
            candidate_text,
        ]
    )
    return "case-" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _validate_text(text: str, field: str, *, minimum_words: int) -> str:
    cleaned = sanitize_snippet(text)
    if CONTACT_PATTERN.search(text) or "[redacted]" in cleaned:
        raise BarrierContractError(f"{field} contains contact or URL-like content.")
    word_count = len(cleaned.split())
    if not minimum_words <= word_count <= 80:
        raise BarrierContractError(f"{field} must contain {minimum_words}–80 words.")
    return cleaned


def validate_generated_case(case: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize one producer record while retaining hidden design metadata."""
    if int(case.get("schema_version", 0)) != BARRIER_SCHEMA_VERSION:
        raise BarrierContractError("Unsupported generated-case schema.")
    barrier_id = str(case.get("barrier_id", ""))
    if barrier_id not in BARRIER_DEFINITIONS:
        raise BarrierContractError(f"Unknown barrier_id: {barrier_id or 'missing'}")
    role_family = str(case.get("role_family", ""))
    if role_family not in ROLE_FAMILIES:
        raise BarrierContractError(f"Unknown role_family: {role_family or 'missing'}")
    producer_id = str(case.get("producer_id", "")).strip()
    run_id = str(case.get("generation_run_id", "")).strip()
    semantic_group = str(case.get("semantic_case_group_id", "")).strip()
    if not producer_id or not run_id or not semantic_group:
        raise BarrierContractError(
            "producer_id, generation_run_id, and semantic_case_group_id are required."
        )
    requirement = _validate_text(str(case.get("requirement", "")), "requirement", minimum_words=4)
    candidates = case.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 4:
        raise BarrierContractError("Each generated case must contain exactly four candidates.")
    normalized_candidates: list[dict[str, str]] = []
    seen_evidence: set[str] = set()
    for index, candidate in enumerate(candidates):
        evidence = _validate_text(
            str(candidate.get("evidence", "")),
            f"candidate[{index}]",
            minimum_words=4,
        )
        evidence_key = normalize_text(evidence)
        if evidence_key in seen_evidence:
            raise BarrierContractError("Candidate evidence must be unique within a case.")
        seen_evidence.add(evidence_key)
        candidate_id = str(candidate.get("candidate_id", "")).strip() or (
            "cand-" + hashlib.sha256(evidence_key.encode("utf-8")).hexdigest()[:12]
        )
        normalized_candidates.append({"candidate_id": candidate_id, "evidence": evidence})
    if len({candidate["candidate_id"] for candidate in normalized_candidates}) != 4:
        raise BarrierContractError("candidate_id values must be unique within a case.")
    intended_label = str(case.get("intended_label", ""))
    allowed = BARRIER_DEFINITIONS[barrier_id]["allowed_labels"]
    if intended_label not in allowed:
        raise BarrierContractError(
            f"{barrier_id} does not allow intended label {intended_label or 'missing'}."
        )
    intended_candidate_id = case.get("intended_candidate_id")
    candidate_ids = {candidate["candidate_id"] for candidate in normalized_candidates}
    if intended_label == "No Support" and intended_candidate_id is not None:
        raise BarrierContractError("No Support cases cannot have intended evidence.")
    if intended_label != "No Support" and intended_candidate_id not in candidate_ids:
        raise BarrierContractError("Supported cases require a valid intended candidate.")
    normalized = {
        "record_type": "generation_proposal",
        "schema_version": BARRIER_SCHEMA_VERSION,
        "case_id": stable_case_id(
            {"requirement": requirement, "candidates": normalized_candidates}
        ),
        "producer_id": producer_id,
        "generation_run_id": run_id,
        "semantic_case_group_id": semantic_group,
        "barrier_id": barrier_id,
        "role_family": role_family,
        "requirement": requirement,
        "candidates": normalized_candidates,
        "intended_label": intended_label,
        "intended_candidate_id": intended_candidate_id,
        "source_kind": "synthetic",
    }
    return normalized


def validate_review(review: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    """Validate one blind review without consulting the producer's intended decision."""
    reviewer_id = str(review.get("reviewer_id", "")).strip()
    if not reviewer_id:
        raise BarrierContractError("reviewer_id is required.")
    support_label = str(review.get("support_label", ""))
    if support_label not in SUPPORT_LABELS:
        raise BarrierContractError(f"Unsupported review label: {support_label or 'missing'}")
    selected_id = review.get("selected_candidate_id")
    candidate_ids = {
        str(candidate["candidate_id"])
        for candidate in case["candidates"]
    }
    raw_judgments = review.get("candidate_judgments")
    if not isinstance(raw_judgments, list) or len(raw_judgments) != len(candidate_ids):
        raise BarrierContractError("Review must label every candidate exactly once.")
    candidate_judgments: list[dict[str, str]] = []
    seen_judgments: set[str] = set()
    for judgment in raw_judgments:
        candidate_id = str(judgment.get("candidate_id", ""))
        candidate_label = str(judgment.get("support_label", ""))
        if candidate_id not in candidate_ids or candidate_id in seen_judgments:
            raise BarrierContractError("Candidate judgments must cover each candidate once.")
        if candidate_label not in SUPPORT_LABELS:
            raise BarrierContractError(f"Unsupported candidate label: {candidate_label or 'missing'}")
        seen_judgments.add(candidate_id)
        candidate_judgments.append(
            {"candidate_id": candidate_id, "support_label": candidate_label}
        )
    judgments_by_id = {
        judgment["candidate_id"]: judgment["support_label"]
        for judgment in candidate_judgments
    }
    label_rank = {"No Support": 0, "Partial": 1, "Direct": 2}
    highest_rank = max(label_rank[label] for label in judgments_by_id.values())
    if label_rank[support_label] != highest_rank:
        raise BarrierContractError("Overall label must equal the strongest candidate judgment.")
    if support_label == "No Support":
        selected_id = None
    elif selected_id not in candidate_ids:
        raise BarrierContractError("Direct and Partial reviews require valid evidence.")
    elif judgments_by_id[str(selected_id)] != support_label:
        raise BarrierContractError("Best evidence must have the overall support label.")
    flags = sorted({str(flag) for flag in review.get("flags", [])})
    unknown_flags = set(flags) - REVIEW_FLAGS
    if unknown_flags:
        raise BarrierContractError(f"Unknown review flags: {sorted(unknown_flags)}")
    return {
        "record_type": "review_decision",
        "schema_version": BARRIER_SCHEMA_VERSION,
        "case_id": case["case_id"],
        "reviewer_id": reviewer_id,
        "support_label": support_label,
        "selected_candidate_id": selected_id,
        "candidate_judgments": sorted(
            candidate_judgments,
            key=lambda judgment: judgment["candidate_id"],
        ),
        "flags": flags,
    }
