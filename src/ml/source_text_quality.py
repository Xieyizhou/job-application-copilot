"""Versioned source-text quality gates for successor dataset construction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
import hashlib
import json
import re

from ml.annotation_generation import CONTACT_PATTERN, normalize_text
from ml.evidence import clean_source_line, useful_tokens


LEGACY_SOURCE_TEXT_QUALITY = "legacy"
SUCCESSOR_V5_SOURCE_TEXT_QUALITY = "successor_v5_complete_text_v1"
SUCCESSOR_V6_SOURCE_TEXT_QUALITY = "successor_v6_canonical_text_v1"
SUCCESSOR_V7_SOURCE_TEXT_QUALITY = "successor_v7_sentence_boundary_v1"
SUCCESSOR_V8_SOURCE_TEXT_QUALITY = "successor_v8_sentence_boundary_v2"
SOURCE_TEXT_QUALITY_POLICIES = (
    LEGACY_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
    SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
)
MARKUP_FRAGMENT = re.compile(r"(?:\*\*|`|^\s*#+)")
DANGLING_PUNCTUATION = re.compile(r"[:,/]\s*$")
UNFINISHED_TAIL = re.compile(
    r"\b(?:and eliminated|as|at|built data|by|company called|for|from|"
    r"implementation of technical|including|in|model that|of|on|or|"
    r"the|to|under|using|which significantly|with)\W*$",
    re.IGNORECASE,
)
HEADING_FRAGMENT = re.compile(
    r"^(?:as an? .+? you will be responsible for|must-haves to succeed "
    r"in the role|you will be personally responsible for)\W*$",
    re.IGNORECASE,
)
GENERIC_TRUNCATED_REQUIREMENT = {
    "hands on experience with data",
}
TERMINAL_SENTENCE_BOUNDARY = re.compile(r"[.!?;][\"'”’\)\]]*$")


def validate_source_text_quality_policy(policy: str) -> None:
    """Reject unknown policies instead of silently changing source selection."""
    if policy not in SOURCE_TEXT_QUALITY_POLICIES:
        raise ValueError(f"Unknown source text quality policy: {policy}")


def canonical_public_text(text: str) -> str:
    """Apply the exact cleanup used in canonical reviewer-facing tasks."""
    cleaned = (
        text.replace("\x92", "'")
        .replace("\x95", "")
        .replace("\u200b", " ")
        .replace("\ufeff", "")
    )
    cleaned = clean_source_line(cleaned)
    cleaned = re.sub(
        r"^\(\s*full-time[^)]*\)\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return re.sub(r"^[^\w(]+", "", cleaned).strip()


def _policy_text(text: str, policy: str) -> str:
    validate_source_text_quality_policy(policy)
    if policy in {
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
        SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    }:
        return canonical_public_text(text)
    return text


def _balanced_delimiters(text: str) -> bool:
    return (
        text.count("(") == text.count(")")
        and text.count("[") == text.count("]")
        and text.count('"') % 2 == 0
        and text.count("“") == text.count("”")
    )


def requirement_quality_reasons(text: str) -> tuple[str, ...]:
    """Return deterministic reasons a requirement is unsafe to annotate."""
    reasons: list[str] = []
    words = text.split()
    normalized = normalize_text(text)
    if not 6 <= len(words) <= 48:
        reasons.append("requirement_word_count")
    if CONTACT_PATTERN.search(text):
        reasons.append("requirement_contact")
    if MARKUP_FRAGMENT.search(text):
        reasons.append("requirement_markup_fragment")
    if DANGLING_PUNCTUATION.search(text):
        reasons.append("requirement_dangling_punctuation")
    if UNFINISHED_TAIL.search(text):
        reasons.append("requirement_unfinished_tail")
    if HEADING_FRAGMENT.search(text):
        reasons.append("requirement_heading_fragment")
    if normalized in GENERIC_TRUNCATED_REQUIREMENT:
        reasons.append("requirement_generic_fragment")
    if not _balanced_delimiters(text):
        reasons.append("requirement_unbalanced_delimiter")
    return tuple(dict.fromkeys(reasons))


def evidence_quality_reasons(text: str) -> tuple[str, ...]:
    """Return deterministic reasons an evidence statement is unsafe to label."""
    reasons: list[str] = []
    if not 6 <= len(text.split()) <= 48:
        reasons.append("evidence_word_count")
    if CONTACT_PATTERN.search(text):
        reasons.append("evidence_contact")
    if MARKUP_FRAGMENT.search(text):
        reasons.append("evidence_markup_fragment")
    if DANGLING_PUNCTUATION.search(text):
        reasons.append("evidence_dangling_punctuation")
    if UNFINISHED_TAIL.search(text):
        reasons.append("evidence_unfinished_tail")
    if not _balanced_delimiters(text):
        reasons.append("evidence_unbalanced_delimiter")
    return tuple(dict.fromkeys(reasons))


def requirement_quality_reasons_for_policy(
    text: str,
    policy: str,
) -> tuple[str, ...]:
    """Validate source requirements under one frozen policy contract."""
    candidate = _policy_text(text, policy)
    reasons = list(requirement_quality_reasons(candidate))
    if (
        policy
        in {
            SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        }
        and not TERMINAL_SENTENCE_BOUNDARY.search(candidate)
    ):
        reasons.append("requirement_missing_sentence_boundary")
    return tuple(dict.fromkeys(reasons))


def evidence_quality_reasons_for_policy(
    text: str,
    policy: str,
) -> tuple[str, ...]:
    """Validate source evidence under one frozen policy contract."""
    candidate = _policy_text(text, policy)
    reasons = list(evidence_quality_reasons(candidate))
    if (
        policy
        in {
            SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
            SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
        }
        and not TERMINAL_SENTENCE_BOUNDARY.search(candidate)
    ):
        reasons.append("evidence_missing_sentence_boundary")
    return tuple(dict.fromkeys(reasons))


def texts_are_near_duplicates(first: str, second: str) -> bool:
    """Detect evidence paraphrases that should not share one source pool."""
    left = useful_tokens(first)
    right = useful_tokens(second)
    union = left | right
    shorter = min(len(left), len(right))
    if not union or not shorter:
        return normalize_text(first) == normalize_text(second)
    intersection = len(left & right)
    return (
        normalize_text(first) == normalize_text(second)
        or intersection / len(union) >= 0.72
        or intersection / shorter >= 0.80
    )


def diverse_quality_evidence(
    evidence: Sequence[str],
    *,
    policy: str = SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
) -> list[str]:
    """Keep complete evidence statements while removing near paraphrases."""
    selected: list[str] = []
    for text in evidence:
        candidate = _policy_text(text, policy)
        if evidence_quality_reasons_for_policy(candidate, policy):
            continue
        if any(
            texts_are_near_duplicates(candidate, prior) for prior in selected
        ):
            continue
        selected.append(candidate)
    return selected


def source_text_quality_manifest(policy: str) -> dict[str, object]:
    """Return a stable, content-free commitment to the selected policy."""
    validate_source_text_quality_policy(policy)
    rules = {
        "policy": policy,
        "requirement_word_range": [6, 48],
        "evidence_word_range": [6, 48],
        "reject_contact": True,
        "reject_markup_fragments": True,
        "reject_dangling_colon_comma_or_slash": True,
        "reject_unfinished_tails": True,
        "reject_unbalanced_delimiters": True,
        "deduplicate_evidence_jaccard": 0.72,
        "deduplicate_evidence_shorter_containment": 0.80,
    }
    if policy in {
        SUCCESSOR_V6_SOURCE_TEXT_QUALITY,
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
        SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    }:
        rules.update(
            {
                "canonicalize_before_validation": True,
                "canonicalization_contract": "canonical_public_text_v1",
            }
        )
    if policy in {
        SUCCESSOR_V7_SOURCE_TEXT_QUALITY,
        SUCCESSOR_V8_SOURCE_TEXT_QUALITY,
    }:
        rules.update(
            {
                "require_terminal_sentence_boundary": True,
                "accepted_terminal_boundaries": [".", "!", "?", ";"],
            }
        )
    encoded = json.dumps(
        rules,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **rules,
        "policy_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def audit_materialized_text_quality(
    tasks: Sequence[Mapping[str, object]],
    *,
    policy: str = SUCCESSOR_V5_SOURCE_TEXT_QUALITY,
) -> dict[str, object]:
    """Audit canonical development text without emitting source content."""
    validate_source_text_quality_policy(policy)
    findings: dict[str, set[str]] = {}
    requirements: Counter[str] = Counter()
    evidence_by_group: dict[str, dict[str, str]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        requirement = str(task["requirement"])
        requirements[normalize_text(requirement)] += 1
        for reason in requirement_quality_reasons_for_policy(
            requirement,
            policy,
        ):
            findings.setdefault(task_id, set()).add(reason)
        group = str(task["source_resume_hash"])
        group_evidence = evidence_by_group.setdefault(group, {})
        candidates = task.get("candidates")
        if not isinstance(candidates, list):
            findings.setdefault(task_id, set()).add(
                "invalid_candidate_container"
            )
            continue
        for candidate in candidates:
            if not isinstance(candidate, Mapping):
                findings.setdefault(task_id, set()).add(
                    "invalid_candidate_record"
                )
                continue
            candidate_id = str(candidate.get("candidate_id", ""))
            evidence = str(candidate.get("evidence", ""))
            for reason in evidence_quality_reasons_for_policy(
                evidence,
                policy,
            ):
                findings.setdefault(task_id, set()).add(reason)
            prior = group_evidence.get(candidate_id)
            if prior is not None and prior != evidence:
                findings.setdefault(task_id, set()).add(
                    "candidate_identity_drift"
                )
            group_evidence[candidate_id] = evidence
    duplicate_requirements = sum(
        count - 1 for count in requirements.values() if count > 1
    )
    if duplicate_requirements:
        for task in tasks:
            if requirements[normalize_text(str(task["requirement"]))] > 1:
                findings.setdefault(str(task["task_id"]), set()).add(
                    "duplicate_requirement"
                )
    near_duplicate_groups = 0
    for group_evidence in evidence_by_group.values():
        values = list(group_evidence.values())
        if any(
            texts_are_near_duplicates(first, second)
            for index, first in enumerate(values)
            for second in values[index + 1 :]
        ):
            near_duplicate_groups += 1
    reason_counts = Counter(
        reason for reasons in findings.values() for reason in reasons
    )
    return {
        "schema_version": 1,
        "tasks_audited": len(tasks),
        "resume_groups_audited": len(evidence_by_group),
        "blocked_tasks": len(findings),
        "blocked_task_ids": sorted(findings),
        "finding_counts": dict(sorted(reason_counts.items())),
        "exact_duplicate_requirements": duplicate_requirements,
        "near_duplicate_resume_groups": near_duplicate_groups,
        "automated_quality_gate_passed": (
            not findings and near_duplicate_groups == 0
        ),
        "content_included": False,
    }
