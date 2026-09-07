"""Build privacy-screened, source-isolated real-text development tasks."""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections.abc import Sequence
from typing import Any

from document_text import normalize_comparison_text as normalize_text
from ml.evidence_text import useful_tokens
from ml.source_text_quality import canonical_public_text


ROLE_FAMILIES = ("Data", "ML", "Software", "Business")
ROLE_TERMS = {
    "Data": (
        "data analyst",
        "analytics",
        "business intelligence",
        "power bi",
        "tableau",
        "sql",
        "reporting",
        "database",
    ),
    "ML": (
        "machine learning",
        "data scientist",
        "artificial intelligence",
        "computer vision",
        "deep learning",
        "nlp",
        "pytorch",
        "tensorflow",
    ),
    "Software": (
        "software",
        "developer",
        "engineer",
        "frontend",
        "backend",
        "full stack",
        "python",
        "java",
        "javascript",
        "devops",
        "qa",
    ),
    "Business": (
        "sales",
        "marketing",
        "customer",
        "product manager",
        "project manager",
        "business analyst",
        "recruit",
        "finance",
        "operations",
    ),
}
ACTION_PREFIX = re.compile(
    r"^(?:i\s+|we\s+)?(?:analy[sz]ed|automated|built|collaborated|configured|"
    r"conducted|coordinated|created|deployed|designed|developed|delivered|"
    r"documented|implemented|improved|increased|integrated|led|maintained|"
    r"managed|migrated|monitored|optimized|prepared|presented|produced|"
    r"reduced|researched|supported|tested|trained|used|worked)\b",
    re.IGNORECASE,
)
LEGACY_EVIDENCE_EXTRACTION = "action_prefix_v1"
CONTEXTUAL_EVIDENCE_EXTRACTION = "contextual_action_v2"
EVIDENCE_EXTRACTION_POLICIES = (
    LEGACY_EVIDENCE_EXTRACTION,
    CONTEXTUAL_EVIDENCE_EXTRACTION,
)
CONTEXTUAL_ACTION_PREFIX = re.compile(
    r"^(?:(?:responsible for|experience (?:includes?|involves?)|"
    r"selected projects?|projects?)\s*[:\-]?\s*)"
    r"(?:analy[sz]ing|automating|building|collaborating|configuring|"
    r"conducting|coordinating|creating|deploying|designing|developing|"
    r"delivering|documenting|implementing|improving|increasing|"
    r"integrating|leading|maintaining|managing|migrating|monitoring|"
    r"optimizing|preparing|presenting|producing|reducing|researching|"
    r"supporting|testing|training|using|working)\b",
    re.IGNORECASE,
)
LABELED_ACTION = re.compile(
    r"^[^:\n]{2,80}:\s*(?:i\s+|we\s+)?(?:analy[sz]ed|automated|built|"
    r"collaborated|configured|conducted|coordinated|created|deployed|"
    r"designed|developed|delivered|documented|implemented|improved|"
    r"increased|integrated|led|maintained|managed|migrated|monitored|"
    r"optimized|prepared|presented|produced|reduced|researched|supported|"
    r"tested|trained|used|worked)\b",
    re.IGNORECASE,
)
ROLE_LED_ACTION = re.compile(
    r"^as an? [^,:]{2,80}[,:]\s*(?:i\s+)?(?:analy[sz]ed|automated|built|"
    r"collaborated|configured|conducted|coordinated|created|deployed|"
    r"designed|developed|delivered|documented|implemented|improved|"
    r"increased|integrated|led|maintained|managed|migrated|monitored|"
    r"optimized|prepared|presented|produced|reduced|researched|supported|"
    r"tested|trained|used|worked)\b",
    re.IGNORECASE,
)
CAPABILITY_EVIDENCE_PREFIX = re.compile(
    r"^(?:(?:professional|relevant|hands-on)\s+)?experience\s+(?:in|with)\b|"
    r"^(?:expertise|proficiency|knowledge)\s+in\b|"
    r"^(?:proficient|skilled|specialized)\s+in\b",
    re.IGNORECASE,
)
PRIVATE_OR_METADATA = re.compile(
    r"(?:https?://|www\.|[\w.+-]+@[\w.-]+\.[a-z]{2,}|\b(?:salary|citizen|"
    r"nationality|date of birth|marital status|phone|telegram|skype|linkedin|"
    r"github|political campaign|personal brand|ministry|university)\b|"
    r"<address>|<location>|<organization>)",
    re.IGNORECASE,
)
LOCATION_CONTEXT = re.compile(
    r"\b(?:ukraine|kyiv|germany|nigeria|abuja|azerbaijan|czech|prague|"
    r"bali|russia|belarus|poland|india|canada|australia|united states|"
    r"united kingdom|european union|azerbaijani|mckinsey)\b",
    re.IGNORECASE,
)
DATE_OR_HEADING = re.compile(
    r"^(?:(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|"
    r"july?|aug(?:ust)?|sept?(?:ember)?|oct(?:ober)?|nov(?:ember)?|"
    r"dec(?:ember)?)\b|\d{4}\s*[-–]\s*(?:\d{4}|present)|[A-Z][\w& .'-]+,\s*"
    r"(?:[A-Z][\w .'-]+))",
    re.IGNORECASE,
)
INCOMPLETE_ENDING = re.compile(
    r"\b(?:a|an|and|as|at|by|for|from|in|of|on|or|over|the|to|under|"
    r"using|which|with|their|this|these|those|my|our|your|supervised|"
    r"cloud|manage|including)\W*$",
    re.IGNORECASE,
)


class DevelopmentDataError(ValueError):
    """Raised when a development task violates a construction boundary."""


def source_hash(namespace: str, source_id: str) -> str:
    """Return the stable source identity used by the existing real-text sets."""
    return hashlib.sha256(f"{namespace}:{source_id}".encode()).hexdigest()


def infer_role_family(*texts: str) -> str | None:
    """Infer one broad role family from explicit title and keyword terms."""
    normalized = " ".join(texts).lower()
    scores = {
        family: sum(
            1 + int(" " in term)
            for term in terms
            if term in normalized
        )
        for family, terms in ROLE_TERMS.items()
    }
    best_score = max(scores.values(), default=0)
    if not best_score:
        return None
    winners = [
        family for family in ROLE_FAMILIES if scores[family] == best_score
    ]
    return winners[0] if len(winners) == 1 else None


def validate_evidence_extraction_policy(policy: str) -> None:
    """Reject unknown extraction policies instead of changing source pools."""
    if policy not in EVIDENCE_EXTRACTION_POLICIES:
        raise ValueError(f"Unknown evidence extraction policy: {policy}")


def evidence_extraction_manifest(policy: str) -> dict[str, object]:
    """Commit to one extraction policy without retaining source text."""
    validate_evidence_extraction_policy(policy)
    rules = {
        "policy": policy,
        "requires_complete_statement": True,
        "rejects_private_or_metadata": True,
        "rejects_location_context": True,
        "rejects_incomplete_endings": True,
        "deduplicates_normalized_text": True,
        "accepted_prefix_groups": (
            ["action"]
            if policy == LEGACY_EVIDENCE_EXTRACTION
            else ["action", "contextual_action", "labeled_action", "capability"]
        ),
    }
    encoded = json.dumps(
        rules,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        **rules,
        "policy_sha256": hashlib.sha256(encoded).hexdigest(),
    }


def _has_supported_evidence_prefix(text: str, policy: str) -> bool:
    if ACTION_PREFIX.search(text):
        return True
    if policy == LEGACY_EVIDENCE_EXTRACTION:
        return False
    return bool(
        CONTEXTUAL_ACTION_PREFIX.search(text)
        or LABELED_ACTION.search(text)
        or ROLE_LED_ACTION.search(text)
        or CAPABILITY_EVIDENCE_PREFIX.search(text)
    )


def extract_action_evidence(
    resume_text: str,
    *,
    policy: str = LEGACY_EVIDENCE_EXTRACTION,
) -> list[str]:
    """Extract factual action statements while rejecting likely personal data."""
    validate_evidence_extraction_policy(policy)
    candidates: list[str] = []
    chunks = re.split(r"[\r\n]+|(?<=[.!?])\s+(?=[A-Z])", resume_text)
    for raw in chunks:
        text = clean_public_text(raw)
        words = text.split()
        supported_prefix = _has_supported_evidence_prefix(text, policy)
        if not 6 <= len(words) <= 48:
            continue
        if (
            PRIVATE_OR_METADATA.search(text)
            or LOCATION_CONTEXT.search(text)
            or INCOMPLETE_ENDING.search(text)
        ):
            continue
        if DATE_OR_HEADING.search(text) and not (
            policy == CONTEXTUAL_EVIDENCE_EXTRACTION
            and supported_prefix
        ):
            continue
        if not supported_prefix:
            continue
        if text.count('"') % 2 or text.count("“") != text.count("”"):
            continue
        if normalize_text(text) not in {
            normalize_text(candidate) for candidate in candidates
        }:
            candidates.append(text)
    return candidates


def evidence_overlap(requirement: str, evidence: str) -> float:
    """Return transparent requirement-token coverage for construction strata."""
    requirement_tokens = useful_tokens(requirement)
    evidence_tokens = useful_tokens(evidence)
    if not requirement_tokens:
        return 0.0
    return len(requirement_tokens & evidence_tokens) / len(requirement_tokens)


def select_evidence_candidates(
    requirement: str,
    evidence: Sequence[str],
    *,
    seed: str,
    count: int = 4,
) -> list[dict[str, str]]:
    """Select one lexical anchor plus diverse factual alternatives."""
    if len(evidence) < count:
        raise DevelopmentDataError("Resume has too few privacy-screened statements.")
    ranked = sorted(
        evidence,
        key=lambda item: (
            evidence_overlap(requirement, item),
            len(useful_tokens(item)),
            normalize_text(item),
        ),
        reverse=True,
    )
    selected = [ranked[0]]
    remaining = ranked[1:]
    randomizer = random.Random(seed)
    randomizer.shuffle(remaining)
    for candidate in remaining:
        if any(_near_duplicate(candidate, existing) for existing in selected):
            continue
        selected.append(candidate)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise DevelopmentDataError("Resume statements are not diverse enough.")
    randomizer.shuffle(selected)
    return [
        {
            "candidate_id": "ev-"
            + hashlib.sha256(text.encode()).hexdigest()[:16],
            "evidence": text,
        }
        for text in selected
    ]


def build_development_task(
    *,
    requirement: str,
    evidence: Sequence[str],
    role_family: str,
    source_job_hash: str,
    source_resume_hash: str,
    source_dataset: str,
    seed: str,
) -> dict[str, Any]:
    """Build one unlabeled task with deterministic identity and candidate order."""
    if role_family not in ROLE_FAMILIES:
        raise DevelopmentDataError(f"Unknown role family: {role_family}")
    normalized_requirement = normalize_text(requirement)
    if not normalized_requirement:
        raise DevelopmentDataError("Requirement is empty.")
    task_material = "\0".join(
        [source_job_hash, source_resume_hash, normalized_requirement]
    )
    return {
        "schema_version": 1,
        "task_id": "real-dev-"
        + hashlib.sha256(task_material.encode()).hexdigest()[:16],
        "role_family": role_family,
        "requirement": clean_public_text(requirement),
        "candidates": select_evidence_candidates(
            requirement,
            evidence,
            seed=seed,
        ),
        "source_resume_hash": source_resume_hash,
        "source_job_hash": source_job_hash,
        "source_dataset": source_dataset,
        "blind_duplicate_of": None,
    }


def _near_duplicate(first: str, second: str) -> bool:
    first_tokens = useful_tokens(first)
    second_tokens = useful_tokens(second)
    union = first_tokens | second_tokens
    intersection = first_tokens & second_tokens
    shorter = min(len(first_tokens), len(second_tokens))
    return bool(union) and (
        len(intersection) / len(union) >= 0.72
        or bool(shorter) and len(intersection) / shorter >= 0.80
    )


def clean_public_text(text: str) -> str:
    """Normalize broken display characters without rewriting source facts."""
    return canonical_public_text(text)
