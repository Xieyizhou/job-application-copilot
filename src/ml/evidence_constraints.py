"""Deterministic constraint features for requirement/evidence support decisions."""

from __future__ import annotations

import re
from typing import TypedDict

from ml.evidence_text import ACTION_PATTERN
from ml.evidence_text import concept_tags, stated_years, useful_tokens
from ml.evidence_constraint_taxonomy import (
    ACRONYM_PATTERN,
    CLAUSE_SPLIT_PATTERN,
    DEGREE_LEVELS,
    ELIGIBILITY_PATTERNS,
    NEGATED_DELIVERY_PATTERN,
    TECHNOLOGY_TERMS,
    TERM_FAMILY_ALIASES,
    WEAK_ASSERTION_PATTERN,
)


class ConstraintFeatures(TypedDict):
    """Stable numeric and Boolean features for one aligned text pair."""

    named_term_count: float
    named_term_coverage: float
    missing_named_term: float
    named_family_coverage: float
    named_constraint_supported: float
    concept_coverage: float
    concept_constraint_supported: float
    clause_coverage: float
    numeric_required: float
    numeric_constraint_supported: float
    degree_required: float
    degree_constraint_supported: float
    eligibility_required: float
    eligibility_constraint_supported: float
    action_assertion: float
    weak_assertion: float
    explicit_negation: float
    evidence_specificity: float


CONSTRAINT_FEATURE_NAMES = tuple(ConstraintFeatures.__annotations__)


def named_terms(text: str) -> set[str]:
    """Return explicit tools, platforms, and acronyms without retaining text."""
    lowered = re.sub(r"\s+", " ", text.lower())
    technologies = {
        term
        for term in TECHNOLOGY_TERMS
        if re.search(rf"(?<![\w+#-]){re.escape(term)}(?![\w+#-])", lowered)
    }
    acronyms = {
        match.group(0).lower()
        for match in ACRONYM_PATTERN.finditer(text)
        if match.group(0).lower() not in {"and", "the", "with"}
    }
    return technologies | acronyms


def _degree_level(text: str) -> int:
    lowered = text.lower()
    return max(
        (
            level
            for phrase, level in DEGREE_LEVELS.items()
            if re.search(rf"\b{re.escape(phrase)}\b", lowered)
        ),
        default=0,
    )


def _eligibility_tags(text: str) -> set[str]:
    return {
        name
        for name, pattern in ELIGIBILITY_PATTERNS.items()
        if pattern.search(text)
    }


def term_families(text: str) -> set[str]:
    """Map named tools and technical phrases into adjacent skill families."""
    normalized = " " + re.sub(r"[^a-z0-9+#./-]+", " ", text.lower()).strip() + " "
    return {
        family
        for family, aliases in TERM_FAMILY_ALIASES.items()
        if any(alias in normalized for alias in aliases)
    }


def _clause_coverage(requirement: str, evidence: str) -> float:
    clauses = [
        clause.strip()
        for clause in CLAUSE_SPLIT_PATTERN.split(requirement)
        if len(useful_tokens(clause)) >= 2
    ]
    if len(clauses) <= 1:
        return 1.0
    evidence_tokens = useful_tokens(evidence)
    evidence_concepts = concept_tags(evidence)
    evidence_terms = named_terms(evidence)
    covered = 0
    for clause in clauses:
        clause_tokens = useful_tokens(clause)
        clause_concepts = concept_tags(clause)
        clause_terms = named_terms(clause)
        token_coverage = (
            len(clause_tokens & evidence_tokens) / len(clause_tokens)
            if clause_tokens
            else 0.0
        )
        if (
            token_coverage >= 0.35
            or bool(clause_concepts & evidence_concepts)
            or bool(clause_terms & evidence_terms)
        ):
            covered += 1
    return covered / len(clauses)


def constraint_features(requirement: str, evidence: str) -> ConstraintFeatures:
    """Extract auditable constraint and assertion signals for a pair."""
    requirement_terms = named_terms(requirement)
    evidence_terms = named_terms(evidence)
    shared_terms = requirement_terms & evidence_terms
    required_concepts = concept_tags(requirement)
    evidence_concepts = concept_tags(evidence)
    requirement_families = term_families(requirement)
    evidence_families = term_families(evidence)
    shared_families = requirement_families & evidence_families
    required_years = stated_years(requirement)
    evidence_years = stated_years(evidence)
    required_degree = _degree_level(requirement)
    evidence_degree = _degree_level(evidence)
    required_eligibility = _eligibility_tags(requirement)
    evidence_eligibility = _eligibility_tags(evidence)
    evidence_tokens = useful_tokens(evidence)
    return {
        "named_term_count": float(len(requirement_terms)),
        "named_term_coverage": (
            len(shared_terms) / len(requirement_terms)
            if requirement_terms
            else 1.0
        ),
        "missing_named_term": float(bool(requirement_terms and not shared_terms)),
        "named_family_coverage": (
            len(shared_families) / len(requirement_families)
            if requirement_families
            else 1.0
        ),
        "named_constraint_supported": float(
            not requirement_terms
            or bool(shared_terms)
            or bool(shared_families)
            or bool(required_concepts & evidence_concepts)
        ),
        "concept_coverage": (
            len(required_concepts & evidence_concepts) / len(required_concepts)
            if required_concepts
            else 1.0
        ),
        "concept_constraint_supported": float(
            not required_concepts
            or bool(required_concepts & evidence_concepts)
            or bool(shared_families)
            or bool(shared_terms)
        ),
        "clause_coverage": _clause_coverage(requirement, evidence),
        "numeric_required": float(bool(required_years)),
        "numeric_constraint_supported": float(
            not required_years
            or bool(evidence_years)
            and max(evidence_years) >= max(required_years)
        ),
        "degree_required": float(bool(required_degree)),
        "degree_constraint_supported": float(
            not required_degree or evidence_degree >= required_degree
        ),
        "eligibility_required": float(bool(required_eligibility)),
        "eligibility_constraint_supported": float(
            not required_eligibility
            or required_eligibility <= evidence_eligibility
        ),
        "action_assertion": float(bool(ACTION_PATTERN.search(evidence))),
        "weak_assertion": float(bool(WEAK_ASSERTION_PATTERN.search(evidence))),
        "explicit_negation": float(bool(NEGATED_DELIVERY_PATTERN.search(evidence))),
        "evidence_specificity": min(
            1.0,
            (
                len(evidence_tokens)
                + 2 * len(evidence_terms)
                + 2 * len(evidence_concepts)
            )
            / 18.0,
        ),
    }


def constraint_feature_vector(requirement: str, evidence: str) -> list[float]:
    """Return constraint features in ``CONSTRAINT_FEATURE_NAMES`` order."""
    features = constraint_features(requirement, evidence)
    return [
        features["named_term_count"],
        features["named_term_coverage"],
        features["missing_named_term"],
        features["named_family_coverage"],
        features["named_constraint_supported"],
        features["concept_coverage"],
        features["concept_constraint_supported"],
        features["clause_coverage"],
        features["numeric_required"],
        features["numeric_constraint_supported"],
        features["degree_required"],
        features["degree_constraint_supported"],
        features["eligibility_required"],
        features["eligibility_constraint_supported"],
        features["action_assertion"],
        features["weak_assertion"],
        features["explicit_negation"],
        features["evidence_specificity"],
    ]


def hard_constraints_supported(features: ConstraintFeatures) -> bool:
    """Reject only explicit numeric, credential, eligibility, or negation failures."""
    return (
        bool(features["eligibility_constraint_supported"])
        and bool(features["named_constraint_supported"])
        and bool(features["concept_constraint_supported"])
        and not bool(features["explicit_negation"])
    )
