"""Conservative production label checks; similarity is not factual support."""

from __future__ import annotations

import re


DEGREES = (
    (3, r"\b(?:ph\.?d\.?|doctorate|doctoral)\b"),
    (2, r"\b(?:master(?:['’]s)?|msc|m\.sc\.?|m\.s\.)\b"),
    (1, r"\b(?:bachelor(?:['’]s)?|undergraduate|bsc|b\.sc\.?)\b"),
)
TASK = re.compile(
    r"\b(?:build|develop|design|implement|deploy|optimize|optimise|evaluate|manage|lead|"
    r"maintain|conduct|create|deliver|analyze|analyse|monitor|troubleshoot|engineer)\b", re.I,
)
WEAK = re.compile(r"\b(?:coursework|course|studied|learning|familiarity|exposure|interested|plan to|hope to)\b", re.I)
NEGATION = re.compile(r"\b(?:no experience|never|not experienced|have not|haven't|did not|didn't|without experience)\b", re.I)
ACTION_FAMILIES = (
    (r"\b(?:build|develop|design|implement|create)\b", r"\b(?:built|developed|designed|implemented|created|building|developing)\b"),
    (r"\b(?:deploy|maintain|monitor|troubleshoot)\b", r"\b(?:deployed|maintained|monitored|troubleshot|operated|deployment|production)\b"),
    (r"\b(?:evaluate|optimize|optimise)\b", r"\b(?:evaluated|optimized|optimised|validated|benchmarked|evaluation|cross-validation)\b"),
    (r"\b(?:lead|manage)\b", r"\b(?:led|managed|supervised|directed)\b"),
)


def label_checks(requirement: str, evidence: str) -> tuple[bool, list[str]]:
    """Return hard rejection and reasons to prohibit Direct for this exact pair."""
    # Local import keeps the research feature module independent at import time.
    from ml.evidence import ACTION_PATTERN
    from ml.evidence_constraints import constraint_features

    reasons: list[str] = []
    required = [(level, match) for level, pattern in DEGREES for match in re.finditer(pattern, requirement, re.I)]
    levels = [level for level, _ in required]
    evidence_level = max((level for level, pattern in DEGREES if re.search(pattern, evidence, re.I)), default=0)
    if levels:
        ordered = sorted(required, key=lambda item: item[1].start())
        alternatives = len(ordered) > 1 and all(
            re.search(r"\bor\b|/", requirement[left[1].end():right[1].start()], re.I)
            for left, right in zip(ordered, ordered[1:])
        )
        minimum = min(levels) if alternatives else max(levels)
        if evidence_level < minimum:
            return True, ["The cited evidence does not establish the required degree level."]
        if re.search(r"\b(?:pursuing|studying|candidate for|expected)\b", evidence, re.I) and not re.search(r"\b(?:pursuing|student|undergraduate|enrolled)\b", requirement, re.I):
            return True, ["An in-progress degree does not establish a completed credential."]
    if NEGATION.search(evidence):
        return True, ["The cited evidence explicitly denies experience."]
    features = constraint_features(requirement, evidence)
    if not features["eligibility_constraint_supported"]:
        return True, ["The cited evidence does not establish the eligibility condition."]
    if features["named_term_coverage"] < 1:
        reasons.append("Not all explicitly named tools or technologies are evidenced.")
    if features["clause_coverage"] < 1 or features["concept_coverage"] < 0.75:
        reasons.append("Only part of the compound requirement is evidenced.")
    if TASK.search(requirement) and not ACTION_PATTERN.search(evidence):
        reasons.append("Knowledge or a credential alone does not demonstrate performing this work.")
    elif any(re.search(demand, requirement, re.I) and not re.search(support, evidence, re.I) for demand, support in ACTION_FAMILIES):
        reasons.append("The cited activity does not establish all requested responsibilities.")
    # 'machine learning' is not a weak assertion; coursework/aspirations are.
    weak_text = re.sub(r"\b(?:machine|deep|reinforcement) learning\b", "", evidence, flags=re.I)
    if WEAK.search(weak_text) and not ACTION_PATTERN.search(evidence):
        reasons.append("The evidence states exposure or study, not demonstrated application.")
    if re.search(r"\b(?:professional|commercial|industry) experience\b", requirement, re.I) and re.search(r"\b(?:coursework|academic|class project|student project)\b", evidence, re.I):
        return True, ["Academic experience does not establish the requested professional experience."]
    return False, reasons
