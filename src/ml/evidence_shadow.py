"""Offline comparison of product evidence rules and the frozen shadow model."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from ml.evidence import (
    build_semantic_evidence_index,
    extract_requirement_records,
    extract_resume_evidence_records,
)
from ml.evidence_multiclass import SUPPORT_CLASSES
from ml.evidence_sentence_artifact import predict_two_stage


SHADOW_CANDIDATE_LIMIT = 4


def comparison_status(
    *,
    baseline_accepted: bool,
    shadow_accepted: bool,
    baseline_evidence: str,
    shadow_evidence: str,
) -> str:
    """Classify one offline rule/model decision comparison."""
    if baseline_accepted and shadow_accepted:
        return (
            "both_accept_same_evidence"
            if baseline_evidence == shadow_evidence
            else "both_accept_different_evidence"
        )
    if shadow_accepted:
        return "shadow_only_accept"
    if baseline_accepted:
        return "baseline_only_accept"
    return "both_reject"


def build_shadow_report(
    job_text: str,
    resume_text: str,
    *,
    artifact: dict[str, Any],
    encoder: Any,
    max_requirements: int = 8,
) -> dict[str, Any]:
    """Compare outputs without changing application scoring or documents."""
    requirements = extract_requirement_records(job_text)[:max_requirements]
    evidence_records = extract_resume_evidence_records(resume_text)
    baseline = build_semantic_evidence_index(
        job_text,
        resume_text,
        max_requirements=max_requirements,
    )
    baseline_by_requirement = {
        str(match["requirement"]): match for match in baseline["matches"]
    }
    if not requirements or not evidence_records:
        return {
            "schema_version": 1,
            "mode": "local_shadow_only",
            "requirement_count": len(requirements),
            "evidence_count": len(evidence_records),
            "candidate_limit": SHADOW_CANDIDATE_LIMIT,
            "comparisons": [],
            "comparison_counts": {},
            "product_state_modified": False,
        }

    flattened_requirements = [
        str(requirement["text"])
        for requirement in requirements
        for _ in evidence_records
    ]
    flattened_evidence = [
        str(evidence["text"])
        for _ in requirements
        for evidence in evidence_records
    ]
    predictions = predict_two_stage(
        artifact,
        encoder,
        flattened_requirements,
        flattened_evidence,
    )
    groups: defaultdict[int, list[int]] = defaultdict(list)
    evidence_count = len(evidence_records)
    for requirement_index in range(len(requirements)):
        start = requirement_index * evidence_count
        groups[requirement_index].extend(range(start, start + evidence_count))

    comparisons: list[dict[str, Any]] = []
    for requirement_index, requirement in enumerate(requirements):
        indices = groups[requirement_index]
        ordered = sorted(
            indices,
            key=lambda index: (
                -float(predictions["hybrid_support_scores"][index]),
                index,
            ),
        )
        shortlisted = ordered[:SHADOW_CANDIDATE_LIMIT]
        top_index = shortlisted[0]
        top_evidence = evidence_records[top_index % evidence_count]
        accepted = any(
            predictions["predictions"][index] != "No Support"
            for index in shortlisted
        )
        baseline_match = baseline_by_requirement[str(requirement["text"])]
        baseline_evidence = str(baseline_match.get("evidence", ""))
        shadow_evidence = str(top_evidence["text"]) if accepted else ""
        pure_probabilities = predictions["pure_probabilities"][top_index]
        comparisons.append(
            {
                "requirement": str(requirement["text"]),
                "demand": str(requirement["demand"]),
                "baseline": {
                    "accepted": bool(baseline_match["accepted"]),
                    "evidence": baseline_evidence,
                    "similarity": float(baseline_match["similarity"]),
                    "match_type": str(baseline_match["match_type"]),
                },
                "shadow": {
                    "accepted": accepted,
                    "evidence": shadow_evidence,
                    "evidence_section": (
                        str(top_evidence["section"]) if accepted else ""
                    ),
                    "top_prediction": str(
                        predictions["predictions"][top_index]
                    ),
                    "pure_probabilities": {
                        label: float(probability)
                        for label, probability in zip(
                            SUPPORT_CLASSES,
                            pure_probabilities,
                            strict=True,
                        )
                    },
                    "hybrid_support_score": float(
                        predictions["hybrid_support_scores"][top_index]
                    ),
                    "candidate_pool_count": len(indices),
                    "shortlisted_candidate_count": len(shortlisted),
                },
                "comparison": comparison_status(
                    baseline_accepted=bool(baseline_match["accepted"]),
                    shadow_accepted=accepted,
                    baseline_evidence=baseline_evidence,
                    shadow_evidence=shadow_evidence,
                ),
            }
        )
    return {
        "schema_version": 1,
        "mode": "local_shadow_only",
        "requirement_count": len(requirements),
        "evidence_count": len(evidence_records),
        "candidate_limit": SHADOW_CANDIDATE_LIMIT,
        "comparisons": comparisons,
        "comparison_counts": dict(
            Counter(str(item["comparison"]) for item in comparisons)
        ),
        "product_state_modified": False,
        "decision_policy": (
            "Shadow results are diagnostic only and cannot affect Role Fit, "
            "Cover Letter evidence, or user-visible recommendations."
        ),
    }
