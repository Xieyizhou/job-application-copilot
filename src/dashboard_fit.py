"""Pure presentation models for dashboard fit and evidence displays."""

from __future__ import annotations

from typing import Any


def _append_unique(items: list[str], value: str) -> None:
    if value and value not in items:
        items.append(value)


def summarize_analysis_requirements(analysis: dict[str, Any]) -> dict[str, Any]:
    """Group recognized terms by requirement type and match strength."""
    parsed = dict(analysis.get("parsed_job", {}))
    required_terms = list(parsed.get("required_skills", [])) + list(parsed.get("experience_level", []))
    preferred_terms = list(parsed.get("preferred_skills", []))
    required_set = set(required_terms)
    preferred_set = set(preferred_terms)
    summary: dict[str, Any] = {
        "matched_required": [],
        "matched_preferred": [],
        "partial_required": [],
        "partial_preferred": [],
        "missing_required": [],
        "missing_preferred": [],
        "active_requirement_count": 0,
        "matched_requirement_count": 0,
    }
    active_order: list[str] = []
    matched_count_terms: list[str] = []
    for category in list(analysis.get("score_breakdown", [])):
        matched = set(category.get("matched", []))
        missing = set(category.get("missing", []))
        partial_map = {str(value).split(" (", 1)[0]: str(value) for value in category.get("partial", [])}
        for term in category.get("active_terms", []):
            term = str(term)
            _append_unique(active_order, term)
            requirement_type = "preferred" if term in preferred_set and term not in required_set else "required"
            if term in matched:
                _append_unique(summary[f"matched_{requirement_type}"], term)
                _append_unique(matched_count_terms, term)
            elif term in partial_map:
                _append_unique(summary[f"partial_{requirement_type}"], partial_map[term])
                _append_unique(matched_count_terms, term)
            elif term in missing:
                _append_unique(summary[f"missing_{requirement_type}"], term)
    summary["active_requirement_count"] = len(active_order)
    summary["matched_requirement_count"] = len(matched_count_terms)
    return summary


def confidence_level(value: Any) -> str:
    """Normalize confidence values stored as either mappings or strings."""
    if isinstance(value, dict):
        return str(value.get("level", "low")).lower()
    return str(value or "low").lower()


def eligibility_status(job: dict[str, Any]) -> str:
    """Normalize a job's eligibility status."""
    eligibility = job.get("eligibility", {})
    return str(eligibility.get("status", "manual_review") if isinstance(eligibility, dict) else eligibility).lower()


def build_fit_presentation(job: dict[str, Any]) -> dict[str, Any]:
    """Build one presentation model for cards, summaries, and Fit Analysis."""
    analysis = dict(job.get("analysis_result", {}) or {})
    available = bool(analysis.get("analysis_available", job.get("analysis_available", False)))
    confidence = dict(job.get("confidence", {}) or analysis.get("confidence", {}) or {})
    level = confidence_level(confidence)
    score = job.get("score")
    recommendation = str(job.get("recommendation", "Manual Review"))
    eligibility = dict(job.get("eligibility", {}) or analysis.get("eligibility", {}) or {})
    terms = summarize_analysis_requirements(analysis)
    if not available:
        legacy_score = job.get("legacy_score")
        legacy_recommendation = str(job.get("legacy_recommendation", "") or "")
        role_fit = "Not available"
        card_status = "Manual Review · Low confidence"
        if legacy_score is not None:
            card_status = f"Stored legacy score: {legacy_score}/100"
            if legacy_recommendation:
                card_status += f" · {legacy_recommendation}"
    elif level == "low" and score is not None:
        role_fit = f"Provisional {int(score)}/100"
        card_status = f"Provisional {int(score)}/100 · Manual Review · Low confidence"
    elif level == "low":
        role_fit = "Not available"
        card_status = "Manual Review · Low confidence"
    else:
        role_fit = f"{int(score)}/100"
        card_status = f"{int(score)}/100 · {recommendation} · {level.title()} confidence"
    eligibility_state = str(eligibility.get("status", "manual_review"))
    reasons = eligibility.get("reasons", [])
    if eligibility_state != "passed" and isinstance(reasons, list) and reasons:
        first_reason = reasons[0] if isinstance(reasons[0], dict) else {}
        reason_label = str(first_reason.get("code", "eligibility review")).replace("_", " ").title()
        card_status += f" · {reason_label}"
    return {
        "analysis_available": available,
        "role_fit": role_fit,
        "card_status": card_status,
        "score": score,
        "recommendation": recommendation,
        "eligibility": eligibility,
        "confidence": confidence,
        "terms": terms,
        "coverage_score": analysis.get("coverage_score", score) if available else None,
    }


def apply_canonical_analysis(job: dict[str, Any], analysis: dict[str, Any]) -> dict[str, Any]:
    """Copy current analyzer values onto an in-memory job record."""
    updated = dict(job)
    updated["analysis_result"] = analysis
    updated["analysis_available"] = bool(analysis.get("analysis_available", False))
    updated["score"] = analysis.get("score") if updated["analysis_available"] else None
    updated["recommendation"] = str(analysis.get("recommendation", "Manual Review"))
    updated["eligibility"] = dict(analysis.get("eligibility", {}))
    updated["confidence"] = dict(analysis.get("confidence", {}))
    updated["score_breakdown"] = list(analysis.get("score_breakdown", []))
    return updated
