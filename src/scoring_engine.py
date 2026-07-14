"""Weighted scoring, confidence, and recommendation logic."""

from __future__ import annotations

import re

from scoring_config import (
    DIRECT_MATCH_STRENGTH,
    NO_MATCH_STRENGTH,
    PARTIAL_RESUME_MATCHES,
    PENALTY_RULES,
    PREFERRED_REQUIREMENT_WEIGHT,
    REQUIRED_REQUIREMENT_WEIGHT,
    SCORE_CATEGORIES,
)
from scoring_eligibility import evaluate_eligibility
from scoring_extraction import (
    find_keywords,
    infer_candidate_experience_profile,
    normalize_text,
    parse_job_description,
)
from scoring_matching import demand_type_for_keyword, match_strength_for_keyword


def score_note(
    category_name: str,
    matched: list[str],
    partial: list[str],
    missing: list[str],
) -> str:
    """Explain the category score in plain English."""
    if matched and not partial and not missing:
        return f"Strong fit: all requested {category_name.lower()} terms are supported by the resume."
    if matched or partial:
        return (
            f"Good fit: the resume supports several requested {category_name.lower()} terms, "
            "with any adjacent matches labeled as partial."
        )
    return f"Weak fit: the JD asks for {category_name.lower()} terms that are not clearly supported by the resume."


def calculate_score_breakdown(
    parsed_job: dict[str, object],
    resume_keywords: list[str],
    candidate_profile: dict[str, object] | None = None,
) -> list[dict[str, object]]:
    """Calculate demand-aware weighted category scores."""
    resume_keyword_set = set(resume_keywords)
    required_skills = set(parsed_job["required_skills"])
    preferred_skills = set(parsed_job["preferred_skills"])
    experience_level = set(parsed_job["experience_level"])
    breakdown = []

    for category_name, category in SCORE_CATEGORIES.items():
        keywords = category["keywords"]
        max_points = category["points"]
        matched = []
        partial = []
        missing = []
        active_terms = []
        earned_weight = 0.0
        possible_weight = 0.0

        for keyword in keywords:
            demand_type = demand_type_for_keyword(
                keyword=keyword,
                category_name=category_name,
                required_skills=required_skills,
                preferred_skills=preferred_skills,
                experience_level=experience_level,
            )
            if demand_type is None:
                continue

            active_terms.append(keyword)
            requirement_weight = (
                REQUIRED_REQUIREMENT_WEIGHT
                if demand_type == "required"
                else PREFERRED_REQUIREMENT_WEIGHT
            )
            possible_weight += requirement_weight

            match_strength = match_strength_for_keyword(keyword, resume_keyword_set, candidate_profile)
            earned_weight += requirement_weight * match_strength

            if match_strength == DIRECT_MATCH_STRENGTH:
                matched.append(keyword)
            elif match_strength > NO_MATCH_STRENGTH:
                detail = (
                    PARTIAL_RESUME_MATCHES[keyword][1]
                    if keyword in PARTIAL_RESUME_MATCHES
                    else "Partial match: candidate experience evidence is adjacent to the requested level."
                )
                partial.append(f"{keyword} ({detail})")
            else:
                missing.append(keyword)

        if possible_weight == 0:
            breakdown.append(
                {
                    "category": category_name,
                    "earned": None,
                    "possible": max_points,
                    "active_terms": [],
                    "matched": [],
                    "partial": [],
                    "missing": [],
                    "note": "N/A: job description does not ask for these terms.",
                }
            )
            continue

        earned_points = round((earned_weight / possible_weight) * max_points, 1)
        breakdown.append(
            {
                "category": category_name,
                "earned": earned_points,
                "possible": max_points,
                "active_terms": active_terms,
                "matched": matched,
                "partial": partial,
                "missing": missing,
                "note": score_note(category_name, matched, partial, missing),
            }
        )

    return breakdown


def find_penalties(job_text: str) -> list[dict[str, object]]:
    """Find configured score penalties."""
    normalized = normalize_text(job_text)
    penalties = []
    for rule in PENALTY_RULES:
        if any(re.search(pattern, normalized) for pattern in rule["patterns"]):
            penalties.append({"name": rule["name"], "points": rule["points"]})
    return penalties


def calculate_match_score(
    score_breakdown: list[dict[str, object]],
    penalties: list[dict[str, object]],
) -> int:
    """Calculate the normalized final score after active categories and penalties."""
    active_items = [item for item in score_breakdown if item["earned"] is not None]
    if not active_items:
        return 0

    earned_points = sum(float(item["earned"]) for item in active_items)
    possible_points = sum(float(item["possible"]) for item in active_items)
    raw_score = (earned_points / possible_points) * 100
    total_penalty = sum(int(item["points"]) for item in penalties)
    final_score = round(raw_score - total_penalty)
    return max(0, min(final_score, 100))


def apply_uk_work_authorization_score_cap(score: int, job_text: str) -> int:
    """Compatibility helper; eligibility no longer changes the role-fit score."""
    _ = job_text
    return score


def recommendation_for_score(score: int) -> str:
    """Convert a score into a human-readable application recommendation."""
    if score >= 80:
        return "Apply"
    if score >= 65:
        return "Apply / Maybe Apply"
    if score >= 50:
        return "Maybe Apply"
    return "Skip or Low Priority"


def calculate_scoring_confidence(
    job_text: str,
    candidate_text: str,
    score_breakdown: list[dict[str, object]],
    candidate_profile: dict[str, object],
) -> dict[str, object]:
    """Describe deterministic evidence coverage independently from role fit."""
    active_count = sum(len(item["active_terms"]) for item in score_breakdown)
    candidate_evidence_count = len(find_keywords(candidate_text)) + len(candidate_profile.get("evidence", []))
    reasons: list[str] = []
    job_word_count = len(job_text.split())
    if not candidate_text.strip():
        reasons.append("Candidate source is missing or empty.")
    if job_word_count < 20:
        reasons.append("Job text is too short for reliable scoring.")
    if active_count < 4:
        reasons.append("Fewer than four scored requirements were extracted.")
    if candidate_evidence_count == 0:
        reasons.append("No usable candidate evidence was extracted.")

    if reasons:
        level = "low"
    elif active_count >= 8:
        level = "high"
        reasons.append("At least eight scored requirements and candidate evidence were extracted.")
    else:
        level = "medium"
        reasons.append("Four to seven scored requirements and candidate evidence were extracted.")
    return {
        "level": level,
        "active_requirement_count": active_count,
        "candidate_evidence_count": candidate_evidence_count,
        "reasons": reasons,
    }


def final_recommendation(score: int, eligibility: dict[str, object], confidence: dict[str, object]) -> str:
    """Apply eligibility and confidence gates to the calibrated score label."""
    if eligibility["status"] == "failed":
        return "Skip / Not Eligible"
    if eligibility["status"] == "manual_review" or confidence["level"] == "low":
        return "Manual Review"
    return recommendation_for_score(score)


def explain_final_decision(
    score: int,
    recommendation: str,
    eligibility: dict[str, object],
    confidence: dict[str, object],
) -> str:
    """Explain the recommendation using role fit, eligibility, and confidence."""
    reasons = eligibility.get("reasons", [])
    if isinstance(reasons, list) and reasons:
        first = reasons[0]
        if isinstance(first, dict) and first.get("message"):
            return f"Role fit is {score}/100. {first['message']}"
    confidence_reasons = confidence.get("reasons", [])
    if confidence.get("level") == "low" and isinstance(confidence_reasons, list) and confidence_reasons:
        return f"Role fit is {score}/100, but manual review is needed because {str(confidence_reasons[0]).lower()}"
    return (
        f"{recommendation} because the role-fit score is {score}/100 after normalizing only "
        "the categories the job description actually mentions."
    )


def score_job_texts(job_text: str, candidate_text: str) -> dict[str, object]:
    """Run the deterministic scoring pipeline without file or network access."""
    job_keywords = find_keywords(job_text)
    resume_keywords = find_keywords(candidate_text)
    candidate_profile = infer_candidate_experience_profile(candidate_text)
    parsed_job = parse_job_description(job_text, candidate_text)
    score_breakdown = calculate_score_breakdown(parsed_job, resume_keywords, candidate_profile)
    penalties = find_penalties(job_text)
    score = calculate_match_score(score_breakdown, penalties)
    eligibility = evaluate_eligibility(job_text, candidate_text, candidate_profile)
    confidence = calculate_scoring_confidence(job_text, candidate_text, score_breakdown, candidate_profile)
    recommendation = final_recommendation(score, eligibility, confidence)
    return {
        "score": score,
        "recommendation": recommendation,
        "score_breakdown": score_breakdown,
        "eligibility": eligibility,
        "confidence": confidence,
        "candidate_profile": candidate_profile,
        "parsed_job": parsed_job,
        "job_keywords": job_keywords,
        "resume_keywords": resume_keywords,
        "penalties": penalties,
    }
