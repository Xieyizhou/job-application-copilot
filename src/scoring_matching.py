"""Candidate-to-requirement matching helpers for deterministic job-fit analysis."""

from __future__ import annotations

from scoring_config import (
    CAREER_LEVELS,
    DIRECT_MATCH_STRENGTH,
    EXPERIENCE_LEVEL_KEYWORDS,
    EXPERIENCE_THEMES,
    KEYWORD_CATALOG,
    NO_MATCH_STRENGTH,
    PARTIAL_MATCH_STRENGTH,
    PARTIAL_RESUME_MATCHES,
)
from scoring_extraction import add_unique, contains_alias, normalize_text, split_job_description_lines
from scoring_types import CandidateProfile, ScoreBreakdownItem


def experience_match_strength(keyword: str, profile: CandidateProfile) -> float:
    """Compare a requested experience level with explicit candidate evidence."""
    level = str(profile.get("career_level", "unknown"))
    if level not in CAREER_LEVELS:
        level = "unknown"
    compatibility = {
        "student": {"intern": 1.0, "entry-level": 0.6},
        "new_grad": {
            "new grad": 1.0,
            "recent graduate": 1.0,
            "entry-level": 1.0,
            "intern": 1.0,
            "junior": 0.6,
        },
        "junior": {
            "junior": 1.0,
            "entry-level": 1.0,
            "new grad": 0.6,
            "recent graduate": 0.6,
            "mid-level": 0.6,
        },
        "mid": {"mid-level": 1.0, "junior": 0.6, "senior-level": 0.6},
        "senior": {"senior-level": 1.0, "mid-level": 1.0},
        "unknown": {},
    }
    return compatibility[level].get(keyword, NO_MATCH_STRENGTH)


def choose_relevant_themes(job_keywords: list[str], resume_keywords: list[str]) -> list[str]:
    """Choose resume-backed themes that overlap with the job description."""
    job_keyword_set = set(job_keywords)
    resume_keyword_set = set(resume_keywords)
    relevant_themes = []

    for theme, theme_keywords in EXPERIENCE_THEMES.items():
        has_job_overlap = bool(job_keyword_set.intersection(theme_keywords))
        has_resume_support = bool(resume_keyword_set.intersection(theme_keywords))
        if has_job_overlap and has_resume_support:
            relevant_themes.append(theme)

    return relevant_themes


def demand_type_for_keyword(
    keyword: str,
    category_name: str,
    required_skills: set[str],
    preferred_skills: set[str],
    experience_level: set[str],
) -> str | None:
    """Return required/preferred when a JD asks for a keyword."""
    if category_name == "Experience level fit" and keyword in experience_level:
        return "required"
    if keyword in preferred_skills:
        return "preferred"
    if keyword in required_skills:
        return "required"
    return None


def match_strength_for_keyword(
    keyword: str,
    resume_keyword_set: set[str],
    candidate_profile: CandidateProfile | None = None,
) -> float:
    """Return 1.0 for direct matches, 0.6 for adjacent matches, and 0.0 for gaps."""
    if keyword in EXPERIENCE_LEVEL_KEYWORDS:
        profile = candidate_profile or {
            "career_level": "unknown",
            "years_experience": None,
            "highest_degree": "unknown",
            "evidence": [],
        }
        return experience_match_strength(keyword, profile)
    if keyword in resume_keyword_set:
        return DIRECT_MATCH_STRENGTH
    if keyword in PARTIAL_RESUME_MATCHES and PARTIAL_RESUME_MATCHES[keyword][0].intersection(resume_keyword_set):
        return PARTIAL_MATCH_STRENGTH
    return NO_MATCH_STRENGTH


def collect_report_matches(
    score_breakdown: list[ScoreBreakdownItem],
) -> tuple[list[str], list[str], list[str]]:
    """Collect matched, partial, and missing terms from active scored categories."""
    matched: list[str] = []
    partial: list[str] = []
    missing: list[str] = []

    for item in score_breakdown:
        for keyword in item["matched"]:
            add_unique(matched, keyword)
        for keyword in item["partial"]:
            add_unique(partial, keyword)
        for keyword in item["missing"]:
            add_unique(missing, keyword)

    return matched, partial, missing


def short_evidence_snippets(text: str, keywords: list[str], limit: int = 3) -> list[str]:
    """Return short source snippets containing matched keywords."""
    snippets = []
    lines = split_job_description_lines(text)
    for keyword in keywords:
        keyword_aliases = KEYWORD_CATALOG.get(keyword, [keyword])
        for line in lines:
            normalized_line = normalize_text(line)
            if any(contains_alias(normalized_line, alias) for alias in keyword_aliases):
                snippets.append(line[:180])
                break
        if len(snippets) >= limit:
            break
    return snippets


def resume_suggestions_for_keywords(
    matched_keywords: list[str],
    missing_keywords: list[str],
    red_flags: list[str],
) -> list[str]:
    """Build concise, human-reviewable resume tailoring suggestions."""
    suggestions = []
    for keyword in matched_keywords[:3]:
        suggestions.append(f"Make relevant {keyword} evidence easy to find in the resume.")
    for keyword in missing_keywords[:2]:
        suggestions.append(f"Only add {keyword} if the resume source has truthful supporting evidence.")
    if red_flags:
        suggestions.append("Review eligibility, seniority, degree, and work-authorization requirements manually.")
    if not suggestions:
        suggestions.append("Review the job description and tailor bullets only where the resume source supports them.")
    return suggestions[:5]
