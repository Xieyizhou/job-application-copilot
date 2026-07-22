"""Text normalization and explicit requirement extraction for job-fit analysis."""

from __future__ import annotations

import re

from scoring_config import (
    CAREER_LEVELS,
    EXPERIENCE_LEVEL_KEYWORDS,
    KEYWORD_CATALOG,
    PREFERRED_LANGUAGE,
    RED_FLAG_RULES,
    SCORE_CATEGORIES,
    UK_ALREADY_AUTHORIZED_WARNING,
    UK_HPI_MANUAL_REVIEW_WARNING,
    UK_HPI_NOTE,
)
from scoring_types import CandidateProfile, ParsedJob


def normalize_text(text: str) -> str:
    """Lowercase text and pad it with spaces to make word matching simpler."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#\-]+", " ", text)
    return f" {text} "


def contains_alias(normalized_text: str, alias: str) -> bool:
    """Return True when an alias appears as a simple phrase or token."""
    normalized_alias = normalize_text(alias).strip()
    if not normalized_alias:
        return False
    if len(normalized_alias) <= 2 and normalized_alias.isalnum():
        return re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])",
            normalized_text,
        ) is not None
    return f" {normalized_alias} " in normalized_text


def find_keywords(text: str) -> list[str]:
    """Find catalog keywords that appear in the provided text."""
    normalized = normalize_text(text)
    found_keywords = []
    for keyword, aliases in KEYWORD_CATALOG.items():
        if any(contains_alias(normalized, alias) for alias in aliases):
            found_keywords.append(keyword)
    return found_keywords


def split_job_description_lines(job_text: str) -> list[str]:
    """Split a job description into simple non-empty lines for lightweight parsing."""
    return [line.strip(" -\t") for line in job_text.splitlines() if line.strip(" -\t")]


def is_preferred_line(line: str) -> bool:
    """Return True when a line describes plus/preferred skills."""
    normalized_line = normalize_text(line)
    return any(contains_alias(normalized_line, phrase) for phrase in PREFERRED_LANGUAGE)


def add_unique(items: list[str], item: str) -> None:
    """Append an item only once while preserving discovery order."""
    if item not in items:
        items.append(item)


def all_scored_keywords() -> set[str]:
    """Return every keyword that belongs to a scoring category."""
    keywords: set[str] = set()
    for category in SCORE_CATEGORIES.values():
        keywords.update(category["keywords"])
    return keywords


def infer_candidate_experience_profile(candidate_text: str) -> CandidateProfile:
    """Infer a conservative experience profile from explicit candidate text."""
    normalized = normalize_text(candidate_text)
    lowered = candidate_text.lower()
    evidence: list[str] = []
    career_level = "unknown"

    level_patterns = [
        ("senior", [r"\bsenior\b", r"\bstaff\b", r"\bprincipal\b"]),
        ("mid", [r"\bmid[- ]level\b"]),
        ("junior", [r"\bjunior\b"]),
        ("new_grad", [r"\brecent graduate\b", r"\bnew graduate\b", r"\bnew grad\b"]),
        ("student", [r"\bcurrent student\b", r"\bundergraduate\b", r"\bgraduate student\b"]),
    ]
    for level, patterns in level_patterns:
        if any(re.search(pattern, normalized) for pattern in patterns):
            career_level = level
            evidence.append(f"Explicit candidate career-level phrase: {level.replace('_', ' ')}.")
            break
    if career_level == "unknown" and re.search(r"\b(?:intern|internship)\b", normalized):
        career_level = "student"
        evidence.append("Candidate source explicitly mentions an internship or intern role.")

    years_experience: float | None = None
    years_matches = [
        float(value)
        for value in re.findall(
            r"\b(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\s+(?:of\s+)?(?:professional\s+|work\s+)?experience\b",
            lowered,
        )
    ]
    if years_matches:
        years_experience = max(years_matches)
        evidence.append(f"Candidate source states {years_experience:g} years of experience.")
        if career_level == "unknown":
            career_level = "senior" if years_experience >= 5 else "mid" if years_experience >= 3 else "junior"

    degree_patterns = [
        ("phd", r"\b(?:ph\s*d|doctorate|doctoral degree)\b"),
        ("master", r"\b(?:master\s+s|masters|master of|m\s+s)\b"),
        ("bachelor", r"\b(?:bachelor\s+s|bachelors|bachelor of|b\s+s)\b"),
    ]
    highest_degree = "unknown"
    for degree, pattern in degree_patterns:
        if re.search(pattern, normalized):
            highest_degree = degree
            evidence.append(f"Candidate source explicitly mentions a {degree} degree.")
            break

    if career_level not in CAREER_LEVELS:
        career_level = "unknown"

    return {
        "career_level": career_level,
        "years_experience": years_experience,
        "highest_degree": highest_degree,
        "evidence": evidence,
    }


def explicit_job_experience_levels(job_text: str) -> tuple[list[str], list[str]]:
    """Return required and preferred experience-level scoring terms."""
    required: list[str] = []
    preferred: list[str] = []
    for line in split_job_description_lines(job_text):
        normalized = normalize_text(line)
        target = preferred if is_preferred_line(line) else required
        if re.search(r"\b(?:5\+?|five|at least 5)\s*(?:years|yrs)\b", normalized):
            add_unique(target, "senior-level")
        elif re.search(r"\b(?:3\+?|three|at least 3)\s*(?:years|yrs)\b", normalized):
            add_unique(target, "mid-level")
        if re.search(r"^(?:role|title)?\s*:?[ ]*(?:senior|staff|principal)\b", line.strip(), re.I):
            add_unique(target, "senior-level")
        if re.search(
            r"^(?:role|title)?\s*:?[ ]*(?:engineering|product|data|machine learning) manager\b",
            line.strip(),
            re.I,
        ):
            add_unique(target, "senior-level")
    return required, preferred


def is_uk_job(job_text: str) -> bool:
    """Return True when a job description appears to be UK or London based."""
    normalized_job = normalize_text(job_text)
    uk_terms = ["london", "united kingdom", "great britain", "uk", "adzuna.co.uk", "adzuna.gb"]
    return any(contains_alias(normalized_job, term) for term in uk_terms)


def asks_for_uk_work_authorization_review(job_text: str) -> bool:
    """Detect UK work authorization or sponsorship language that needs review."""
    normalized_job = normalize_text(job_text)
    phrases = [
        "right to work in the uk",
        "uk work authorization",
        "uk work authorisation",
        "visa sponsorship",
        "sponsorship required",
        "sponsorship is not available",
        "no visa sponsorship",
        "skilled worker sponsorship",
    ]
    return any(contains_alias(normalized_job, phrase) for phrase in phrases)


def must_already_have_uk_work_authorization(job_text: str) -> bool:
    """Detect language requiring current UK right-to-work status."""
    normalized_job = normalize_text(job_text)
    patterns = [
        r"\bmust\s+(?:already|currently)\s+have\s+(?:the\s+)?right\s+to\s+work\s+in\s+the\s+uk\b",
        r"\bmust\s+have\s+(?:the\s+)?right\s+to\s+work\s+in\s+the\s+uk\b",
        r"\bmust\s+be\s+(?:already\s+|currently\s+)?(?:authorized|authorised)\s+to\s+work\s+in\s+the\s+uk\b",
        r"\bcurrently\s+(?:authorized|authorised)\s+to\s+work\s+in\s+the\s+uk\b",
        r"\balready\s+(?:authorized|authorised)\s+to\s+work\s+in\s+the\s+uk\b",
    ]
    return any(re.search(pattern, normalized_job) for pattern in patterns)


def find_red_flags(job_text: str, resume_text: str) -> list[str]:
    """Find requirements that need human review without assuming eligibility."""
    normalized_job = normalize_text(job_text)
    normalized_resume = normalize_text(resume_text)
    red_flags = []

    for rule in RED_FLAG_RULES:
        job_mentions_rule = any(re.search(pattern, normalized_job) for pattern in rule["patterns"])
        resume_mentions_rule = any(re.search(pattern, normalized_resume) for pattern in rule["patterns"])
        if job_mentions_rule and not resume_mentions_rule:
            red_flags.append(
                f"{rule['name']}: job mentions this, but the candidate source does not provide eligibility information."
            )

    if is_uk_job(job_text):
        red_flags.append(UK_HPI_NOTE)
        if asks_for_uk_work_authorization_review(job_text):
            red_flags.append(UK_HPI_MANUAL_REVIEW_WARNING)
        if must_already_have_uk_work_authorization(job_text):
            red_flags.append(UK_ALREADY_AUTHORIZED_WARNING)

    return red_flags


def parse_job_description(job_text: str, resume_text: str) -> ParsedJob:
    """Extract transparent, demand-aware facts from the job description."""
    required_skills: list[str] = []
    preferred_skills: list[str] = []
    domain_keywords: list[str] = []
    experience_level: list[str] = []
    degree_requirements: list[str] = []
    red_flags = find_red_flags(job_text, resume_text)

    domain_terms = set(SCORE_CATEGORIES["Domain fit"]["keywords"])
    for line in split_job_description_lines(job_text):
        line_keywords = find_keywords(line)
        preferred_line = is_preferred_line(line)
        target_list = preferred_skills if preferred_line else required_skills

        for keyword in line_keywords:
            if keyword in EXPERIENCE_LEVEL_KEYWORDS:
                add_unique(preferred_skills if preferred_line else experience_level, keyword)
            elif keyword in domain_terms:
                add_unique(domain_keywords, keyword)
                if preferred_line and keyword in required_skills:
                    required_skills.remove(keyword)
                add_unique(target_list, keyword)
            elif keyword in all_scored_keywords():
                if preferred_line and keyword in required_skills:
                    required_skills.remove(keyword)
                add_unique(target_list, keyword)

    normalized_job = normalize_text(job_text)
    if re.search(
        r"\b(?:bachelor'?s|bs|b\.s\.)\s+(?:degree\s+)?(?:required|preferred|or equivalent)?\b",
        normalized_job,
    ):
        degree_requirements.append("Bachelor's degree mentioned: verify it against the candidate source.")
    if re.search(
        r"\b(?:master\s+s|masters|m\s+s|ph\s*d|doctorate)\s+(?:degree\s+)?required\b|"
        r"\brequires?\s+(?:a\s+)?(?:master\s+s|masters|m\s+s|ph\s*d|doctorate)\b",
        normalized_job,
    ):
        degree_requirements.append("Master's degree or PhD required")
        red_flags.append("Graduate degree requirement: verify the required degree against the candidate source.")

    required_experience, preferred_experience = explicit_job_experience_levels(job_text)
    for keyword in required_experience:
        add_unique(experience_level, keyword)
    for keyword in preferred_experience:
        add_unique(preferred_skills, keyword)

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_level": experience_level,
        "degree_requirements": degree_requirements,
        "domain_keywords": domain_keywords,
        "red_flags": red_flags,
    }
