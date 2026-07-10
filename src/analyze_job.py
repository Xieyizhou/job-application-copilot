"""Analyze a job description against the resume source of truth.

Version 0 intentionally uses simple keyword matching. It does not scrape job
sites, submit applications, or invent any candidate facts.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

from output_paths import application_package_dir
from workspace import Workspace, WorkspaceError, demo_workspace, personal_workspace


PROJECT_ROOT = Path(__file__).resolve().parents[1]


# Each keyword has optional aliases so the script can match common variations
# while still reporting a clean skill name.
KEYWORD_CATALOG = {
    "Python": ["python"],
    "pandas": ["pandas"],
    "scikit-learn": ["scikit-learn", "sklearn"],
    "SQL": ["sql"],
    "machine learning": ["machine learning", "ml"],
    "model evaluation": ["model evaluation", "evaluate models", "model performance"],
    "data visualization": ["data visualization", "visualization", "charts"],
    "data analysis": ["data analysis", "analytics"],
    "UAV": ["uav", "drone", "aerial"],
    "robotics": ["robotics", "robot"],
    "sensor data": ["sensor data", "inspection data"],
    "thermal data": ["thermal data", "thermal-analysis", "temperature-difference"],
    "route planning": ["route planning", "route-planning", "path planning"],
    "game AI": ["game ai", "npc", "game artificial intelligence"],
    "econometrics": ["econometrics"],
    "statistical analysis": ["statistical analysis", "statistics"],
    "entry-level": ["entry-level", "entry level"],
    "new grad": ["new grad", "new graduate"],
    "intern": ["intern", "internship"],
    "junior": ["junior"],
    "recent graduate": ["recent graduate", "recent grad", "undergraduate graduate"],
    "classification": ["classification", "supervised classification"],
    "PCA": ["pca", "principal component analysis"],
    "feature engineering": ["feature engineering", "engineered timing-summary features"],
    "CNN": ["cnn", "cnns", "convolutional neural network"],
    "reinforcement learning": ["reinforcement learning"],
    "causal inference": ["causal inference", "difference-in-differences", "did"],
    "communication": ["communication", "technical explanation", "instructional sessions"],
    "teaching": ["teaching", "peer instruction", "instructional assistant"],
    "teamwork": ["teamwork", "collaboration", "collaborate", "collaborative"],
    "documentation": ["documentation", "document", "write reports"],
    "presentation": ["presentation", "present", "presenting"],
    "R": [" r ", "r programming"],
    "MATLAB": ["matlab"],
    "C#": ["c#", "c sharp"],
    "Excel": ["excel", "spreadsheets"],
    "NumPy": ["numpy"],
}


SCORE_CATEGORIES = {
    "Core technical skills": {
        "points": 40,
        "keywords": [
            "Python",
            "pandas",
            "scikit-learn",
            "SQL",
            "machine learning",
            "model evaluation",
            "data visualization",
            "data analysis",
        ],
    },
    "Domain fit": {
        "points": 25,
        "keywords": [
            "UAV",
            "robotics",
            "sensor data",
            "thermal data",
            "route planning",
            "game AI",
            "econometrics",
            "statistical analysis",
        ],
    },
    "Experience level fit": {
        "points": 15,
        "keywords": [
            "entry-level",
            "new grad",
            "intern",
            "junior",
            "recent graduate",
        ],
    },
    "Project relevance": {
        "points": 10,
        "keywords": [
            "classification",
            "PCA",
            "feature engineering",
            "CNN",
            "reinforcement learning",
            "causal inference",
        ],
    },
    "Communication / collaboration fit": {
        "points": 10,
        "keywords": [
            "communication",
            "teaching",
            "teamwork",
            "documentation",
            "presentation",
        ],
    },
}


PENALTY_RULES = [
    {
        "name": "5+ years of experience",
        "points": 20,
        "patterns": [
            r"\b5\+?\s*(?:years|yrs)\b",
            r"\bfive\+?\s*(?:years|yrs)\b",
            r"\bat least\s+5\s*(?:years|yrs)\b",
        ],
    },
    {
        "name": "3+ years of experience",
        "points": 10,
        "patterns": [
            r"\b3\+?\s*(?:years|yrs)\b",
            r"\bthree\+?\s*(?:years|yrs)\b",
            r"\bat least\s+3\s*(?:years|yrs)\b",
        ],
    },
    {
        "name": "Master's degree or PhD required",
        "points": 15,
        "patterns": [
            r"\brequires?\s+(?:a\s+)?(?:master'?s|ms|ph\.?d\.?|doctorate)\b",
            r"\bmust\s+have\s+(?:a\s+)?(?:master'?s|ms|ph\.?d\.?|doctorate)\b",
            r"\b(?:master'?s|ms|ph\.?d\.?|doctorate)\s+(?:degree\s+)?required\b",
            r"\bminimum\s+(?:of\s+)?(?:a\s+)?(?:master'?s|ms|ph\.?d\.?|doctorate)\b",
        ],
    },
    {
        "name": "Senior, staff, principal, or manager-level role",
        "points": 20,
        "patterns": [
            r"\bsenior\b",
            r"\bstaff\b",
            r"\bprincipal\b",
            r"\bmanager\b",
            r"\bmanagement\b",
        ],
    },
]


RED_FLAG_RULES = [
    {
        "name": "Citizenship, permanent residency, or work authorization requirement",
        "patterns": [
            r"\bcitizenship\b",
            r"\bcitizen\b",
            r"\bpermanent\s+residen(?:t|cy)\b",
            r"\bgreen\s+card\b",
            r"\bwork\s+authorization\b",
            r"\bauthorized\s+to\s+work\b",
            r"\bvisa\b",
            r"\bsponsorship\b",
        ],
    }
]

UK_HPI_NOTE = (
    "UK work authorization note: verify the candidate's current status and any available visa route "
    "against official guidance. Do not claim work authorization unless the candidate has confirmed it."
)
UK_HPI_MANUAL_REVIEW_WARNING = (
    "Manual review required: confirm whether the employer's sponsorship and work-authorization "
    "requirements match the candidate's actual status."
)
UK_ALREADY_AUTHORIZED_WARNING = (
    "Manual review required: this JD appears to require candidates to already or currently "
    "have the right to work in the UK. Because current authorization is not confirmed, the score is capped at 74."
)


# These are honest adjacent matches. They should improve the score, but the
# report should not pretend they are exact resume claims.
PARTIAL_RESUME_MATCHES = {
    "robotics": (
        {"UAV", "route planning"},
        "Partial match: the candidate source contains adjacent UAV or route-planning evidence.",
    ),
    "sensor data": (
        {"UAV", "thermal data"},
        "Partial match: the candidate source contains adjacent UAV or thermal-data evidence.",
    ),
}


PREFERRED_LANGUAGE = [
    "plus",
    "preferred",
    "nice to have",
    "bonus",
    "desired",
    "would be helpful",
]


EXPERIENCE_LEVEL_KEYWORDS = [
    "entry-level",
    "new grad",
    "intern",
    "junior",
    "recent graduate",
]


EXPERIENCE_THEMES = {
    "Python data analysis": [
        "Python",
        "pandas",
        "NumPy",
        "data analysis",
        "statistics",
        "data visualization",
    ],
    "Machine learning model evaluation": [
        "machine learning",
        "model evaluation",
        "scikit-learn",
        "classification",
        "PCA",
    ],
    "UAV inspection algorithms": [
        "UAV",
        "sensor data",
        "route planning",
        "obstacle avoidance",
    ],
    "Game AI and reinforcement learning": [
        "game AI",
        "reinforcement learning",
        "CNN",
        "C#",
    ],
    "Econometrics and statistical reasoning": [
        "econometrics",
        "regression",
        "causal inference",
        "statistical analysis",
    ],
    "Teaching and communication": [
        "communication",
        "teaching",
        "Python",
        "data analysis",
    ],
}


def normalize_text(text: str) -> str:
    """Lowercase text and pad it with spaces to make word matching simpler."""
    text = text.lower()
    text = re.sub(r"[^a-z0-9+#.\-]+", " ", text)
    return f" {text} "


def contains_alias(normalized_text: str, alias: str) -> bool:
    """Return True when an alias appears as a simple phrase or token."""
    normalized_alias = normalize_text(alias).strip()
    if not normalized_alias:
        return False

    # Short aliases like "r" and "ml" need token boundaries to avoid matching
    # unrelated words. Longer phrases can be found directly in normalized text.
    if len(normalized_alias) <= 2 and normalized_alias.isalnum():
        return re.search(rf"(?<![a-z0-9]){re.escape(normalized_alias)}(?![a-z0-9])", normalized_text) is not None

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


def parse_job_description(job_text: str, resume_text: str) -> dict[str, object]:
    """Extract simple demand-aware facts from the job description.

    This is intentionally lightweight and transparent. It does not infer hidden
    requirements; it only labels keywords that are actually present in the JD.
    """
    required_skills = []
    preferred_skills = []
    domain_keywords = []
    experience_level = []
    degree_requirements = []
    red_flags = find_red_flags(job_text, resume_text)

    domain_terms = set(SCORE_CATEGORIES["Domain fit"]["keywords"])
    lines = split_job_description_lines(job_text)

    for line in lines:
        line_keywords = find_keywords(line)
        preferred_line = is_preferred_line(line)
        target_list = preferred_skills if preferred_line else required_skills

        for keyword in line_keywords:
            if keyword in EXPERIENCE_LEVEL_KEYWORDS:
                add_unique(experience_level, keyword)
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
    if any(
        re.search(pattern, normalized_job)
        for pattern in PENALTY_RULES[2]["patterns"]
    ):
        degree_requirements.append("Master's degree or PhD required")
        red_flags.append(
            "Graduate degree requirement: verify the required degree against the candidate source."
        )

    return {
        "required_skills": required_skills,
        "preferred_skills": preferred_skills,
        "experience_level": experience_level,
        "degree_requirements": degree_requirements,
        "domain_keywords": domain_keywords,
        "red_flags": red_flags,
    }


def all_scored_keywords() -> set[str]:
    """Return every keyword that belongs to a scoring category."""
    keywords = set()
    for category in SCORE_CATEGORIES.values():
        keywords.update(category["keywords"])
    return keywords


def add_unique(items: list[str], item: str) -> None:
    """Append an item only once while preserving discovery order."""
    if item not in items:
        items.append(item)


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


def calculate_score_breakdown(
    parsed_job: dict[str, object],
    resume_keywords: list[str],
) -> list[dict[str, object]]:
    """Calculate demand-aware weighted category scores.

    A category is scored only if the JD mentions relevant terms. Required terms
    count at full weight. Preferred/plus terms count at half weight so missing
    nice-to-have items do not dominate the final score.
    """
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
            possible_weight += 1.0

            match_strength = match_strength_for_keyword(keyword, resume_keyword_set)
            earned_weight += match_strength if demand_type == "required" else match_strength * 0.5

            if match_strength == 1.0:
                matched.append(keyword)
            elif match_strength > 0:
                partial.append(f"{keyword} ({PARTIAL_RESUME_MATCHES[keyword][1]})")
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


def match_strength_for_keyword(keyword: str, resume_keyword_set: set[str]) -> float:
    """Return 1.0 for direct matches, 0.6 for adjacent matches, and 0.0 for gaps."""
    if keyword in EXPERIENCE_LEVEL_KEYWORDS:
        return 1.0
    if keyword in resume_keyword_set:
        return 1.0
    if keyword in PARTIAL_RESUME_MATCHES and PARTIAL_RESUME_MATCHES[keyword][0].intersection(resume_keyword_set):
        return 0.6
    return 0.0


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
        return f"Good fit: the resume supports several requested {category_name.lower()} terms, with any adjacent matches labeled as partial."
    return f"Weak fit: the JD asks for {category_name.lower()} terms that are not clearly supported by the resume."


def find_penalties(job_text: str) -> list[dict[str, object]]:
    """Find score penalties requested by the user."""
    normalized = normalize_text(job_text)
    penalties = []

    for rule in PENALTY_RULES:
        if any(re.search(pattern, normalized) for pattern in rule["patterns"]):
            penalties.append(
                {
                    "name": rule["name"],
                    "points": rule["points"],
                }
            )

    return penalties


def find_red_flags(job_text: str, resume_text: str) -> list[str]:
    """Find requirements that need human review without assuming eligibility."""
    normalized_job = normalize_text(job_text)
    normalized_resume = normalize_text(resume_text)
    red_flags = []

    for rule in RED_FLAG_RULES:
        job_mentions_rule = any(
            re.search(pattern, normalized_job) for pattern in rule["patterns"]
        )
        resume_mentions_rule = any(
            re.search(pattern, normalized_resume) for pattern in rule["patterns"]
        )

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


def is_uk_job(job_text: str) -> bool:
    """Return True when a job description appears to be UK or London based."""
    normalized_job = normalize_text(job_text)
    uk_terms = [
        "london",
        "united kingdom",
        "great britain",
        "uk",
        "adzuna.co.uk",
        "adzuna.gb",
    ]
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
    """Cap UK jobs that require current right-to-work status."""
    if is_uk_job(job_text) and must_already_have_uk_work_authorization(job_text):
        return min(score, 74)
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


def find_resume_evidence(themes: list[str]) -> list[str]:
    """Return resume-backed evidence bullets for the selected themes."""
    return [f"Candidate source contains keywords related to {theme}." for theme in themes]


def build_markdown_report(
    job_description_path: Path,
    resume_source_path: Path,
    parsed_job: dict[str, object],
    matched_skills: list[str],
    partial_matches: list[str],
    missing_skills: list[str],
    themes: list[str],
    score_breakdown: list[dict[str, object]],
    penalties: list[dict[str, object]],
    red_flags: list[str],
    resume_evidence: list[str],
    score: int,
    recommendation: str,
) -> str:
    """Build the Markdown report saved for human review."""
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M")

    return "\n".join(
        [
            "# Job Match Analysis",
            "",
            f"- Job description file: `{job_description_path}`",
            f"- Candidate source: `{resume_source_path}`",
            f"- Generated at: {generated_at}",
            "",
            "## Summary",
            "",
            f"- Match score: **{score}/100**",
            f"- Recommendation: **{recommendation}**",
            f"- Why: {explain_overall_score(score, recommendation, penalties, red_flags)}",
            "",
            "## Parsed Job Requirements",
            "",
            f"- Required skills: {format_inline_list(parsed_job['required_skills'])}",
            f"- Preferred / plus skills: {format_inline_list(parsed_job['preferred_skills'])}",
            f"- Experience level: {format_inline_list(parsed_job['experience_level'])}",
            f"- Degree requirements: {format_inline_list(parsed_job['degree_requirements'])}",
            f"- Domain keywords: {format_inline_list(parsed_job['domain_keywords'])}",
            "",
            "## Score Breakdown",
            "",
            format_score_breakdown(score_breakdown),
            "",
            "## Penalties",
            "",
            format_penalties(penalties),
            "",
            "## Matched Skills",
            "",
            format_bullets(matched_skills),
            "",
            "## Partial / Adjacent Matches",
            "",
            format_bullets(partial_matches),
            "",
            "## Missing Skills",
            "",
            format_bullets(missing_skills),
            "",
            "## Relevant Experience Themes",
            "",
            format_bullets(themes),
            "",
            "## Red Flags",
            "",
            format_bullets(red_flags),
            "",
            "## Relevant Resume Evidence",
            "",
            format_bullets(resume_evidence),
            "",
            "## Human Review Notes",
            "",
            "- This report uses weighted keyword matching and simple penalty rules.",
            "- Required JD terms carry full weight; preferred or plus terms carry partial weight.",
            "- Categories the JD does not mention are marked N/A and are not counted against the final score.",
            "- It should be reviewed by a person before preparing application materials.",
            "- It does not invent experience, skills, degree level, metrics, visa status, or work authorization.",
            "- Confirm the resume source's degree level before relying on education-related statements.",
            "- It does not submit applications or interact with job platforms.",
            "",
        ]
    )


def format_bullets(items: list[str]) -> str:
    """Format a list as Markdown bullets, or show a placeholder if empty."""
    if not items:
        return "- None found"
    return "\n".join(f"- {item}" for item in items)


def format_score_breakdown(score_breakdown: list[dict[str, object]]) -> str:
    """Format weighted category scores for the report."""
    lines = []

    for item in score_breakdown:
        if item["earned"] is None:
            lines.append(f"- {item['category']}: **N/A**")
            lines.append(f"  - {item['note']}")
            continue

        lines.append(f"- {item['category']}: **{item['earned']}/{item['possible']}**")
        lines.append(f"  - JD terms scored: {format_inline_list(item['active_terms'])}")
        lines.append(f"  - Matched: {format_inline_list(item['matched'])}")
        lines.append(f"  - Partial / adjacent: {format_inline_list(item['partial'])}")
        lines.append(f"  - Missing required or preferred terms: {format_inline_list(item['missing'])}")
        lines.append(f"  - Note: {item['note']}")

    return "\n".join(lines)


def format_penalties(penalties: list[dict[str, object]]) -> str:
    """Format score penalties for the report."""
    if not penalties:
        return "- None found"
    return "\n".join(f"- -{item['points']}: {item['name']}" for item in penalties)


def format_inline_list(items: object) -> str:
    """Format a list for use inside a sentence."""
    if not isinstance(items, list) or not items:
        return "None"
    return ", ".join(str(item) for item in items)


def collect_report_matches(
    score_breakdown: list[dict[str, object]]
) -> tuple[list[str], list[str], list[str]]:
    """Collect matched, partial, and missing terms from active scored categories."""
    matched = []
    partial = []
    missing = []

    for item in score_breakdown:
        for keyword in item["matched"]:
            add_unique(matched, keyword)
        for keyword in item["partial"]:
            add_unique(partial, keyword)
        for keyword in item["missing"]:
            add_unique(missing, keyword)

    return matched, partial, missing


def explain_overall_score(
    score: int,
    recommendation: str,
    penalties: list[dict[str, object]],
    red_flags: list[str],
) -> str:
    """Write a short plain-English explanation for the final score."""
    explanation = (
        f"{recommendation} because the score is {score}/100 after normalizing only "
        "the categories the job description actually mentions."
    )
    if penalties:
        explanation += " Penalties were applied for seniority, experience, or degree requirements."
    if red_flags:
        explanation += " Red flags need human review before applying."
    return explanation


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


def analyze_job_structured(job_text: str, resume_text: str, raw_analysis: str = "") -> dict[str, object]:
    """Return structured, dependency-light fit analysis for UI display.

    This helper preserves the original Markdown analyzer contract by sitting
    beside ``analyze_job()`` instead of changing its return value.
    """
    job_keywords = find_keywords(job_text)
    resume_keywords = find_keywords(resume_text)
    parsed_job = parse_job_description(job_text, resume_text)
    themes = choose_relevant_themes(job_keywords, resume_keywords)
    score_breakdown = calculate_score_breakdown(parsed_job, resume_keywords)
    matched_keywords, partial_matches, missing_keywords = collect_report_matches(score_breakdown)
    penalties = find_penalties(job_text)
    red_flags = list(parsed_job["red_flags"])
    score = calculate_match_score(score_breakdown, penalties)
    score = apply_uk_work_authorization_score_cap(score, job_text)
    recommendation = recommendation_for_score(score)
    main_reason = explain_overall_score(score, recommendation, penalties, red_flags)

    matched_strengths = [
        f"Resume source supports requested keyword: {keyword}."
        for keyword in matched_keywords[:6]
    ]
    if partial_matches and len(matched_strengths) < 6:
        matched_strengths.extend(
            f"Adjacent evidence may support: {keyword}."
            for keyword in partial_matches[: 6 - len(matched_strengths)]
        )
    if not matched_strengths:
        matched_strengths.append("No strong keyword overlap was detected; review manually.")

    weak_areas = [f"Missing or unclear evidence for: {keyword}." for keyword in missing_keywords[:6]]
    if red_flags and len(weak_areas) < 6:
        weak_areas.extend(red_flags[: 6 - len(weak_areas)])
    if not weak_areas:
        weak_areas.append("No major weak areas were detected by the lightweight analyzer.")

    resume_evidence = find_resume_evidence(themes)
    return {
        "score": score,
        "recommendation": recommendation,
        "main_reason": main_reason,
        "main_risk": red_flags[0] if red_flags else (weak_areas[0] if weak_areas else "No major risk detected."),
        "matched_strengths": matched_strengths[:6],
        "weak_areas": weak_areas[:6],
        "matched_keywords": matched_keywords,
        "missing_keywords": missing_keywords,
        "optional_keywords": list(parsed_job["preferred_skills"]),
        "resume_suggestions": resume_suggestions_for_keywords(matched_keywords, missing_keywords, red_flags),
        "jd_evidence": short_evidence_snippets(job_text, matched_keywords or job_keywords),
        "profile_evidence": resume_evidence[:3],
        "raw_analysis": raw_analysis,
    }


def save_report(
    job_description_path: Path,
    report: str,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> Path:
    """Save a Markdown report in a structured generated application folder."""
    if package_dir is None:
        package_dir = application_package_dir(workspace.generated_dir, job_description_path.stem)

    package_dir.mkdir(parents=True, exist_ok=True)
    report_path = package_dir / "analysis.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def analyze_job(
    job_description_path: Path,
    workspace: Workspace,
    package_dir: Path | None = None,
) -> tuple[str, Path]:
    """Run the analysis and return the report text plus saved report path."""
    workspace.require_writable()
    assert workspace.resume_source_path is not None
    resume_text = workspace.resume_source_path.read_text(encoding="utf-8")
    job_text = job_description_path.read_text(encoding="utf-8")

    job_keywords = find_keywords(job_text)
    resume_keywords = find_keywords(resume_text)

    parsed_job = parse_job_description(job_text, resume_text)
    themes = choose_relevant_themes(job_keywords, resume_keywords)
    score_breakdown = calculate_score_breakdown(parsed_job, resume_keywords)
    matched_skills, partial_matches, missing_skills = collect_report_matches(score_breakdown)
    penalties = find_penalties(job_text)
    red_flags = parsed_job["red_flags"]
    resume_evidence = find_resume_evidence(themes)
    score = calculate_match_score(score_breakdown, penalties)
    score = apply_uk_work_authorization_score_cap(score, job_text)
    recommendation = recommendation_for_score(score)

    report = build_markdown_report(
        job_description_path=job_description_path,
        resume_source_path=workspace.resume_source_path,
        parsed_job=parsed_job,
        matched_skills=matched_skills,
        partial_matches=partial_matches,
        missing_skills=missing_skills,
        themes=themes,
        score_breakdown=score_breakdown,
        penalties=penalties,
        red_flags=red_flags,
        resume_evidence=resume_evidence,
        score=score,
        recommendation=recommendation,
    )
    report_path = save_report(job_description_path, report, workspace, package_dir)
    return report, report_path


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze a job description against the selected candidate workspace."
    )
    parser.add_argument(
        "job_description",
        help="Path to a Markdown or text job description file.",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Analyze with sanitized Demo candidate data without writing files.",
    )
    return parser.parse_args()


def main() -> None:
    """Command-line entry point."""
    args = parse_args()
    job_description_path = Path(args.job_description).expanduser()

    if not job_description_path.is_absolute():
        job_description_path = PROJECT_ROOT / job_description_path

    if not job_description_path.exists():
        raise FileNotFoundError(f"Job description file was not found: {job_description_path}")

    if args.demo:
        workspace = demo_workspace()
        workspace.require_ready()
        assert workspace.resume_source_path is not None
        analysis = analyze_job_structured(
            job_description_path.read_text(encoding="utf-8"),
            workspace.resume_source_path.read_text(encoding="utf-8"),
        )
        print(json.dumps(analysis, indent=2))
        return

    workspace = personal_workspace()
    try:
        workspace.require_ready()
    except WorkspaceError as error:
        raise SystemExit(str(error)) from None
    report, report_path = analyze_job(job_description_path, workspace)
    print(report)
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
