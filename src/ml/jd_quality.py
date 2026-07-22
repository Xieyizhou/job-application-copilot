"""Local, explainable job-description quality classification.

The classifier evaluates whether a saved posting contains enough job-specific
evidence to support fit analysis.  It does not score the candidate and does not
use personal resume data.
"""

from __future__ import annotations

import re
from typing import Any


QUALITY_SCHEMA_VERSION = 1

_SECTION_PATTERNS = {
    "responsibilities": re.compile(
        r"(?im)^\s*(?:#+\s*)?(?:responsibilities|what you(?:'|’)ll do|the role|duties)\s*:?[ \t]*$"
    ),
    "requirements": re.compile(
        r"(?im)^\s*(?:#+\s*)?(?:requirements|qualifications|what you(?:'|’)ll need|about you)\s*:?[ \t]*$"
    ),
    "company": re.compile(r"(?im)^\s*(?:#+\s*)?(?:about (?:us|the company)|company)\s*:?[ \t]*$"),
    "benefits": re.compile(r"(?im)^\s*(?:#+\s*)?(?:benefits|what we offer|compensation)\s*:?[ \t]*$"),
}
_REQUIREMENT_SIGNALS = re.compile(
    r"(?i)\b(?:required|requirements?|qualifications?|must have|need to have|"
    r"proficien(?:t|cy)|experience (?:with|in)|knowledge of|ability to|"
    r"years? of experience|degree|skills?)\b"
)
_RESPONSIBILITY_SIGNALS = re.compile(
    r"(?i)\b(?:responsibilities|you will|design|develop|build|manage|analy[sz]e|"
    r"implement|maintain|collaborate|lead|support)\b"
)
_BOILERPLATE_SIGNALS = re.compile(
    r"(?i)\b(?:equal opportunity employer|affirmative action|reasonable accommodation|"
    r"background check|privacy policy|terms and conditions|diversity and inclusion|"
    r"protected veteran|disability status|employment eligibility verification)\b"
)


def extract_description_body(job_text: str) -> str:
    """Return the complete JD body from a saved Markdown job record.

    Employer postings commonly use level-two Markdown headings for sections such
    as Responsibilities and Requirements.  Those headings are part of the job
    description and must not be mistaken for toolkit metadata boundaries.
    """
    marker = re.search(r"(?im)^##\s+Job Description\s*$", job_text)
    if marker is None:
        return job_text.strip()
    return job_text[marker.end() :].strip()


def _metadata_value(job_text: str, field: str) -> str:
    match = re.search(rf"(?im)^{re.escape(field)}:\s*(.+?)\s*$", job_text)
    return match.group(1).strip().lower() if match else ""


def _content_units(body: str) -> list[str]:
    units: list[str] = []
    for line in body.splitlines():
        cleaned = line.strip(" \t-*•")
        if not cleaned:
            continue
        units.extend(part.strip() for part in re.split(r"(?<=[.!?])\s+", cleaned) if part.strip())
    return units or ([body.strip()] if body.strip() else [])


def _boilerplate_share(units: list[str]) -> float:
    total_words = sum(len(unit.split()) for unit in units)
    boilerplate_words = sum(len(unit.split()) for unit in units if _BOILERPLATE_SIGNALS.search(unit))
    return boilerplate_words / total_words if total_words else 0.0


def _source_hints(job_text: str, word_count: int) -> tuple[bool, bool, bool]:
    source = _metadata_value(job_text, "Source")
    description_source = _metadata_value(job_text, "Description Source")
    fetch_status = _metadata_value(job_text, "JD Fetch Status")
    saved_source_record = bool(source or description_source or fetch_status)
    explicit_full = (
        "full_jd" in description_source
        or fetch_status in {"complete", "full", "full_jd"}
        or source in {"jsearch", "manual", "company website", "company_site"}
    )
    explicit_snippet = (
        "snippet" in description_source
        or "summary" in description_source
        or fetch_status in {"snippet", "snippet_only", "missing", "partial"}
        or (source in {"adzuna", "jooble"} and word_count < 180)
    )
    return saved_source_record, explicit_full, explicit_snippet


def classify_jd_quality(job_text: str) -> dict[str, Any]:
    """Classify JD completeness from local text and source provenance.

    The returned score describes document quality, not candidate fit.  Rules and
    component signals are returned so every classification remains auditable.
    """
    body = extract_description_body(job_text)
    units = _content_units(body)
    word_count = len(re.findall(r"\b[\w+#.-]+\b", body))
    section_hits = [name for name, pattern in _SECTION_PATTERNS.items() if pattern.search(body)]
    requirement_count = sum(1 for unit in units if _REQUIREMENT_SIGNALS.search(unit))
    responsibility_count = sum(1 for unit in units if _RESPONSIBILITY_SIGNALS.search(unit))
    boilerplate_share = _boilerplate_share(units)
    visible_truncation = bool(re.search(r"(?:\.\.\.|…)", body))
    saved_source_record, explicit_full, explicit_snippet = _source_hints(job_text, word_count)

    length_points = min(35.0, (word_count / 250.0) * 35.0)
    requirement_points = min(30.0, requirement_count * 6.0)
    section_points = min(20.0, len(section_hits) * 5.0)
    content_points = 15.0 * max(0.0, 1.0 - boilerplate_share) if word_count else 0.0
    provenance_adjustment = 8.0 if explicit_full and word_count else (-18.0 if explicit_snippet else 0.0)
    truncation_penalty = 15.0 if visible_truncation else 0.0
    quality_score = round(
        max(
            0.0,
            min(
                100.0,
                length_points
                + requirement_points
                + section_points
                + content_points
                + provenance_adjustment
                - truncation_penalty,
            ),
        )
    )

    scoring_ready = (
        word_count >= 120
        and requirement_count >= 3
        and responsibility_count >= 2
        and not visible_truncation
        and not explicit_snippet
        and boilerplate_share < 0.45
    )
    provisional_ready = (
        word_count >= 20
        and requirement_count >= 1
        and boilerplate_share < 0.65
    )

    if word_count < 10:
        label = "empty_or_unreadable"
        display_label = "Empty or unreadable"
    elif explicit_snippet or (word_count < 80 and visible_truncation):
        label = "likely_snippet"
        display_label = "Likely snippet"
    elif boilerplate_share >= 0.45:
        label = "boilerplate_heavy"
        display_label = "Boilerplate-heavy"
    elif requirement_count < 2:
        label = "requirements_missing"
        display_label = "Requirements missing"
    elif scoring_ready:
        label = "scoring_ready"
        display_label = "Scoring-ready"
    else:
        label = "partial_jd"
        display_label = "Partial JD"

    classified_incomplete = label in {
        "empty_or_unreadable",
        "likely_snippet",
        "boilerplate_heavy",
        "requirements_missing",
    }
    # Raw strings in regression tests and manual analysis can be intentionally concise.
    # Source-backed records receive the stronger classifier gate; unsaved text retains
    # the historic minimum-text compatibility boundary.
    appears_incomplete = classified_incomplete if saved_source_record else word_count < 20
    reasons: list[str] = []
    if explicit_snippet:
        reasons.append("Source metadata or provider behavior indicates a discovery snippet.")
    if visible_truncation:
        reasons.append("Visible ellipsis or truncation marker was detected.")
    if word_count < 120:
        reasons.append(f"Only {word_count} JD words were available; reliable scoring usually needs a fuller posting.")
    if requirement_count < 3:
        reasons.append(f"Only {requirement_count} requirement-like statements were detected.")
    if boilerplate_share >= 0.45:
        reasons.append("Employer boilerplate occupies too much of the available text.")
    if scoring_ready:
        reasons.append("Responsibilities and multiple requirements are present without truncation signals.")

    next_action = {
        "empty_or_unreadable": "Paste or upload the complete job description.",
        "likely_snippet": "Replace the discovery snippet with the original full posting.",
        "boilerplate_heavy": "Add the role-specific responsibilities and qualifications.",
        "requirements_missing": "Add the qualifications or requirements section.",
        "partial_jd": "Verify the posting against the employer's complete job page.",
        "scoring_ready": "Review the extracted requirements before trusting the fit result.",
    }[label]

    confidence = "high" if explicit_full or explicit_snippet or label in {"empty_or_unreadable", "scoring_ready"} else "medium"
    return {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "label": label,
        "display_label": display_label,
        "quality_score": quality_score,
        "classification_confidence": confidence,
        "word_count": word_count,
        "section_count": len(section_hits),
        "sections": section_hits,
        "requirement_statement_count": requirement_count,
        "responsibility_statement_count": responsibility_count,
        "boilerplate_share": round(boilerplate_share, 3),
        "visible_truncation": visible_truncation,
        "saved_source_record": saved_source_record,
        "explicit_full_source": explicit_full,
        "explicit_snippet_source": explicit_snippet,
        "appears_incomplete": appears_incomplete,
        "provisional_scoring_ready": provisional_ready,
        "reliable_scoring_ready": scoring_ready,
        "reasons": reasons[:4],
        "next_action": next_action,
    }
