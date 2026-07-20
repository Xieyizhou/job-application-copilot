"""Pure job-title normalization helpers for the Streamlit dashboard."""

from __future__ import annotations

import re
from typing import Any


INTERNAL_TITLE_FALLBACKS = {
    "example_ml_job": "Machine Learning Intern",
    "sample_package": "Sample Cover Letter Bundle",
}

PLACEHOLDER_JOB_TITLES = {
    "sample job",
    "test job",
    "demo job",
    "unknown role",
    "untitled job",
    "not provided",
    "n/a",
}

NON_ROLE_HEADINGS = {
    "job description",
    "description",
    "about the role",
    "requirements",
    "qualifications",
    "company",
    "location",
}

ROLE_HEADING_TERMS = {
    "administrator",
    "analyst",
    "architect",
    "associate",
    "consultant",
    "coordinator",
    "developer",
    "director",
    "engineer",
    "fellow",
    "intern",
    "lead",
    "manager",
    "officer",
    "owner",
    "researcher",
    "scientist",
    "specialist",
}


def read_markdown_field(markdown_text: str, field_name: str, default: str = "") -> str:
    """Read a simple ``Field: value`` line from Markdown."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


def looks_like_internal_slug(value: str) -> bool:
    """Return True for internal file or bundle slugs that should not be displayed."""
    cleaned = str(value or "").strip()
    if not cleaned:
        return False
    lower_cleaned = cleaned.lower()
    if lower_cleaned in INTERNAL_TITLE_FALLBACKS:
        return True
    if "\\" in cleaned or cleaned.lower().endswith((".md", ".json", ".txt")):
        return True
    if cleaned.startswith("/") or cleaned.count("/") > 1:
        return True
    if "_" not in cleaned:
        return False
    if not re.fullmatch(r"[a-z0-9]+(?:_[a-z0-9]+)+", lower_cleaned):
        return False
    internal_tokens = {"cover", "demo", "example", "generated", "job", "package", "sample"}
    return bool(set(lower_cleaned.split("_")) & internal_tokens)


def is_placeholder_job_title(value: Any) -> bool:
    """Return True only for exact normalized placeholder job titles."""
    normalized = " ".join(str(value or "").split()).casefold()
    return normalized in PLACEHOLDER_JOB_TITLES


def display_title_from_value(value: Any, fallback: str = "Missing job title") -> str:
    """Convert a stored title or role into a safe user-facing label."""
    cleaned = " ".join(str(value or "").split()).strip()
    if not cleaned or is_placeholder_job_title(cleaned):
        return fallback
    if looks_like_internal_slug(cleaned):
        return INTERNAL_TITLE_FALLBACKS.get(cleaned.lower(), fallback)
    return cleaned


def first_role_heading(markdown_text: str, company: str = "") -> str:
    """Return the first clear role-like H1 heading from saved Markdown."""
    normalized_company = " ".join(str(company or "").split()).casefold()
    for raw_line in markdown_text.splitlines():
        match = re.match(r"^#\s+(.+?)\s*$", raw_line.strip())
        if not match:
            continue
        heading = " ".join(match.group(1).split()).strip()
        normalized = heading.casefold().rstrip(":")
        heading_words = set(re.findall(r"[a-z]+", normalized))
        if (
            not heading
            or is_placeholder_job_title(heading)
            or normalized in NON_ROLE_HEADINGS
            or normalized.startswith(("company:", "location:", "source:", "role:"))
            or (normalized_company and normalized == normalized_company)
            or looks_like_internal_slug(heading)
            or not heading_words.intersection(ROLE_HEADING_TERMS)
        ):
            return ""
        return heading
    return ""


def resolve_canonical_job_title(job: dict[str, Any], fallback: str = "Missing job title") -> str:
    """Resolve one canonical title without letting placeholders block Markdown."""
    for key in ("display_role", "title", "role"):
        candidate = display_title_from_value(job.get(key), fallback="")
        if candidate and not is_placeholder_job_title(candidate) and not looks_like_internal_slug(candidate):
            return candidate
    preview = str(job.get("preview", "") or job.get("job_description", "") or "")
    if preview:
        parsed_role = read_markdown_field(preview, "Role", "")
        candidate = display_title_from_value(parsed_role, fallback="")
        if candidate and not is_placeholder_job_title(candidate) and not looks_like_internal_slug(candidate):
            return candidate
        heading = first_role_heading(preview, str(job.get("company", "")))
        if heading:
            return heading
    return fallback


def get_job_display_title(job: dict[str, Any], fallback: str = "Missing job title") -> str:
    """Return the canonical user-facing title for a job."""
    return resolve_canonical_job_title(job, fallback)
