"""Streamlit dashboard for the local job application copilot.

This dashboard wraps the existing local scripts. It does not submit
applications, scrape websites, or expose API credentials.
"""

from __future__ import annotations

import contextlib
import hashlib
import html
import io
import json
import re
import sqlite3
import sys
import zipfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"
DEMO_JOB_DIR = DEMO_DATA_DIR / "jobs"
DEMO_PACKAGE_DIR = DEMO_DATA_DIR / "sample_package"
UI_STATE_DIR = PROJECT_ROOT / "data" / "ui_state"
RECENT_REGIONS_PATH = UI_STATE_DIR / "recent_regions.json"
DEFAULT_FETCH_LIMIT_PER_SOURCE = 20
MAX_FETCH_LIMIT_PER_SOURCE = 20
DEFAULT_RECOMMENDATION_LIMIT = 12
MIN_RECOMMENDATION_LIMIT = 5
MAX_RECOMMENDATION_LIMIT = 30
SHOW_DEBUG_UI = False
DASHBOARD_SCORING_VERSION = "canonical-v1"
INTERNAL_PACKAGE_FILES = {
    "analysis.md",
    "cover_letter.md",
    "cover_letter.docx",
    "resume.docx",
    "tailored_resume.docx",
    "tailored_resume.md",
    "tailoring_notes.md",
    "cover_letter_notes.md",
}
PACKAGE_ZIP_FILE_ORDER = [
    "tailored_resume.docx",
    "resume.docx",
    "tailored_resume.md",
    "cover_letter.docx",
    "cover_letter.md",
    "analysis.md",
    "cover_letter_notes.md",
    "tailoring_notes.md",
]
INTERNAL_NOTES_FILE_ORDER = ["cover_letter_notes.md", "tailoring_notes.md"]
PAGE_NAMES = [
    "Dashboard",
    "Find Jobs",
    "Add Target Job",
    "Review Jobs",
    "Application Package",
    "Tracker",
    "Settings",
]
SCREENING_KEYWORDS = {
    "python": 8,
    "pandas": 8,
    "numpy": 5,
    "scikit-learn": 8,
    "sklearn": 8,
    "machine learning": 12,
    "model evaluation": 8,
    "data visualization": 7,
    "classification": 6,
    "pca": 5,
    "uav": 8,
    "robotics": 5,
    "sensor": 5,
    "thermal": 5,
    "route planning": 5,
    "game ai": 6,
    "reinforcement learning": 5,
    "cnn": 5,
    "econometrics": 5,
    "statistics": 5,
    "communication": 4,
}
HARD_RED_FLAG_PATTERNS = {
    "PhD required": ["phd required", "ph.d. required", "doctorate required"],
    "PhD internship/candidate": ["phd internship", "ph.d. internship", "phd candidate", "ph.d. candidate"],
    "Master's required": ["master's required", "masters required", "m.s. required", "ms required"],
    "Graduate degree required": ["graduate degree required"],
    "Senior/research fellow": ["senior research fellow", "research fellow"],
    "Currently enrolled required": ["currently enrolled", "current student"],
    "Return to school required": ["return to school", "returning to school"],
    "Penultimate year student": ["penultimate year"],
    "3+ years required": ["3+ years", "3 years of experience", "three years of experience"],
    "5+ years required": ["5+ years", "5 years of experience", "five years of experience"],
}
RECOMMENDATION_RANK = {
    "Apply": 5,
    "Apply / Maybe Apply": 4,
    "Maybe Apply": 3,
    "Manual Review": 2,
    "Skip or Low Priority": 1,
    "Skip / Not Eligible": 0,
}
CONFIDENCE_RANK = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
}
REGION_OPTIONS = [
    "Remote",
    "United States",
    "Canada",
    "Australia",
    "Custom",
]
SHORTLIST_REGION_OPTIONS = [
    "all",
    "Remote",
    "United States",
    "Canada",
    "Australia",
]
COMMON_HIGH_LEVEL_REGIONS = ["United States", "Canada", "Australia", "Remote", "Other"]
ADZUNA_SUPPORTED_COUNTRIES = {"sg", "gb", "us", "ca", "au", "nz", "de", "fr", "it", "nl", "pl", "br", "za", "in"}
REGION_CONFIG = {
    "Remote": {
        "adzuna_country": "us",
        "adzuna_location": "Remote",
        "jooble_location": "Remote",
    },
    "United States": {
        "adzuna_country": "us",
        "adzuna_location": "United States",
        "jooble_location": "United States",
    },
    "Canada": {
        "adzuna_country": "ca",
        "adzuna_location": "Canada",
        "jooble_location": "Canada",
    },
    "Australia": {
        "adzuna_country": "au",
        "adzuna_location": "Australia",
        "jooble_location": "Australia",
    },
    "Custom": {
        "adzuna_country": "us",
        "adzuna_location": "",
        "jooble_location": "",
    },
}

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_job import analyze_job_structured  # noqa: E402
from apply_package import create_application_package, parse_job_metadata  # noqa: E402
from company_verification import (  # noqa: E402
    company_verification_fields,
    confirm_markdown_company,
    dedupe_strings,
    normalize_company_name,
    verification_from_markdown,
    verification_status_label,
)
from export_documents import (  # noqa: E402
    export_cover_letter_to_docx,
    export_resume_to_docx,
    parse_job_metadata_from_package,
)
from fetch_history import latest_successful_fetch_run, load_fetch_runs  # noqa: E402
from fetch_jobs import fetch_and_save_jobs  # noqa: E402
import manual_jobs as manual_jobs_module  # noqa: E402
from manual_jobs import (  # noqa: E402
    SOURCE_OPTIONS,
    STATUS_OPTIONS,
    clean_extracted_job_text,
    duplicate_manual_job_exists,
    extract_text_from_upload,
    is_valid_url,
    job_description_quality_warnings,
    load_manual_jobs,
    normalize_job_title,
    parse_job_description_suggestions,
    confirm_manual_job_company,
    save_manual_job,
    update_manual_job,
)
from output_paths import safe_slug, timestamp_slug  # noqa: E402
from output_cleanup import delete_directory_tree  # noqa: E402
from tracker import add_application, delete_application, update_status, VALID_STATUSES  # noqa: E402
from workspace import (  # noqa: E402
    SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS,
    SUPPORTED_EXPERIENCE_BANK_EXTENSIONS,
    SUPPORTED_RESUME_EXTENSIONS,
    Workspace,
    WorkspaceError,
    generic_cover_letter_template,
    initialize_personal_workspace,
    resolve_workspace,
)


MANUAL_FORM_STATE_KEYS = {
    "manual_company",
    "manual_title",
    "manual_location",
    "manual_url",
    "manual_salary_range",
    "manual_visa_note",
    "manual_notes",
    "manual_job_description",
}
MANUAL_TRANSIENT_STATE_KEYS = {
    "manual_pending_suggestions",
    "manual_pending_clean_text",
    "manual_extracted_text",
    "manual_raw_extracted_text",
    "manual_cleaned_extracted_text",
    "manual_source_upload_filenames",
    "manual_parser_suggestions",
    "manual_extraction_reports",
    "manual_generated_summary",
    "manual_generated_backend_output",
    "manual_generated_error",
    "manual_last_extracted_upload_signature",
    "manual_parser_display_mode",
}
MANUAL_SELECTION_STATE_KEY_PREFIXES = (
    "manual_upload_",
    "manual_saved_selected",
    "manual_generate_selected",
    "manual_edit_status_",
    "manual_edit_notes_",
    "manual_generate_",
)


def run_with_captured_output(func: Any, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    """Run a backend helper and capture anything it prints."""
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    output = stdout_buffer.getvalue().strip()
    error_output = stderr_buffer.getvalue().strip()
    combined = "\n".join(part for part in [output, error_output] if part)
    return result, combined


def read_text_file(path: Path) -> str:
    """Read text safely for preview panels."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def clean_job_description_path(path: Path) -> bool:
    """Return True when a file looks like a user-facing job description."""
    if not path.is_file() or path.suffix.lower() != ".md":
        return False
    if any(part == "generated_applications" for part in path.parts):
        return False
    if path.name in INTERNAL_PACKAGE_FILES:
        return False
    return True


def demo_mode_enabled() -> bool:
    """Return True when the UI should use bundled sanitized demo data."""
    return current_workspace().mode == "demo"


def current_workspace() -> Workspace:
    """Resolve the selected workspace, defaulting every fresh session to Demo."""
    return resolve_workspace(str(st.session_state.get("workspace_mode", "Demo")))


def go_to_page(page_name: str) -> None:
    """Navigate to one of the top-level dashboard pages."""
    if page_name not in PAGE_NAMES:
        raise ValueError(f"Unknown page: {page_name}")
    st.session_state["active_page"] = page_name
    st.rerun()


def list_job_description_files(search_text: str = "") -> list[Path]:
    """Return saved job description Markdown files recursively."""
    query = search_text.strip().lower()
    if demo_mode_enabled():
        if not DEMO_JOB_DIR.exists():
            return []
        demo_paths = [path for path in DEMO_JOB_DIR.glob("*.md") if clean_job_description_path(path)]
        return sorted(
            [path for path in demo_paths if not query or query in str(path).lower()],
            key=lambda path: str(path).lower(),
        )

    search_roots = [current_workspace().jobs_dir]
    paths = []
    for root in search_roots:
        if not root.exists():
            continue
        paths.extend(
            path
            for path in root.rglob("*.md")
            if clean_job_description_path(path)
            and (not query or query in str(path).lower())
        )
    return sorted(paths, key=lambda path: str(path).lower())


def relative_path(path: Path) -> str:
    """Show project-relative paths when possible."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def relocate_fetched_jobs_to_workspace(paths: list[object], source: str) -> list[Path]:
    """Move newly fetched job files into the selected Personal workspace."""
    workspace = current_workspace()
    workspace.require_writable()
    destination_dir = (workspace.jobs_dir / safe_slug(source)).resolve()
    destination_dir.mkdir(parents=True, exist_ok=True)
    relocated = []
    for raw_path in paths:
        source_path = Path(str(raw_path)).resolve()
        if not source_path.is_file():
            continue
        if source_path.is_relative_to(workspace.jobs_dir.resolve()):
            relocated.append(source_path)
            continue
        destination = destination_dir / source_path.name
        if destination.exists():
            destination = destination_dir / f"{safe_slug(source_path.stem)}_{timestamp_slug()}{source_path.suffix.lower()}"
        source_path.replace(destination)
        relocated.append(destination)
    return relocated


def normalize_text(text: str) -> str:
    """Normalize text for dashboard keyword matching."""
    return " " + " ".join(text.lower().replace("-", " ").split()) + " "


def read_markdown_field(markdown_text: str, field_name: str, default: str = "") -> str:
    """Read a simple 'Field: value' line from Markdown."""
    prefix = f"{field_name}:"
    for line in markdown_text.splitlines():
        if line.lower().startswith(prefix.lower()):
            value = line.split(":", 1)[1].strip()
            if value and value.lower() != "not provided":
                return value
    return default


def company_generation_allowed(fields: dict[str, Any]) -> bool:
    """Return True when cover-letter generation may use this company name."""
    company = str(fields.get("company_normalized", "") or fields.get("normalized_company", "")).strip()
    confidence = str(fields.get("company_confidence", "") or fields.get("confidence", "")).lower()
    confirmed = bool(fields.get("company_confirmed_by_user") or fields.get("confirmed_by_user"))
    needs_review = bool(fields.get("company_needs_review", fields.get("needs_review", True)))
    return bool(company) and (confidence == "high" or confirmed) and not needs_review


def compact_company_evidence(fields: dict[str, Any]) -> str:
    """Format short company verification evidence for captions."""
    evidence = fields.get("company_evidence", fields.get("evidence", [])) or []
    if isinstance(evidence, str):
        evidence = [item.strip() for item in evidence.split("|") if item.strip()]
    return " | ".join(str(item) for item in evidence[:2]) or "-"


def company_candidate_names(fields: dict[str, Any]) -> list[str]:
    """Return normalized candidate names for selectors."""
    candidates = fields.get("company_candidates", fields.get("candidates", [])) or []
    names = []
    for item in candidates:
        if isinstance(item, dict):
            name = str(item.get("normalized_company", "") or item.get("company", "")).strip()
        else:
            name = str(item).strip()
        if name:
            names.append(name)
    current = str(fields.get("company_normalized", "") or fields.get("normalized_company", "") or fields.get("company_raw", "")).strip()
    if current:
        names.insert(0, current)
    return dedupe_strings(names)


def render_company_verification_summary(fields: dict[str, Any]) -> None:
    """Show compact verification status and supporting evidence."""
    status = verification_status_label(fields)
    company = str(fields.get("company_normalized", "") or fields.get("normalized_company", "") or fields.get("company_raw", "") or "-")
    st.write(f"Company: {company}")
    st.write(f"Verification: {status}")
    if status in {"Needs review", "Missing"}:
        st.warning("Company name needs confirmation before generating a cover letter.")
    evidence_text = compact_company_evidence(fields)
    if evidence_text != "-":
        st.caption(f"Evidence: {evidence_text}")


def render_markdown_company_confirmation(path: Path, key_prefix: str) -> dict[str, Any]:
    """Render editable company confirmation controls for a Markdown job file."""
    fields = verification_from_markdown(path)
    render_company_verification_summary(fields)
    candidates = company_candidate_names(fields)
    if candidates:
        selected = st.selectbox(
            "Suggested company candidates",
            candidates,
            key=f"{key_prefix}_company_candidate",
        )
    else:
        selected = str(fields.get("company_raw", "") or "")
    edited_company = st.text_input(
        "Editable company name",
        value=selected,
        key=f"{key_prefix}_company_confirm_input",
    )
    if st.button("Confirm company name", key=f"{key_prefix}_company_confirm_button"):
        normalized = normalize_company_name(edited_company)
        if not normalized:
            st.error("Enter a valid company name before confirming.")
        else:
            confirm_markdown_company(path, normalized)
            st.success("Company name confirmed.")
            st.rerun()
    return fields


def render_manual_company_confirmation(record: dict[str, Any], key_prefix: str) -> dict[str, Any]:
    """Render editable company confirmation controls for a saved manual record."""
    fields = company_verification_fields(
        str(record.get("company_raw") or record.get("company", "")),
        {
            "job_text": str(record.get("job_description", "") or ""),
            "role": str(record.get("title", "") or ""),
            "location": str(record.get("location", "") or ""),
            "job_url": str(record.get("url", "") or ""),
            "company_confirmed_by_user": bool(record.get("company_confirmed_by_user")),
            "company_source_confidence": str(record.get("company_confidence", "") or ""),
            "company_source_evidence": compact_company_evidence(record),
        },
        confirmed_by_user=bool(record.get("company_confirmed_by_user")),
        confirmed_at=str(record.get("company_confirmed_at", "") or ""),
    )
    fields.update({key: record.get(key) for key in record if key.startswith("company_")})
    render_company_verification_summary(fields)
    candidates = company_candidate_names(fields)
    selected = st.selectbox(
        "Suggested company candidates",
        candidates or [str(record.get("company", "") or "")],
        key=f"{key_prefix}_company_candidate",
    )
    edited_company = st.text_input(
        "Editable company name",
        value=selected,
        key=f"{key_prefix}_company_confirm_input",
    )
    if st.button("Confirm company name", key=f"{key_prefix}_company_confirm_button"):
        normalized = normalize_company_name(edited_company)
        if not normalized:
            st.error("Enter a valid company name before confirming.")
        else:
            updated = confirm_manual_job_company(str(record["id"]), normalized)
            if updated:
                st.success("Company name confirmed.")
                st.rerun()
            else:
                st.error("Could not update that target job record.")
    return fields


def path_contains_slug(path: Path, slug: str) -> bool:
    """Return True when a normalized path segment contains a location slug."""
    normalized_parts = [part.lower().replace("-", "_") for part in path.parts]
    return slug in normalized_parts


def infer_location_from_path(path: Path) -> str:
    """Infer a readable location for older files without Location metadata."""
    if path_contains_slug(path, "london"):
        return "London"
    if path_contains_slug(path, "singapore") or path_contains_slug(path, "sg"):
        return "Singapore"
    if path_contains_slug(path, "united_kingdom") or path_contains_slug(path, "uk") or path_contains_slug(path, "gb"):
        return "United Kingdom"
    if path_contains_slug(path, "remote"):
        return "Remote"
    if path_contains_slug(path, "tokyo"):
        return "Tokyo"
    if path_contains_slug(path, "shenzhen"):
        return "Shenzhen"
    if path_contains_slug(path, "hangzhou"):
        return "Hangzhou"
    return "Unknown"


def normalize_location(location: str) -> str:
    """Clean display/filter locations from metadata, paths, and OCR-derived text."""
    value = " ".join(str(location or "").replace("\n", " ").split())
    value = re.sub(r"\.{2,}$", "", value).strip()
    value = value.strip(" ,;:|·•-–—")
    lower_value = value.lower()
    aliases = {
        "uk": "United Kingdom",
        "u.k.": "United Kingdom",
        "gb": "United Kingdom",
        "great britain": "United Kingdom",
        "usa": "United States",
        "u.s.": "United States",
        "us": "United States",
        "united states of america": "United States",
    }
    return aliases.get(lower_value, value)


def infer_high_level_region(location: str) -> str:
    """Map a normalized location to a broad region used by dashboard filters."""
    normalized_location = normalize_location(location)
    normalized = normalize_text(normalized_location)
    if " remote " in normalized:
        return "Remote"
    if any(marker in normalized for marker in [" china ", " beijing ", " shanghai ", " shenzhen ", " hangzhou "]):
        return "China"
    if " singapore " in normalized:
        return "Singapore"
    if any(marker in normalized for marker in [" united kingdom ", " london ", " england ", " scotland ", " wales "]):
        return "United Kingdom"
    if any(
        marker in normalized
        for marker in [
            " united states ",
            " usa ",
            " california ",
            " ca ",
            " new york ",
            " washington ",
            " boston ",
            " seattle ",
        ]
    ):
        return "United States"
    return "Other"


def region_option_key(option_type: str, value: str) -> str:
    """Build a stable key for a region option."""
    if option_type == "all":
        return "all"
    return f"{option_type}:{safe_slug(value)}"


def region_label(option: dict[str, Any], include_count: bool = True) -> str:
    """Build display text while keeping option keys stable internally."""
    if option["type"] == "all":
        base = "all"
    elif option["type"] == "high_level":
        base = f"High-level: {option['value']}"
    else:
        base = f"Exact: {option['value']}"
    return f"{base} ({option['count']})" if include_count else base


def region_search_blob(option: dict[str, Any]) -> str:
    """Return searchable text including aliases such as UK."""
    aliases = []
    value = str(option.get("value", ""))
    high_level_value = value if option.get("type") == "high_level" else infer_high_level_region(value)
    if high_level_value == "United Kingdom":
        aliases.extend(["uk", "u.k.", "gb", "great britain", "london"])
    if high_level_value == "United States":
        aliases.extend(["us", "u.s.", "usa", "america"])
    if high_level_value == "China":
        aliases.extend(["cn", "beijing", "shanghai", "shenzhen", "hangzhou"])
    return " ".join([region_label(option, include_count=False), value, *aliases]).lower()


def load_recent_region_keys() -> list[str]:
    """Read recent region keys; missing or malformed state is treated as empty."""
    try:
        data = json.loads(RECENT_REGIONS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return []
    return [str(item) for item in data if isinstance(item, str)][:10] if isinstance(data, list) else []


def save_recent_region_key(region_key: str) -> None:
    """Persist a small MRU list for the compact default region dropdown."""
    if demo_mode_enabled() or not region_key or region_key == "all":
        return
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    recent_keys = [key for key in load_recent_region_keys() if key != region_key]
    recent_keys.insert(0, region_key)
    RECENT_REGIONS_PATH.write_text(json.dumps(recent_keys[:10], indent=2), encoding="utf-8")


def build_region_options(jobs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build all searchable region options from currently loaded jobs."""
    total_count = len(jobs)
    options: dict[str, dict[str, Any]] = {
        "all": {"key": "all", "label": "all", "type": "all", "value": "all", "count": total_count}
    }

    for job in jobs:
        normalized_location = normalize_location(str(job.get("location", "")))
        if not normalized_location or normalized_location.lower() in {"unknown", "not provided"}:
            normalized_location = "Other"
        high_level = infer_high_level_region(normalized_location)

        for option_type, value in [("high_level", high_level), ("exact", normalized_location)]:
            key = region_option_key(option_type, value)
            if key not in options:
                options[key] = {
                    "key": key,
                    "label": "",
                    "type": option_type,
                    "value": value,
                    "count": 0,
                }
            options[key]["count"] += 1

    for option in options.values():
        option["label"] = region_label(option)
    return options


def default_region_option_keys(options_by_key: dict[str, dict[str, Any]]) -> list[str]:
    """Return the short default region list before the user searches."""
    keys = ["all"]
    valid_recent_keys = [key for key in load_recent_region_keys() if key in options_by_key]
    keys.extend(valid_recent_keys[:5])

    for region in COMMON_HIGH_LEVEL_REGIONS:
        key = region_option_key("high_level", region)
        if key in options_by_key:
            keys.append(key)

    exact_options = [
        option
        for option in options_by_key.values()
        if option["type"] == "exact" and option["value"] not in {"Other", "Unknown", "Not provided"}
    ]
    exact_options.sort(key=lambda option: (-int(option["count"]), str(option["value"]).lower()))
    keys.extend(option["key"] for option in exact_options[:5])

    deduped_keys = []
    for key in keys:
        if key in options_by_key and key not in deduped_keys:
            deduped_keys.append(key)
    return deduped_keys


def filtered_region_option_keys(options_by_key: dict[str, dict[str, Any]], query: str) -> list[str]:
    """Search all region options by label, value, and common aliases."""
    cleaned_query = " ".join(query.lower().split())
    if not cleaned_query:
        return default_region_option_keys(options_by_key)

    matches = [
        option
        for option in options_by_key.values()
        if option["key"] == "all" or cleaned_query in region_search_blob(option)
    ]
    matches.sort(key=lambda option: (option["key"] != "all", option["type"] != "high_level", -int(option["count"])))
    return [option["key"] for option in matches] or ["all"]


def job_matches_region_option(job: dict[str, Any], selected_option: dict[str, Any]) -> bool:
    """Filter jobs by exact normalized location or inferred high-level region."""
    if selected_option["type"] == "all":
        return True
    if selected_option["type"] == "exact":
        return normalize_location(str(job.get("location", ""))).lower() == str(selected_option["value"]).lower()
    return str(job.get("high_level_region", "")) == selected_option["value"]


def source_display_name(source: str) -> str:
    """Normalize source labels for dynamic source filtering."""
    cleaned = " ".join(str(source or "unknown").replace("_", " ").split()).strip()
    lower_cleaned = cleaned.lower()
    aliases = {
        "company website": "Company Website",
        "manual": "Manual",
        "linkedin": "LinkedIn",
        "indeed": "Indeed",
        "handshake": "Handshake",
        "jooble": "Jooble",
        "adzuna": "Adzuna",
    }
    return aliases.get(lower_cleaned, cleaned.title() if cleaned else "Unknown")


INTERNAL_TITLE_FALLBACKS = {
    "example_ml_job": "Machine Learning Intern",
    "sample_package": "Sample Application Package",
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


def looks_like_internal_slug(value: str) -> bool:
    """Return True for internal file/package slugs that should not be user-facing."""
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
    """Convert a stored title/role value into a safe user-facing label."""
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


def dynamic_source_options(jobs: list[dict[str, Any]]) -> list[str]:
    """Build source filter options from loaded jobs."""
    discovered = sorted({source_display_name(str(job.get("source", ""))) for job in jobs})
    preferred = ["LinkedIn", "Jooble", "Adzuna", "Company Website", "Indeed", "Handshake", "Manual"]
    ordered = ["all"]
    for source in preferred + discovered:
        if source in discovered and source not in ordered:
            ordered.append(source)
    return ordered


def detect_dashboard_red_flags(job_text: str) -> list[str]:
    """Detect requirements that should be downgraded or hidden by default."""
    normalized = normalize_text(job_text)
    red_flags = []
    for label, patterns in HARD_RED_FLAG_PATTERNS.items():
        if any(normalize_text(pattern).strip() in normalized for pattern in patterns):
            red_flags.append(label)
    return red_flags


def score_job_for_dashboard(job_text: str) -> int:
    """Calculate a lightweight dashboard score without writing analysis files."""
    normalized = normalize_text(job_text)
    score = 45
    for keyword, points in SCREENING_KEYWORDS.items():
        if normalize_text(keyword).strip() in normalized:
            score += points

    for red_flag in detect_dashboard_red_flags(job_text):
        if "5+" in red_flag or "PhD" in red_flag or "Senior" in red_flag:
            score -= 25
        else:
            score -= 15

    return max(0, min(100, score))


def recommendation_for_score(score: int, red_flags: list[str]) -> str:
    """Convert dashboard score and red flags into a recommendation label."""
    if red_flags and score >= 65:
        return "Maybe Apply"
    if score >= 80:
        return "Apply"
    if score >= 50:
        return "Maybe Apply"
    return "Skip or Low Priority"


def confidence_for_job(job_text: str, score: int) -> str:
    """Estimate confidence based on available JD detail."""
    word_count = len(job_text.split())
    if word_count >= 250 and score >= 70:
        return "High"
    if word_count >= 120:
        return "Medium"
    return "Low"


def warnings_for_job(job_text: str) -> list[str]:
    """Return dashboard review warnings for incomplete or high-risk postings."""
    warnings = []
    if len(job_text.split()) < 120:
        warnings.append("API description may be an incomplete snippet")
    for phrase in ["work authorization", "visa", "citizenship", "sponsorship"]:
        if phrase in job_text.lower():
            warnings.append(f"Review {phrase} requirement manually")
    if is_uk_job_text(job_text):
        warnings.append(
            "UK HPI review: user may be eligible to apply, but should not claim current UK work authorization."
        )
        if asks_for_uk_work_authorization(job_text):
            warnings.append("Confirm whether the employer accepts candidates planning to use the HPI visa route")
    return warnings


def is_uk_job_text(job_text: str) -> bool:
    """Return True when a saved job appears to be UK/London based."""
    normalized = normalize_text(job_text)
    return any(
        phrase in normalized
        for phrase in [
            " london ",
            " united kingdom ",
            " uk ",
            " great britain ",
            " adzuna.co.uk ",
            " adzuna.gb ",
        ]
    )


def asks_for_uk_work_authorization(job_text: str) -> bool:
    """Detect UK work authorization or sponsorship language for review."""
    normalized = normalize_text(job_text)
    phrases = [
        "right to work in the uk",
        "uk work authorization",
        "uk work authorisation",
        "visa sponsorship",
        "sponsorship required",
        "skilled worker sponsorship",
    ]
    return any(phrase in normalized for phrase in phrases)


def unavailable_dashboard_analysis(reason: str) -> dict[str, Any]:
    """Return a safe current-analysis result without promoting a legacy score."""
    return {
        "score": None,
        "recommendation": "Manual Review",
        "score_breakdown": [],
        "eligibility": {"status": "manual_review", "reasons": []},
        "confidence": {
            "level": "low",
            "active_requirement_count": 0,
            "candidate_evidence_count": 0,
            "reasons": [reason],
        },
        "candidate_profile": {"career_level": "unknown", "years_experience": None, "highest_degree": "unknown", "evidence": []},
        "parsed_job": {"required_skills": [], "preferred_skills": [], "experience_level": []},
        "matched_skills": [],
        "partial_matches": [],
        "missing_skills": [],
        "main_reason": reason,
        "main_risk": "Review the full job description manually.",
        "analysis_available": False,
    }


def analyze_job_for_dashboard(
    job: dict[str, Any],
    job_text: str,
    candidate_text: str | None = None,
    *,
    use_cache: bool | None = None,
) -> dict[str, Any]:
    """Return the canonical full analysis for one loaded dashboard job."""
    from streamlit.runtime.scriptrunner import get_script_run_ctx

    if use_cache is None:
        use_cache = get_script_run_ctx(suppress_warning=True) is not None
    if not job_text.strip():
        return unavailable_dashboard_analysis("Job description is empty or unreadable.")
    if candidate_text is None:
        candidate_path = current_workspace().resume_source_path
        if candidate_path is None or not candidate_path.is_file():
            return unavailable_dashboard_analysis("Candidate source is missing or unreadable.")
        candidate_text = read_text_file(candidate_path)
    if not candidate_text.strip():
        return unavailable_dashboard_analysis("Candidate source is missing or empty.")

    workspace = current_workspace() if use_cache else None
    cache_material = "\0".join(
        [
            DASHBOARD_SCORING_VERSION,
            workspace.mode if workspace else "provided",
            str(workspace.root) if workspace else "provided",
            str(job.get("canonical_job_key", "") or job.get("path", "")),
            hashlib.sha256(job_text.encode("utf-8")).hexdigest(),
            hashlib.sha256(candidate_text.encode("utf-8")).hexdigest(),
        ]
    )
    cache_key = hashlib.sha256(cache_material.encode("utf-8")).hexdigest()
    cache = st.session_state.setdefault("dashboard_analysis_cache", {}) if use_cache else {}
    if cache_key in cache:
        return dict(cache[cache_key])

    try:
        analysis = dict(analyze_job_structured(job_text, candidate_text))
    except (OSError, ValueError, TypeError) as error:
        return unavailable_dashboard_analysis(f"Current analysis could not run: {error}")

    confidence = dict(analysis.get("confidence", {}))
    if int(confidence.get("active_requirement_count", 0) or 0) == 0:
        return unavailable_dashboard_analysis("Structured requirements could not be extracted reliably.")
    analysis["analysis_available"] = True
    if use_cache:
        cache[cache_key] = dict(analysis)
    return analysis


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
        partial_map = {
            str(value).split(" (", 1)[0]: str(value)
            for value in category.get("partial", [])
        }
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
    if isinstance(value, dict):
        return str(value.get("level", "low")).lower()
    return str(value or "low").lower()


def eligibility_status(job: dict[str, Any]) -> str:
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
    elif level == "low":
        role_fit = "Insufficient evidence"
        card_status = "Insufficient evidence · Manual Review · Low confidence"
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
        "coverage_score": score if available else None,
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


def build_dashboard_job_record(path: Path) -> dict[str, Any]:
    """Build one ranked dashboard row from a saved job Markdown file."""
    job_text = read_text_file(path)
    company_fields = verification_from_markdown(path)
    company = str(company_fields.get("company_normalized") or read_markdown_field(job_text, "Company", "Not provided"))
    stored_role = read_markdown_field(job_text, "Role", path.stem)
    role = resolve_canonical_job_title(
        {"company": company, "role": stored_role, "preview": job_text}
    )
    display_role = role
    location = normalize_location(read_markdown_field(job_text, "Location", infer_location_from_path(path)))
    high_level_region = infer_high_level_region(location)
    source = read_markdown_field(job_text, "Source", infer_source_from_path(path))
    job_url = read_markdown_field(job_text, "Job URL", "")
    first_seen_at = read_markdown_field(job_text, "First Seen At", read_markdown_field(job_text, "Created at", "unknown"))
    last_seen_at = read_markdown_field(job_text, "Last Seen At", first_seen_at)
    latest_fetch_run_id = read_markdown_field(job_text, "Latest Fetch Run ID", read_markdown_field(job_text, "Last Seen Fetch Run ID", ""))
    first_seen_fetch_run_id = read_markdown_field(job_text, "First Seen Fetch Run ID", "")
    canonical_job_key = read_markdown_field(job_text, "Canonical Job Key", "")
    red_flags = detect_dashboard_red_flags(job_text)
    score_text = read_markdown_field(job_text, "Match Score", "")
    legacy_score = int(score_text) if score_text.isdigit() else None
    legacy_recommendation = read_markdown_field(job_text, "Recommendation", "")
    warnings = warnings_for_job(job_text)
    hard_red_flag = bool(red_flags)

    return {
        "company": company,
        "company_raw": company_fields.get("company_raw", ""),
        "company_normalized": company_fields.get("company_normalized", company),
        "company_confidence": company_fields.get("company_confidence", ""),
        "company_needs_review": company_fields.get("company_needs_review", True),
        "company_evidence": company_fields.get("company_evidence", []),
        "company_candidates": company_fields.get("company_candidates", []),
        "company_confirmed_by_user": company_fields.get("company_confirmed_by_user", False),
        "company_confirmed_at": company_fields.get("company_confirmed_at", ""),
        "company_status": verification_status_label(company_fields),
        "role": role,
        "display_role": display_role,
        "location": location,
        "normalized_location": location,
        "high_level_region": high_level_region,
        "source": source.lower(),
        "score": None,
        "recommendation": "Manual Review",
        "confidence": {"level": "low", "active_requirement_count": 0, "candidate_evidence_count": 0, "reasons": []},
        "eligibility": {"status": "manual_review", "reasons": []},
        "score_breakdown": [],
        "analysis_result": {},
        "analysis_available": False,
        "legacy_score": legacy_score,
        "legacy_recommendation": legacy_recommendation,
        "red_flags": red_flags,
        "warnings": warnings,
        "hard_red_flag": hard_red_flag,
        "job_url": job_url,
        "canonical_job_key": canonical_job_key,
        "first_seen_at": first_seen_at,
        "last_seen_at": last_seen_at,
        "first_seen_fetch_run_id": first_seen_fetch_run_id,
        "latest_fetch_run_id": latest_fetch_run_id,
        "is_manual": "manual_jobs" in path.parts,
        "path": path,
        "preview": job_text[:1200],
        "label": f"{company} | {display_role} | {location}",
    }


def job_duplicate_key(job: dict[str, Any]) -> tuple[str, str, str, str]:
    """Return URL key plus company/role/location fallback fields."""
    return (
        str(job["job_url"]).strip().lower(),
        str(job["company"]).strip().lower(),
        str(job["role"]).strip().lower(),
        str(job["location"]).strip().lower(),
    )


def deduplicate_dashboard_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Remove duplicate saved job descriptions before ranking."""
    seen_urls: set[str] = set()
    seen_fallbacks: set[tuple[str, str, str]] = set()
    unique_jobs = []

    for job in jobs:
        job_url, company, role, location = job_duplicate_key(job)
        fallback = (company, role, location)

        if job_url and job_url in seen_urls:
            continue
        if all(fallback) and fallback in seen_fallbacks:
            continue

        unique_jobs.append(job)
        if job_url:
            seen_urls.add(job_url)
        if all(fallback):
            seen_fallbacks.add(fallback)

    return unique_jobs


def dashboard_rank_key(job: dict[str, Any]) -> tuple[int, int, int, int]:
    """Sort by recommendation, score, confidence, then fewer red flags."""
    return (
        RECOMMENDATION_RANK.get(job["recommendation"], 0),
        int(job.get("score") or 0),
        CONFIDENCE_RANK.get(confidence_level(job.get("confidence")).title(), 0),
        -len(job["red_flags"]),
    )


def infer_source_from_path(path: Path) -> str:
    """Infer source from organized job description paths."""
    parts = list(path.parts)
    if "manual_jobs" in parts:
        return "manual"
    if "job_descriptions" in parts:
        index = parts.index("job_descriptions")
        if len(parts) > index + 2:
            return parts[index + 1]
    return "unknown"


def load_screened_jobs(search_text: str = "") -> list[dict[str, Any]]:
    """Load jobs and attach canonical full-analysis results in memory."""
    _ = search_text
    records = [build_dashboard_job_record(path) for path in list_job_description_files()]
    unique_records = deduplicate_dashboard_jobs(records)
    candidate_path = current_workspace().resume_source_path
    candidate_text = read_text_file(candidate_path) if candidate_path and candidate_path.is_file() else ""
    analyzed_records = []
    for job in unique_records:
        job_text = read_text_file(Path(job["path"]))
        analysis = analyze_job_for_dashboard(job, job_text, candidate_text)
        analyzed = apply_canonical_analysis(job, analysis)
        presentation = build_fit_presentation(analyzed)
        analyzed["label"] = (
            f"{analyzed['company']} | {get_job_display_title(analyzed)} | "
            f"{analyzed['location']} | {presentation['role_fit']}"
        )
        analyzed_records.append(analyzed)
    unique_records = analyzed_records
    unique_records.sort(key=dashboard_rank_key, reverse=True)
    return unique_records


def load_tracker_rows(
    statuses: list[str] | None = None,
    minimum_score: int = 0,
    company_search: str = "",
    sort_by: str = "created_at",
    descending: bool = True,
) -> list[dict[str, Any]]:
    """Load tracker rows with simple local filtering and sorting."""
    database_path = current_workspace().tracker_database_path
    if database_path is None or not database_path.exists():
        return []

    order_map = {
        "match_score": "COALESCE(match_score, -1)",
        "created_at": "created_at",
        "status": "status",
    }
    order_column = order_map.get(sort_by, "created_at")
    order_direction = "DESC" if descending else "ASC"

    conditions = []
    params: list[Any] = []

    if statuses:
        placeholders = ", ".join("?" for _ in statuses)
        conditions.append(f"status IN ({placeholders})")
        params.extend(statuses)

    if minimum_score > 0:
        conditions.append("COALESCE(match_score, 0) >= ?")
        params.append(minimum_score)

    if company_search.strip():
        conditions.append("LOWER(company) LIKE ?")
        params.append(f"%{company_search.strip().lower()}%")

    where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
    query = f"""
        SELECT
            id,
            company,
            role,
            location,
            job_url,
            match_score,
            recommendation,
            status,
            resume_file,
            cover_letter_file,
            notes,
            created_at,
            applied_date
        FROM applications
        {where_clause}
        ORDER BY {order_column} {order_direction}, id DESC
    """

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        rows = connection.execute(query, params).fetchall()

    return [dict(row) for row in rows]


def load_tracker_record(application_id: int) -> dict[str, Any] | None:
    """Load one tracker record by id."""
    database_path = current_workspace().tracker_database_path
    if database_path is None or not database_path.exists():
        return None

    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row
        row = connection.execute(
            """
            SELECT
                id,
                company,
                role,
                location,
                job_url,
                match_score,
                recommendation,
                status,
                resume_file,
                cover_letter_file,
                notes,
                created_at,
                applied_date
            FROM applications
            WHERE id = ?
            """,
            (application_id,),
        ).fetchone()

    return dict(row) if row else None


def resolve_package_dir_from_tracker(row: dict[str, Any]) -> Path | None:
    """Infer the package folder from tracker file paths."""
    for key in ["cover_letter_file", "resume_file"]:
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        if candidate.exists() and candidate.is_relative_to(current_workspace().generated_dir.resolve()):
            return candidate.parent
    return None


def latest_package_for_company_role(company: str, role: str) -> Path | None:
    """Find the latest package folder when tracker file paths are unavailable."""
    family = safe_slug(f"{company}_{role}")
    family_dir = current_workspace().generated_dir / family
    if not family_dir.exists():
        return None

    timestamp_dirs = [path for path in family_dir.iterdir() if path.is_dir()]
    if not timestamp_dirs:
        return None
    return sorted(timestamp_dirs, key=lambda path: path.name, reverse=True)[0]


def count_generated_packages() -> int:
    """Count generated application package folders without opening user content."""
    if demo_mode_enabled():
        return 1 if DEMO_PACKAGE_DIR.exists() else 0
    generated_dir = current_workspace().generated_dir
    if not generated_dir.exists():
        return 0
    return sum(
        1
        for path in generated_dir.rglob("*")
        if path.is_dir()
        and any(
            (path / filename).exists()
            for filename in [
                "analysis.md",
                "tailored_resume.md",
                "tailored_resume.docx",
                "cover_letter.md",
                "cover_letter.docx",
            ]
        )
    )


def tracker_args_for_job(job: dict[str, Any]) -> SimpleNamespace:
    """Prepare canonical current-analysis values for a new tracker row."""
    eligibility = eligibility_status(job).replace("_", " ").title()
    confidence = confidence_level(job.get("confidence")).title()
    return SimpleNamespace(
        company=str(job.get("company", "")).strip() or "Unknown company",
        role=resolve_canonical_job_title(job),
        location=str(job.get("normalized_location", "") or job.get("location", "")).strip(),
        job_url=str(job.get("job_url", "")).strip(),
        match_score=int(job["score"]) if job.get("analysis_available") and job.get("score") is not None else None,
        recommendation=str(job.get("recommendation", "Manual Review")).strip(),
        status="saved",
        resume_file="",
        cover_letter_file="",
        notes=(
            f"Saved from Review Jobs. Eligibility: {eligibility}. "
            f"Scoring confidence: {confidence}. No application was submitted."
        ),
    )


def save_job_to_tracker(job: dict[str, Any]) -> tuple[int | None, str]:
    """Save a reviewed job lead to the local tracker without generating documents."""
    args = tracker_args_for_job(job)
    workspace = current_workspace()
    workspace.require_writable()
    assert workspace.tracker_database_path is not None
    return run_with_captured_output(add_application, args, workspace.tracker_database_path)


def tracker_row_for_job(job: dict[str, Any], tracker_rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Find a tracker row for a job using URL first, then company/role/location."""
    job_url = str(job.get("job_url", "") or "").strip().lower()
    company = str(job.get("company", "") or "").strip().lower()
    role = resolve_canonical_job_title(job).lower()
    location = str(job.get("normalized_location", "") or job.get("location", "") or "").strip().lower()

    if job_url:
        for row in tracker_rows:
            if str(row.get("job_url", "") or "").strip().lower() == job_url:
                return row

    for row in tracker_rows:
        row_key = (
            str(row.get("company", "") or "").strip().lower(),
            str(row.get("role", "") or "").strip().lower(),
            str(row.get("location", "") or "").strip().lower(),
        )
        if row_key == (company, role, location):
            return row
    return None


def tracker_status_for_job(job: dict[str, Any], tracker_rows: list[dict[str, Any]]) -> str:
    """Return the current tracker status for display on job cards."""
    if demo_mode_enabled():
        return "Demo only"
    row = tracker_row_for_job(job, tracker_rows)
    return str(row.get("status", "") or "Not tracked") if row else "Not tracked"


def package_dir_for_job(job: dict[str, Any], tracker_rows: list[dict[str, Any]]) -> Path | None:
    """Find a generated package for display without changing package behavior."""
    row = tracker_row_for_job(job, tracker_rows)
    if row:
        package_dir = resolve_package_dir_from_tracker(row)
        if package_dir:
            return package_dir
    return latest_package_for_company_role(str(job.get("company", "")), resolve_canonical_job_title(job))


def package_status_for_job(job: dict[str, Any], tracker_rows: list[dict[str, Any]]) -> str:
    """Return a compact package status for Review Jobs display."""
    if demo_mode_enabled():
        report_path = DEMO_PACKAGE_DIR / "analysis.md"
        if not report_path.is_file():
            return "Demo only"
        match = re.search(r"Job description file: `(.+?)`", read_text_file(report_path))
        if not match:
            return "Demo only"
        bundled_job_path = Path(match.group(1))
        if not bundled_job_path.is_absolute():
            bundled_job_path = PROJECT_ROOT / bundled_job_path
        job_path = Path(str(job.get("path", "")))
        if not job_path.is_absolute():
            job_path = PROJECT_ROOT / job_path
        return "Demo package" if bundled_job_path.resolve() == job_path.resolve() else "Demo only"
    return "Package ready" if package_dir_for_job(job, tracker_rows) else "No package"


def is_ignored_tracker_status(status: str) -> bool:
    """Return True for tracker statuses that mean the job is intentionally ignored."""
    return str(status or "").strip().lower() in {"archived", "ignored", "not interested", "rejected", "skip"}


def review_inbox_view_matches(
    job: dict[str, Any],
    inbox_view: str,
    tracker_status: str,
    package_status: str,
) -> bool:
    """Map Review Jobs inbox views to existing job, tracker, and package state."""
    recommendation = str(job.get("recommendation", ""))
    score = int(job.get("score") or 0)
    confidence = confidence_level(job.get("confidence"))
    eligibility = eligibility_status(job)
    has_package = package_status in {"Package ready", "Demo package"}
    is_tracked = tracker_status not in {"Not tracked", "Demo only"}
    is_ignored = is_ignored_tracker_status(tracker_status)

    if inbox_view == "Recommended":
        return (
            bool(job.get("analysis_available"))
            and eligibility == "passed"
            and confidence in {"medium", "high"}
            and recommendation in {"Apply", "Apply / Maybe Apply", "Maybe Apply"}
            and score >= 50
            and not is_ignored
        )
    if inbox_view == "Needs Review":
        canonical_review_needed = (
            not bool(job.get("analysis_available"))
            or eligibility == "manual_review"
            or confidence == "low"
            or recommendation == "Manual Review"
        )
        operational_review_needed = tracker_status != "Demo only" and (not has_package or not is_tracked)
        return not is_ignored and (canonical_review_needed or operational_review_needed)
    if inbox_view == "Package Ready":
        return has_package and not is_ignored
    if inbox_view == "Not Tracked":
        return not is_ignored and not is_tracked
    if inbox_view == "Ignored":
        return is_ignored
    return True


def default_review_inbox_view(jobs: list[dict[str, Any]], tracker_rows: list[dict[str, Any]], *, demo: bool) -> str:
    """Choose the initial Review Jobs view without changing filtering rules."""
    if demo:
        return "All Jobs"
    if any(
        review_inbox_view_matches(
            job,
            "Recommended",
            tracker_status_for_job(job, tracker_rows),
            package_status_for_job(job, tracker_rows),
        )
        for job in jobs
    ):
        return "Recommended"
    if any(
        review_inbox_view_matches(
            job,
            "Needs Review",
            tracker_status_for_job(job, tracker_rows),
            package_status_for_job(job, tracker_rows),
        )
        for job in jobs
    ):
        return "Needs Review"
    return "All Jobs"


def review_job_sort_key(job: dict[str, Any], sort_by: str) -> tuple[Any, ...]:
    """Sort Review Jobs inbox rows by the selected user-facing option."""
    score = int(job.get("score", 0) or 0)
    newest = str(job.get("last_seen_at", "") or job.get("first_seen_at", ""))
    recommendation_rank = RECOMMENDATION_RANK.get(str(job.get("recommendation", "")), 0)
    package_rank = 1 if job.get("package_status") in {"Package ready", "Demo package"} else 0
    tracker_rank = 1 if job.get("tracker_status") not in {"Not tracked", "Demo only"} else 0
    if sort_by == "Newest first":
        return (newest, score, recommendation_rank)
    if sort_by == "Recommendation":
        return (recommendation_rank, score, newest)
    if sort_by == "Company A-Z":
        return (str(job.get("company", "")).lower(), -score, newest)
    if sort_by == "Package status":
        return (package_rank, score, newest)
    if sort_by == "Tracker status":
        return (tracker_rank, score, newest)
    return (score, newest, recommendation_rank)


def is_strong_match(job: dict[str, Any]) -> bool:
    """Return True only for a confident, eligible canonical Apply result."""
    return (
        bool(job.get("analysis_available"))
        and eligibility_status(job) == "passed"
        and confidence_level(job.get("confidence")) in {"medium", "high"}
        and str(job.get("recommendation", "")) == "Apply"
        and int(job.get("score") or 0) >= 80
    )


def is_current_recommendation(job: dict[str, Any]) -> bool:
    """Return True for current eligible recommendations shown on Dashboard."""
    return (
        bool(job.get("analysis_available"))
        and eligibility_status(job) == "passed"
        and confidence_level(job.get("confidence")) in {"medium", "high"}
        and str(job.get("recommendation", "")) in {"Apply", "Apply / Maybe Apply", "Maybe Apply"}
    )


def sorted_review_jobs(jobs: list[dict[str, Any]], sort_by: str) -> list[dict[str, Any]]:
    """Return Review Jobs sorted for inbox display."""
    reverse = sort_by != "Company A-Z"
    return sorted(jobs, key=lambda job: review_job_sort_key(job, sort_by), reverse=reverse)


def mark_job_not_interested(job: dict[str, Any], tracker_rows: list[dict[str, Any]]) -> tuple[int | None, str]:
    """Archive a tracked job, creating a tracker row first if needed."""
    row = tracker_row_for_job(job, tracker_rows)
    if row is None:
        tracker_id, output = save_job_to_tracker(job)
    else:
        tracker_id = int(row["id"])
        output = ""
    database_path = current_workspace().tracker_database_path
    if database_path is None:
        raise WorkspaceError("Tracker is unavailable in Demo workspace.")
    _, status_output = run_with_captured_output(update_status, int(tracker_id), "archived", database_path)
    return tracker_id, "\n".join(part for part in [output, status_output] if part)


def remember_generated_package(summary: dict[str, Any]) -> None:
    """Store the latest generated package selection for the Application Package page."""
    tracker_id = summary.get("tracker_id")
    package_dir = summary.get("package_dir")
    if tracker_id:
        st.session_state["package_viewer_tracker_id"] = int(tracker_id)
    if package_dir:
        st.session_state["latest_generated_package_dir"] = str(package_dir)
    st.session_state["latest_generated_package_summary"] = {
        "resume": bool(summary.get("resume_path")),
        "cover_letter": bool(summary.get("cover_letter_path")),
        "match_report": bool(summary.get("analysis_path")),
        "internal_notes": bool(summary.get("tailoring_notes_path") or summary.get("cover_letter_notes_path")),
        "tracker_id": tracker_id,
    }


def load_package_notes(package_dir: Path) -> str:
    """Read the most useful internal notes file available for a package."""
    for name in ["cover_letter_notes.md", "tailoring_notes.md", "analysis.md"]:
        candidate = package_dir / name
        if candidate.exists():
            return read_text_file(candidate)
    return ""


def first_existing_package_file(package_dir: Path, names: list[str]) -> Path | None:
    """Return the first known generated package file that exists."""
    for name in names:
        candidate = package_dir / name
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def existing_package_files(package_dir: Path, names: list[str]) -> list[Path]:
    """Return existing known generated files directly inside one package folder."""
    try:
        resolved_package_dir = package_dir.resolve()
    except OSError:
        return []

    files = []
    for name in names:
        candidate = package_dir / name
        try:
            if (
                candidate.exists()
                and candidate.is_file()
                and candidate.resolve().parent == resolved_package_dir
                and candidate.name in INTERNAL_PACKAGE_FILES
            ):
                files.append(candidate)
        except OSError:
            continue
    return files


def build_application_package_zip(package_dir: Path) -> tuple[bytes, list[Path]]:
    """Create an in-memory ZIP containing only selected generated package files."""
    package_files = existing_package_files(package_dir, PACKAGE_ZIP_FILE_ORDER)
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for path in package_files:
            zip_file.write(path, arcname=path.name)
    return zip_buffer.getvalue(), package_files


def package_zip_filename(package_dir: Path) -> str:
    """Build a readable ZIP download name for one selected package."""
    family = package_dir.parent.name if package_dir.parent != package_dir else "application"
    base_name = safe_slug(f"{family}_{package_dir.name}") or "application_package"
    return f"{base_name}.zip"


def readiness_status(
    *,
    source_exists: bool,
    docx_exists: bool | None = None,
    optional: bool = False,
    read_only_sample: bool = False,
) -> str:
    """Return a short user-facing package readiness status."""
    if docx_exists is True:
        return "Ready"
    if docx_exists is False:
        if read_only_sample:
            return "Unavailable"
        return "Can generate" if source_exists else "Missing source"
    if source_exists:
        return "Ready"
    return "Optional" if optional else "Missing"


def render_readiness_checklist(
    resume_md_path: Path,
    resume_docx_path: Path | None,
    cover_letter_md_path: Path,
    cover_letter_docx_path: Path,
    analysis_path: Path,
    internal_notes_paths: list[Path],
) -> None:
    """Show package readiness without exposing raw paths in the main flow."""
    st.markdown("**Application Materials**")
    st.table(
        [
            {"Material": "Tailored Resume", "Status": readiness_status(source_exists=resume_md_path.exists())},
            {
                "Material": "Resume DOCX",
                "Status": readiness_status(
                    source_exists=resume_md_path.exists(),
                    docx_exists=resume_docx_path is not None,
                    read_only_sample=demo_mode_enabled(),
                ),
            },
            {"Material": "Cover Letter", "Status": readiness_status(source_exists=cover_letter_md_path.exists())},
            {
                "Material": "Cover Letter DOCX",
                "Status": readiness_status(
                    source_exists=cover_letter_md_path.exists(),
                    docx_exists=cover_letter_docx_path.exists(),
                    read_only_sample=demo_mode_enabled(),
                ),
            },
            {"Material": "Match Report", "Status": readiness_status(source_exists=analysis_path.exists())},
            {
                "Material": "Internal Notes",
                "Status": readiness_status(source_exists=bool(internal_notes_paths), optional=True),
            },
        ]
    )


def generate_resume_docx_for_package(package_dir: Path) -> tuple[Path | None, list[str]]:
    """Generate tailored_resume.docx for a selected package when possible."""
    resume_md_path = package_dir / "tailored_resume.md"
    if not resume_md_path.exists():
        return None, ["Tailored resume source is missing."]
    resume_docx_path = package_dir / "tailored_resume.docx"
    warnings = export_resume_to_docx(resume_md_path, resume_docx_path)
    return resume_docx_path, warnings


def generate_cover_letter_docx_for_package(package_dir: Path) -> tuple[Path | None, list[str]]:
    """Regenerate cover_letter.docx for a selected package when possible."""
    cover_letter_md_path = package_dir / "cover_letter.md"
    if not cover_letter_md_path.exists():
        return None, ["Cover letter source is missing."]
    cover_letter_docx_path = package_dir / "cover_letter.docx"
    warnings = export_cover_letter_to_docx(
        cover_letter_md_path,
        cover_letter_docx_path,
        parse_job_metadata_from_package(package_dir),
        generic_cover_letter_template(current_workspace()),
    )
    return cover_letter_docx_path, warnings


def fetch_run_label(run: dict[str, Any]) -> str:
    """Build a readable fetch-run selector label."""
    return (
        f"{run.get('created_at', '')} | {source_display_name(str(run.get('source', '')))} | "
        f"{run.get('region', '-') or '-'} | {run.get('query', '-') or '-'} | "
        f"{run.get('new_jobs_count', 0)} new"
    )


def render_fetch_run_job_table(jobs: list[dict[str, Any]], empty_message: str) -> None:
    """Render compact job summaries stored on a fetch run."""
    if not jobs:
        st.info(empty_message)
        return
    st.dataframe(
        [
            {
                "Company": job.get("company", ""),
                "Role": get_job_display_title(job),
                "Location": normalize_location(str(job.get("location", ""))),
                "Path": job.get("path", ""),
            }
            for job in jobs
        ],
        width="stretch",
        hide_index=True,
    )


def render_fetch_run_job_cards(jobs: list[dict[str, Any]], empty_message: str) -> None:
    """Render fetched jobs as simple cards without changing fetch behavior."""
    if not jobs:
        st.info(empty_message)
        return

    for index, job in enumerate(jobs, start=1):
        with st.container(border=True):
            company = str(job.get("company", "") or "Unknown company")
            role = get_job_display_title(job, fallback="Unknown role")
            location = normalize_location(str(job.get("location", ""))) or "-"
            source = source_display_name(str(job.get("source", "")))
            st.markdown(f"**{company}**")
            st.write(role)
            st.caption(f"{location} | {source}")
            if job.get("path"):
                with st.expander("View Details", expanded=False):
                    path = PROJECT_ROOT / str(job.get("path", ""))
                    if path.exists():
                        st.markdown(read_text_file(path)[:1200])
                    else:
                        st.write(f"Saved path: `{job.get('path', '')}`")


def render_fetch_run_details(run: dict[str, Any]) -> None:
    """Show new jobs first, with repeated jobs collapsed by default."""
    if not run:
        return
    st.write(
        f"{source_display_name(str(run.get('source', '')))} / {run.get('region', '-') or '-'} / "
        f"{run.get('query', '-') or '-'}"
    )
    st.caption(
        f"{run.get('total_jobs_returned', 0)} returned | "
        f"{run.get('new_jobs_count', 0)} new | "
        f"{run.get('duplicate_jobs_count', 0)} already seen | "
        f"Status: {run.get('fetch_status', '-')}"
    )
    notes = str(run.get("notes", "") or "").strip()
    if notes:
        st.warning(notes)
    render_fetch_run_job_cards(run.get("new_jobs", []) or [], "No new jobs were discovered in this search.")
    with st.expander("Compact table view", expanded=False):
        render_fetch_run_job_table(run.get("new_jobs", []) or [], "No new jobs were discovered in this search.")
    with st.expander("Advanced: already seen jobs from this search", expanded=False):
        render_fetch_run_job_cards(run.get("previously_seen_jobs", []) or [], "No already seen jobs were returned.")


def render_fetch_history_section() -> None:
    """Render recent fetch-run history and a past-run review selector."""
    runs = load_fetch_runs(limit=20)
    st.markdown("**Fetch History**")
    if not runs:
        st.info("No fetch history yet.")
        return
    st.dataframe(
        [
            {
                "Date/time": run.get("created_at", ""),
                "Source": source_display_name(str(run.get("source", ""))),
                "Region": run.get("region", ""),
                "Query": run.get("query", ""),
                "Total returned": run.get("total_jobs_returned", 0),
                "New jobs": run.get("new_jobs_count", 0),
                "Already seen": run.get("duplicate_jobs_count", 0),
                "Status": run.get("fetch_status", ""),
            }
            for run in runs
        ],
        width="stretch",
        hide_index=True,
    )
    labels = [fetch_run_label(run) for run in runs]
    selected_label = st.selectbox("Review search results", labels, key="fetch_history_selected")
    render_fetch_run_details(runs[labels.index(selected_label)])


def render_markdown_file(path: Path, title: str) -> None:
    """Show one Markdown file in a simple expander."""
    if not path.exists():
        st.info(f"{title} not found.")
        return

    with st.expander(title, expanded=False):
        st.markdown(read_text_file(path))


def render_page_header(title: str, subtitle: str | None = None) -> None:
    """Render a compact page heading below the top navigation."""
    subtitle_html = ""
    if subtitle:
        subtitle_html = f'<div class="page-subtitle">{html.escape(subtitle)}</div>'
    st.markdown(
        f"""
        <div class="page-header">
          <div class="page-title">{html.escape(title)}</div>
          {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def dashboard_tab() -> None:
    """Render the default customer-facing home page."""
    render_page_header(
        "Dashboard",
        "Your local-first application workspace. Find and save jobs, generate drafts, then review before applying.",
    )
    if demo_mode_enabled():
        st.info("Demo workspace is using sanitized sample jobs and a sample package. Live fetch and tracker writes are disabled.")

    jobs = load_screened_jobs()
    tracker_rows = [] if demo_mode_enabled() else load_tracker_rows(sort_by="created_at", descending=True)
    analyzed_jobs = [job for job in jobs if job.get("analysis_available")]
    strong_matches = sum(1 for job in jobs if is_strong_match(job))

    metric_cols = st.columns(4)
    metric_cols[0].metric("Saved jobs", len(jobs))
    metric_cols[1].metric("Strong matches", strong_matches if analyzed_jobs else "Not available")
    metric_cols[2].metric("Generated packages", count_generated_packages())
    metric_cols[3].metric("Tracked applications", len(tracker_rows))

    st.markdown("**How this copilot works**")
    st.write(
        "Jobs, generated documents, and tracker records stay on this machine. The app can draft a safer resume "
        "and cover letter package from your source resume, but it never submits an application or answers forms for you."
    )

    recommended_jobs = [
        job
        for job in jobs
        if is_current_recommendation(job)
    ][:3]
    st.markdown("**Recent recommended jobs**")
    if not recommended_jobs:
        st.info("Not available yet. Find or add jobs to see recommendations here.")
    for job in recommended_jobs:
        with st.container(border=True):
            st.markdown(f"**{job['company']}**")
            st.write(get_job_display_title(job))
            presentation = build_fit_presentation(job)
            st.caption(f"{job['normalized_location']} | {presentation['card_status']}")


def fetch_jobs_tab() -> None:
    """Render the fetch-jobs workflow."""
    render_page_header(
        "Find Jobs",
        "Search supported job sources and save roles for review.",
    )
    if demo_mode_enabled():
        st.info("Demo workspace is active. Live job fetching is disabled; use Review Jobs to explore sample cards.")
    backend_outputs = []

    query = st.text_input("Target role / query", value="data analyst")
    region = st.selectbox("Region", REGION_OPTIONS, index=0, key="fetch_region")
    region_config = REGION_CONFIG[region]
    adzuna_country = region_config["adzuna_country"]
    adzuna_location = region_config["adzuna_location"]
    jooble_location = region_config["jooble_location"]

    if region == "Custom":
        location_text = st.text_input("Custom Location", key="fetch_custom_location")
        if SHOW_DEBUG_UI:
            adzuna_country = st.text_input("Developer: Adzuna country", value=adzuna_country)
        adzuna_location = location_text
        jooble_location = location_text

    with st.form("fetch_jobs_form"):
        recommendation_limit = st.slider(
            "Number of recommendations",
            min_value=MIN_RECOMMENDATION_LIMIT,
            max_value=MAX_RECOMMENDATION_LIMIT,
            value=DEFAULT_RECOMMENDATION_LIMIT,
            help="How many ranked jobs to display after filtering and duplicate removal.",
        )
        sources = st.multiselect(
            "Sources",
            ["adzuna", "jooble"],
            default=["adzuna", "jooble"],
        )
        adzuna_is_supported = adzuna_country.lower() in ADZUNA_SUPPORTED_COUNTRIES
        if "adzuna" in sources and not adzuna_is_supported:
            st.warning(
                "Adzuna is not available for this region. Jooble can still search this location."
            )
        fetch_limit_per_source = st.slider(
            "Jobs per source",
            min_value=5,
            max_value=MAX_FETCH_LIMIT_PER_SOURCE,
            value=DEFAULT_FETCH_LIMIT_PER_SOURCE,
            help="How many jobs to request from each source before filtering.",
        )
        submitted = st.form_submit_button("Find Jobs")

    if submitted:
        if demo_mode_enabled():
            st.info("Demo workspace does not call external job APIs. Select Personal and add API keys in `.env` for live fetch.")
            return
        if not sources:
            st.error("Select at least one source.")
            return

        all_saved_paths = []
        fetch_results = []
        fetch_errors = []

        for source in sources:
            if source == "adzuna" and not adzuna_is_supported:
                backend_outputs.append(
                    f"[adzuna] Skipped because country `{adzuna_country}` is not supported."
                )
                continue

            source_location = adzuna_location if source == "adzuna" else jooble_location
            args = SimpleNamespace(
                source=source,
                country=adzuna_country,
                query=query,
                location=source_location,
                max_results=fetch_limit_per_source,
            )
            try:
                result, output = run_with_captured_output(
                    fetch_and_save_jobs,
                    args,
                )
                saved_paths = list(result.get("saved_paths", []) if isinstance(result, dict) else [])
                saved_paths = relocate_fetched_jobs_to_workspace(saved_paths, source)
                run_record = dict(result.get("fetch_run", {}) if isinstance(result, dict) else {})
                all_saved_paths.extend(saved_paths)
                if run_record:
                    fetch_results.append(run_record)
                    st.session_state["latest_fetch_run_id"] = run_record.get("fetch_run_id", "")
                if output:
                    backend_outputs.append(f"[{source}]\n{output}")
            except Exception as error:  # noqa: BLE001
                fetch_errors.append(f"{source}: {error}")
                backend_outputs.append(f"[{source}] {error}")

        total_returned = sum(int(run.get("total_jobs_returned", 0) or 0) for run in fetch_results)
        total_new = sum(int(run.get("new_jobs_count", 0) or 0) for run in fetch_results)
        total_seen = sum(int(run.get("duplicate_jobs_count", 0) or 0) for run in fetch_results)
        total_skipped = sum(int(run.get("skipped_jobs_count", 0) or 0) for run in fetch_results)
        total_errors = len(fetch_errors) + total_skipped
        all_new_jobs = [
            job
            for run in fetch_results
            for job in (run.get("new_jobs", []) or [])
        ]
        all_seen_jobs = [
            job
            for run in fetch_results
            for job in (run.get("previously_seen_jobs", []) or [])
        ]
        if fetch_results and fetch_errors:
            st.warning("Search completed for some sources. One or more sources could not be searched.")
        elif fetch_results:
            st.success("Search complete.")
        else:
            st.error("Search failed. Check API keys or use Demo workspace.")
        if any(".env" in error or "API_KEY" in error or "APP_ID" in error or "APP_KEY" in error for error in fetch_errors):
            st.info("Live job search requires API keys. You can use Demo workspace or add keys to `.env`.")
        st.caption(
            f"Search complete: {total_returned} returned · {total_new} new · "
            f"{total_seen} already seen · {len(all_saved_paths)} saved · {total_errors} errors"
        )
        st.session_state["recommendation_limit"] = recommendation_limit
        if total_new == 0 and fetch_results:
            st.info(
                "No new jobs found.\n\n"
                "All returned jobs were already seen in previous searches.\n\n"
                "Try broadening the query, increasing jobs per source, changing region, or reviewing saved jobs."
            )
            next_left, next_right = st.columns(2)
            with next_left:
                if st.button("Review Saved Jobs", width="stretch"):
                    go_to_page("Review Jobs")
            with next_right:
                if st.button("Add Target Job Manually", width="stretch"):
                    go_to_page("Add Target Job")
        elif all_new_jobs:
            for run in fetch_results:
                new_jobs = run.get("new_jobs", []) or []
                if not new_jobs:
                    continue
                st.markdown(f"**New jobs from {source_display_name(str(run.get('source', '')))}**")
                render_fetch_run_job_cards(new_jobs, "No new jobs in this search.")
            with st.expander("Compact table view", expanded=False):
                render_fetch_run_job_table(all_new_jobs, "No new jobs in this search.")

        if fetch_results:
            with st.expander("Search details", expanded=False):
                st.dataframe(
                    [
                        {
                            "Source": source_display_name(str(run.get("source", ""))),
                            "Returned": int(run.get("total_jobs_returned", 0) or 0),
                            "New": int(run.get("new_jobs_count", 0) or 0),
                            "Already seen": int(run.get("duplicate_jobs_count", 0) or 0),
                            "Saved": len(run.get("new_jobs", []) or []),
                            "Errors": int(run.get("skipped_jobs_count", 0) or 0),
                        }
                        for run in fetch_results
                    ],
                    width="stretch",
                    hide_index=True,
                )
                if all_seen_jobs:
                    st.markdown("**Already seen jobs**")
                    render_fetch_run_job_table(all_seen_jobs, "No already seen jobs in this search.")
                if SHOW_DEBUG_UI and backend_outputs:
                    st.markdown("**Developer fetch output**")
                    st.text("\n\n".join(backend_outputs))

    if SHOW_DEBUG_UI:
        with st.expander("Developer search details", expanded=False):
            st.markdown("**Source mapping**")
            st.write(f"Adzuna country: `{adzuna_country}`")
            st.write(f"Adzuna location: `{adzuna_location}`")
            st.write(f"Jooble location: `{jooble_location}`")
            st.markdown("**Rate-limit notes**")
            st.warning(
                "Do not repeatedly open many Adzuna links in a short time. "
                "If Adzuna shows 'Too Many Requests', wait 10-30 minutes and avoid refreshing."
            )
            if backend_outputs:
                st.markdown("**Internal fetch metadata**")
                st.text("\n\n".join(backend_outputs))
            st.markdown("**Raw fetch history**")
            render_fetch_history_section()


def clear_manual_job_session_state(clear_upload: bool = True) -> None:
    """Clear transient Add Target Job UI state without deleting saved jobs.

    Streamlit file upload widgets cannot be assigned directly after creation, so
    upload reset is handled by bumping a key suffix before the widget is rendered.
    """
    keys_to_clear = set(MANUAL_FORM_STATE_KEYS) | set(MANUAL_TRANSIENT_STATE_KEYS)
    keys_to_clear.update(
        key
        for key in st.session_state.keys()
        if key != "manual_upload_key_suffix" and key.startswith(MANUAL_SELECTION_STATE_KEY_PREFIXES)
    )
    if clear_upload:
        current_suffix = int(st.session_state.get("manual_upload_key_suffix", 0) or 0)
        st.session_state["manual_upload_key_suffix"] = current_suffix + 1
    for key in keys_to_clear:
        st.session_state.pop(key, None)
    st.session_state["manual_status"] = "Saved"
    st.session_state["manual_source"] = SOURCE_OPTIONS[0]
    st.session_state["manual_last_cleanup_timestamp"] = datetime.now().replace(microsecond=0).isoformat()


def clear_manual_state_for_new_extraction() -> None:
    """Reset stale form/parser state while preserving the current upload widget."""
    clear_manual_job_session_state(clear_upload=False)


def clean_generated_outputs() -> list[Path]:
    """Delete generated application packages only; never delete saved manual jobs."""
    workspace = current_workspace()
    workspace.require_writable()
    output_root = workspace.generated_dir.resolve()
    if output_root != (workspace.root / "generated").resolve():
        raise ValueError("Generated output cleanup is restricted to the Personal workspace.")
    output_root.mkdir(parents=True, exist_ok=True)

    deleted_paths = []
    for child in output_root.iterdir():
        if child.name == ".gitkeep":
            continue
        if child.is_dir():
            delete_directory_tree(child)
        else:
            child.unlink()
        deleted_paths.append(child)
    return deleted_paths


def render_manual_cleanup_controls() -> None:
    """Render clearly separated cleanup actions for manual workflow state."""
    cleanup_left, cleanup_right = st.columns(2)
    with cleanup_left:
        if st.button("Clear Current Target Job Form", key="clear_manual_job_form"):
            clear_manual_job_session_state(clear_upload=True)
            st.session_state["manual_cleanup_message"] = "Current target job form and extraction state cleared."
            st.rerun()
    with cleanup_right:
        if st.button("Clean Generated Outputs", key="clean_generated_outputs"):
            try:
                deleted_paths = clean_generated_outputs()
                for key in ["manual_generated_summary", "manual_generated_backend_output", "manual_generated_error"]:
                    st.session_state.pop(key, None)
                st.session_state["manual_cleanup_message"] = (
                    f"Generated outputs cleaned. Removed {len(deleted_paths)} item(s). "
                    "Saved target jobs and tracker records were not deleted."
                )
            except Exception as error:  # noqa: BLE001
                st.session_state["manual_cleanup_message"] = f"Could not clean generated outputs: {error}"
            st.rerun()


MANUAL_SUGGESTION_FIELD_RULES = {
    "manual_company": ("company", {"high", "medium"}, "company_confidence"),
    "manual_title": ("title", {"high", "medium"}, "job_title_confidence"),
    "manual_location": ("location", {"high", "medium"}, "location_confidence"),
    "manual_source": ("source", {"high", "medium"}, "source_confidence"),
    "manual_url": ("url", {"high", "medium"}, "url_confidence"),
    "manual_salary_range": ("salary_range", {"high", "medium"}, "salary_confidence"),
    "manual_visa_note": ("visa_note", {"high", "medium"}, "visa_confidence"),
    "manual_status": ("status", {"high", "medium"}, "status_confidence"),
    "manual_notes": ("notes", {"high", "medium"}, "notes_confidence"),
    "manual_job_description": ("job_description", {"high", "medium"}, "job_description_confidence"),
}


def apply_suggestions_to_empty_fields(suggestions: dict[str, Any]) -> None:
    """Synchronize parser suggestions into the exact form widget keys.

    This runs before the form widgets are instantiated. Title suggestions with
    high or medium confidence are accepted into `manual_title` when it is empty,
    so the compact summary and the editable field cannot contradict each other.
    """
    confidence_rules = {
        key: rule
        for key, rule in MANUAL_SUGGESTION_FIELD_RULES.items()
        if key in {
            "manual_company",
            "manual_title",
            "manual_location",
            "manual_visa_note",
            "manual_source",
            "manual_url",
            "manual_salary_range",
            "manual_status",
            "manual_notes",
            "manual_job_description",
        }
    }
    for state_key, (suggestion_key, allowed_confidences, confidence_key) in confidence_rules.items():
        suggested_value = suggestions.get(suggestion_key)
        confidence = str(suggestions.get(confidence_key) or ("medium" if suggested_value else "")).lower()
        if st.session_state.get(state_key) or not suggested_value or confidence not in allowed_confidences:
            continue
        if state_key == "manual_source" and suggested_value not in SOURCE_OPTIONS:
            continue
        if state_key == "manual_status" and suggested_value not in STATUS_OPTIONS:
            continue
        st.session_state[state_key] = suggested_value


def apply_suggestion_to_form_field(state_key: str, value: Any) -> None:
    """Apply one suggestion to the Streamlit form state before widgets render."""
    clean_value = str(value or "").strip()
    if clean_value:
        st.session_state[state_key] = clean_value


def form_field_needs_suggestion(state_key: str, suggested_value: Any) -> bool:
    """Return True when a visible Use action would change the form value."""
    suggested = str(suggested_value or "").strip()
    current = str(st.session_state.get(state_key, "") or "").strip()
    return bool(suggested) and current != suggested


def current_manual_suggestions(current_text: str) -> dict[str, Any]:
    """Use one shared parser suggestion object for summary, actions, and save."""
    if not current_text.strip():
        st.session_state["manual_parser_suggestions"] = {}
        return {}
    suggestions = parse_job_description_suggestions(current_text, current_manual_source_metadata())
    st.session_state["manual_parser_suggestions"] = suggestions
    return suggestions


def split_suggestion_lines(value: Any) -> list[str]:
    """Convert parser values to displayable lines."""
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [line.strip() for line in str(value or "").splitlines() if line.strip()]


def manual_source_metadata_from_reports(reports: list[dict[str, Any]]) -> dict[str, list[str]]:
    """Expose PDF/browser metadata to parser heuristics without changing uploads."""
    metadata_titles = []
    for report in reports:
        metadata_title = " ".join(str(report.get("metadata_title", "") or "").split())
        if metadata_title and metadata_title not in metadata_titles:
            metadata_titles.append(metadata_title)
    return {"metadata_titles": metadata_titles}


def current_manual_source_metadata() -> dict[str, list[str]]:
    """Return parser metadata for the current extracted upload."""
    return manual_source_metadata_from_reports(st.session_state.get("manual_extraction_reports", []) or [])


def format_confidence(confidence: Any) -> str:
    """Normalize parser confidence for display."""
    value = str(confidence or "").strip().lower()
    return value if value in {"high", "medium", "low"} else "unknown"


def accepted_manual_value(state_key: str, fallback: str = "Needs manual input") -> str:
    """Read the accepted form value used by both At a Glance and the form."""
    value = str(st.session_state.get(state_key, "") or "").strip()
    return value or fallback


def compact_location_value(location: str) -> str:
    """Keep multi-office locations readable in the one-line summary strip."""
    if location.lower().startswith("multiple offices:"):
        return "Multiple offices"
    return location


def build_manual_red_flags(
    suggestions: dict[str, Any],
    *,
    url: str,
    job_description: str,
    reports: list[dict[str, Any]],
) -> list[str]:
    """Create user-facing warnings from parser confidence and extraction state."""
    warnings = []
    title_confidence = format_confidence(suggestions.get("job_title_confidence"))
    if not suggestions.get("title"):
        warnings.append("Job title was not confidently detected.")
    elif title_confidence != "high":
        warnings.append(f"Job title is a {title_confidence}-confidence suggestion; review before saving.")
    if not suggestions.get("company"):
        warnings.append("Company was not detected.")
    if not suggestions.get("location"):
        warnings.append("Location was not detected.")
    location_options = suggestions.get("location_options")
    if isinstance(location_options, list) and len(location_options) > 1:
        warnings.append("Multiple office locations detected.")
    visa_note = str(suggestions.get("visa_note", "") or "").lower()
    if "no visa sponsorship" in visa_note:
        warnings.append("No visa sponsorship indicated.")
    if not url.strip():
        warnings.append("Missing official job URL.")
    if len(job_description.split()) < 80 and job_description.strip():
        warnings.append("Job description is short; extraction may be incomplete.")
    for report in reports:
        for warning in report.get("warnings") or []:
            if warning and str(warning) not in warnings:
                warnings.append(str(warning))
    return warnings[:6]


def match_readiness_for(suggestions: dict[str, Any], warnings: list[str], job_description: str) -> tuple[str, str]:
    """Return a compact readiness label and one short reason."""
    title_confidence = format_confidence(suggestions.get("job_title_confidence"))
    if not job_description.strip() or len(job_description.split()) < 80:
        return "Missing key info", "job description is short or empty."
    if not st.session_state.get("manual_company"):
        return "Missing key info", "company is missing."
    if not st.session_state.get("manual_title"):
        return "Missing key info", "title is missing."
    if not st.session_state.get("manual_location"):
        return "Missing key info", "location is missing."
    if title_confidence == "medium":
        return "Needs review", "title may need confirmation."
    location_options = suggestions.get("location_options")
    if isinstance(location_options, list) and len(location_options) > 1:
        return "Needs review", "multiple locations detected."
    if not st.session_state.get("manual_url"):
        return "Needs review", "job URL is missing."
    visa_note = str(st.session_state.get("manual_visa_note", "") or "").lower()
    if "no visa sponsorship" in visa_note:
        return "Needs review", "visa sponsorship restriction detected."
    if warnings:
        return "Needs review", "one or more fields may need confirmation."
    return "Ready to save", ""


def render_suggestion_action(label: str, state_key: str, suggested_value: Any, button_label: str | None = None) -> None:
    """Render a concise Use button when a parser suggestion differs from the form."""
    if not form_field_needs_suggestion(state_key, suggested_value):
        return
    clean_value = str(suggested_value or "").strip()
    action_left, action_right = st.columns([3, 2])
    with action_left:
        st.caption(f"{label}: {clean_value}")
    with action_right:
        if st.button(button_label or "Use", key=f"use_{state_key}", help=f"Apply suggested {label.lower()} to the form."):
            apply_suggestion_to_form_field(state_key, clean_value)
            st.rerun()


def render_suggestion_actions(suggestions: dict[str, Any]) -> None:
    """Show visible form-sync actions for suggestions that are not yet applied."""
    actions = [
        ("Company", "manual_company", suggestions.get("company"), None),
        ("Job title", "manual_title", suggestions.get("title"), "Use suggested title"),
        ("Location", "manual_location", suggestions.get("location"), None),
        ("Visa note", "manual_visa_note", suggestions.get("visa_note"), None),
    ]
    visible_actions = [
        (label, state_key, value, button_label)
        for label, state_key, value, button_label in actions
        if form_field_needs_suggestion(state_key, value)
    ]
    if not visible_actions:
        return
    st.caption("Suggested fields not applied yet")
    for label, state_key, value, button_label in visible_actions:
        render_suggestion_action(label, state_key, value, button_label)


def render_suggestion_details(
    suggestions: dict[str, Any],
    *,
    responsibilities: list[str],
    requirements: list[str],
    keywords: list[Any],
) -> None:
    """Keep parser suggestions and evidence out of the main At a Glance strip."""
    has_unapplied = any(
        form_field_needs_suggestion(state_key, value)
        for _, state_key, value, _ in [
            ("Company", "manual_company", suggestions.get("company"), None),
            ("Job title", "manual_title", suggestions.get("title"), "Use suggested title"),
            ("Location", "manual_location", suggestions.get("location"), None),
            ("Visa note", "manual_visa_note", suggestions.get("visa_note"), None),
        ]
    )
    if has_unapplied:
        with st.expander("Suggestions not applied", expanded=False):
            render_suggestion_actions(suggestions)
            if st.button("Use Suggestions for Empty Fields", key="manual_use_all_suggestions"):
                apply_suggestions_to_empty_fields(suggestions)
                st.rerun()

    has_evidence = any(
        suggestions.get(key)
        for key in [
            "company_confidence",
            "company_evidence",
            "job_title_confidence",
            "job_title_evidence",
            "location_confidence",
            "location_evidence",
            "visa_confidence",
            "visa_evidence",
        ]
    )
    if has_evidence:
        with st.expander("Why this was detected", expanded=False):
            for label, confidence_key, evidence_key in [
                ("Company", "company_confidence", "company_evidence"),
                ("Title", "job_title_confidence", "job_title_evidence"),
                ("Location", "location_confidence", "location_evidence"),
                ("Visa", "visa_confidence", "visa_evidence"),
            ]:
                confidence = suggestions.get(confidence_key)
                evidence = suggestions.get(evidence_key)
                if confidence or evidence:
                    st.caption(f"{label} confidence: {format_confidence(confidence)}")
                    if evidence:
                        st.write(evidence)

    if keywords:
        with st.expander("Keywords", expanded=False):
            st.caption(", ".join(str(keyword) for keyword in keywords[:5]))

    if responsibilities or requirements:
        with st.expander("Responsibilities / requirements", expanded=False):
            if responsibilities:
                st.caption(f"Top responsibilities: {len(responsibilities)} detected")
                for line in responsibilities[:5]:
                    st.write(f"- {line}")
            if requirements:
                st.caption(f"Top requirements: {len(requirements)} detected")
                for line in requirements[:5]:
                    st.write(f"- {line}")

    parsed_sections = suggestions.get("parsed_sections")
    if SHOW_DEBUG_UI and isinstance(parsed_sections, dict) and parsed_sections:
        with st.expander("Advanced: parser details", expanded=False):
            st.json(parsed_sections)


def render_compact_at_a_glance(
    suggestions: dict[str, Any],
    *,
    job_description: str = "",
    reports: list[dict[str, Any]] | None = None,
) -> None:
    """Render a compact summary strip from accepted form state, not raw suggestions."""
    reports = reports or []
    st.markdown("**At a Glance**")
    if not job_description.strip():
        st.caption("Extract or paste a job description to see a summary.")
        return

    responsibilities = split_suggestion_lines(suggestions.get("responsibilities", ""))
    requirements = split_suggestion_lines(suggestions.get("requirements", ""))
    keywords = suggestions.get("keywords") if isinstance(suggestions.get("keywords"), list) else []
    warnings = build_manual_red_flags(
        suggestions,
        url=str(st.session_state.get("manual_url", "")),
        job_description=job_description,
        reports=reports,
    )
    readiness, reason = match_readiness_for(suggestions, warnings, job_description)

    company = accepted_manual_value("manual_company")
    title = accepted_manual_value("manual_title")
    location = compact_location_value(accepted_manual_value("manual_location"))
    visa_note = accepted_manual_value("manual_visa_note", "Not detected")

    st.caption(f"Company: {company} | Title: {title} | Location: {location}")
    st.caption(f"Work auth: {visa_note}")
    status_text = readiness if not reason else f"{readiness}: {reason}"
    st.caption(f"Status: {status_text}")
    render_suggestion_details(
        suggestions,
        responsibilities=responsibilities,
        requirements=requirements,
        keywords=keywords,
    )


def combine_upload_extraction_results(uploaded_files: list[Any]) -> tuple[str, str, str, list[str], list[dict[str, Any]]]:
    """Extract each uploaded file in order and combine text with visible separators."""
    raw_parts = []
    cleaned_parts = []
    messages = []
    filenames = []
    reports = []

    # Multi-file uploads matter for long LinkedIn posts where one screenshot only
    # captures the visible viewport. Each file keeps a separator for traceability.
    for index, uploaded_file in enumerate(uploaded_files, start=1):
        file_bytes = uploaded_file.getvalue()
        filenames.append(uploaded_file.name)
        result = extract_text_from_upload(uploaded_file.name, file_bytes)
        if result.error:
            messages.append(f"{uploaded_file.name}: {result.error}")
        if result.warning:
            messages.append(f"{uploaded_file.name}: {result.warning}")
        if result.report:
            report = {"file_name": uploaded_file.name, **result.report}
            reports.append(report)
        if not result.text:
            continue

        header = f"--- Extracted text from file {index}: {uploaded_file.name} ---"
        raw_parts.append(f"{header}\n{result.text}")
        cleaned_parts.append(f"{header}\n{clean_extracted_job_text(result.text)}")

    return "\n\n".join(raw_parts).strip(), "\n\n".join(cleaned_parts).strip(), "\n".join(messages), filenames, reports


def render_extraction_reports(reports: list[dict[str, Any]]) -> None:
    """Render compact extraction status with detailed report collapsed."""
    if not reports:
        return
    total_pages = sum(int(report.get("pages_processed", 0) or 0) for report in reports)
    total_chars = sum(int(report.get("characters_extracted", 0) or 0) for report in reports)
    total_sections = sum(int(report.get("sections_detected", 0) or 0) for report in reports)
    methods = ", ".join(sorted({str(report.get("method", "unknown")) for report in reports}))
    st.success(
        f"Extraction complete: {total_pages or '-'} pages, {total_chars:,} characters extracted, "
        f"{total_sections} sections detected. Method: {methods}."
    )
    if not SHOW_DEBUG_UI:
        return
    with st.expander("Advanced: extraction report", expanded=False):
        for report in reports:
            st.write(f"File name: {report.get('file_name', '-')}")
            st.write(f"Pages processed: {report.get('pages_processed', '-')}")
            st.write(f"Characters extracted: {report.get('characters_extracted', '-')}")
            st.write(f"Extraction method used: {report.get('method', '-')}")
            if report.get("metadata_title"):
                st.write(f"PDF/browser title metadata: {report.get('metadata_title')}")
            headings = report.get("detected_section_headings") or []
            st.write("Detected section headings: " + (", ".join(headings) if headings else "-"))
            warnings = report.get("warnings") or []
            if warnings:
                st.write("Warnings:")
                for warning in warnings:
                    st.write(f"- {warning}")


def sorted_manual_records() -> list[dict[str, Any]]:
    """Return saved manual records newest-first for selectors and tables."""
    return sorted(load_manual_jobs(), key=lambda record: str(record.get("created_at", "")), reverse=True)


def manual_record_label(record: dict[str, Any]) -> str:
    """Build a compact stable label for saved manual job selectors."""
    return (
        f"{record.get('company', '')} | {record.get('title', '')} | "
        f"{record.get('created_at', '')} | {record.get('id', '')}"
    )


def select_manual_record(records: list[dict[str, Any]], key: str) -> dict[str, Any] | None:
    """Render a saved manual job selector and return the selected record."""
    if not records:
        st.info("No target jobs saved yet.")
        return None
    labels = [manual_record_label(record) for record in records]
    selected_label = st.selectbox("Select saved target job", labels, key=key)
    return records[labels.index(selected_label)]


def render_manual_record_long_details(record: dict[str, Any]) -> None:
    """Keep long saved-job details collapsed by default."""
    url = str(record.get("url", "") or "")
    if url:
        st.link_button("Open Job URL", url)
    else:
        st.write("Job URL: -")

    with st.expander("Full job description", expanded=False):
        st.text(record.get("job_description", ""))
    if SHOW_DEBUG_UI:
        with st.expander("Advanced: raw Markdown paths and uploads", expanded=False):
            st.write(f"Markdown file: `{record.get('markdown_path', '')}`")
            upload_filenames = record.get("source_upload_filenames") or []
            if upload_filenames:
                st.write("Uploads:")
                for filename in upload_filenames:
                    st.write(f"- `{filename}`")
            elif record.get("source_upload_filename"):
                st.write(f"Upload: `{record.get('source_upload_filename')}`")

        with st.expander("Advanced: OCR and parser details", expanded=False):
            st.write("Raw extracted OCR text")
            st.text(record.get("raw_extracted_text", "") or record.get("extracted_text", "") or "")
            st.write("Cleaned OCR text")
            st.text(record.get("cleaned_extracted_text", "") or "")
            st.write("Parser metadata")
            st.json(record.get("parser_suggestions", {}) or {})


def generate_package_for_manual_record(record: dict[str, Any], button_key: str) -> None:
    """Run the existing package workflow for a saved manual job."""
    markdown_path = PROJECT_ROOT / str(record.get("markdown_path", ""))
    if not markdown_path.exists():
        st.error("Saved Markdown file was not found for this target job.")
        return

    fields = render_manual_company_confirmation(record, key_prefix=f"{button_key}_manual_company")
    if not company_generation_allowed(fields):
        st.info(
            "Company name needs confirmation before generating a cover letter. "
            "This prevents using the wrong company name in your application."
        )
        return

    if st.button("Generate Package", key=button_key, type="primary"):
        st.session_state.pop("manual_generated_error", None)
        try:
            summary, output = run_with_captured_output(
                create_application_package,
                job_description_path=markdown_path,
                workspace=current_workspace(),
                company=str(record.get("company", "")).strip(),
                role=str(record.get("title", "")).strip(),
                location=str(record.get("location", "")).strip(),
                job_url=str(record.get("url", "")).strip(),
            )
            update_manual_job(str(record["id"]), status="Resume Generated", notes=str(record.get("notes", "")))
            st.session_state["manual_generated_summary"] = {
                "match_score": summary["match_score"],
                "recommendation": summary["recommendation"],
                "analysis_path": relative_path(summary["analysis_path"]),
                "resume_path": relative_path(summary["resume_path"]),
                "resume_docx_path": relative_path(summary["resume_docx_path"]),
                "cover_letter_path": relative_path(summary["cover_letter_path"]),
                "cover_letter_docx_path": relative_path(summary["cover_letter_docx_path"]),
                "tracker_id": summary["tracker_id"],
            }
            st.session_state["manual_generated_backend_output"] = output
            st.success("Manual job analyzed and application package generated.")
            st.write(f"Overall score: {summary['match_score']}/100")
            st.write(f"Recommendation: {summary['recommendation']}")
            st.write(f"Analysis file: `{relative_path(summary['analysis_path'])}`")
            st.write(f"Tailored resume Markdown: `{relative_path(summary['resume_path'])}`")
            st.write(f"Resume DOCX: `{relative_path(summary['resume_docx_path'])}`")
            st.write(f"Cover letter Markdown: `{relative_path(summary['cover_letter_path'])}`")
            st.write(f"Cover letter DOCX: `{relative_path(summary['cover_letter_docx_path'])}`")
            st.write(f"Tracker id: {summary['tracker_id']}")
            if summary.get("uk_review_notes"):
                st.warning("UK work authorization review")
                for note in summary["uk_review_notes"]:
                    st.write(f"- {note}")
            if output:
                with st.expander("Technical output (advanced)", expanded=False):
                    st.text(output)
        except Exception as error:  # noqa: BLE001
            st.session_state["manual_generated_error"] = str(error)
            st.error(str(error))


def prepare_manual_job_session_state() -> None:
    """Apply queued state updates before widgets are created."""
    pending_suggestions = st.session_state.pop("manual_pending_suggestions", None)
    if pending_suggestions:
        apply_suggestions_to_empty_fields(pending_suggestions)

    pending_clean_text = st.session_state.pop("manual_pending_clean_text", None)
    if pending_clean_text is not None:
        st.session_state["manual_job_description"] = pending_clean_text


def render_manual_add_extract_tab() -> None:
    """Render the compact add/extract workflow for manual jobs."""
    prepare_manual_job_session_state()
    render_manual_cleanup_controls()
    cleanup_message = st.session_state.pop("manual_cleanup_message", "")
    if cleanup_message:
        st.success(cleanup_message)

    # Two-column layout: the core workflow stays in the wide left column while
    # At a Glance remains a stable helper panel in the narrow right column.
    left_col, right_col = st.columns([0.72, 0.28], gap="large")
    uploaded_files: list[Any] = []
    current_suggestions: dict[str, Any] = {}

    with left_col:
        st.markdown("**Add / Extract Job**")
        st.caption("Upload a job screenshot/PDF, or paste the job description below. You can edit extracted text before saving.")
        uploaded_files = st.file_uploader(
            "Upload job file",
            type=["png", "jpg", "jpeg", "webp", "pdf", "txt", "md"],
            key=f"manual_upload_{st.session_state.get('manual_upload_key_suffix', 0)}",
            accept_multiple_files=True,
        )
        uploaded_files = uploaded_files or []

        if uploaded_files:
            st.caption("Selected uploads: " + ", ".join(f"`{uploaded_file.name}`" for uploaded_file in uploaded_files))

        if uploaded_files and st.button("Extract Text from Upload"):
            clear_manual_state_for_new_extraction()
            st.session_state["manual_source_upload_filenames"] = [uploaded_file.name for uploaded_file in uploaded_files]
            st.session_state["manual_last_extracted_upload_signature"] = " | ".join(
                f"{uploaded_file.name}:{uploaded_file.size}" for uploaded_file in uploaded_files
            )
            raw_text, cleaned_text, messages, filenames, reports = combine_upload_extraction_results(uploaded_files)
            if messages:
                st.warning(messages)
            st.session_state["manual_extraction_reports"] = reports
            if cleaned_text:
                st.session_state["manual_job_description"] = cleaned_text
                st.session_state["manual_extracted_text"] = cleaned_text
                st.session_state["manual_raw_extracted_text"] = raw_text
                st.session_state["manual_cleaned_extracted_text"] = cleaned_text
                st.session_state["manual_source_upload_filenames"] = filenames
                suggestions = parse_job_description_suggestions(
                    cleaned_text,
                    manual_source_metadata_from_reports(reports),
                )
                st.session_state["manual_parser_suggestions"] = suggestions
                apply_suggestions_to_empty_fields(suggestions)
                st.success("Extracted and cleaned text added to the editable job description.")
            elif not messages:
                st.warning("No text could be extracted. Please paste the job description manually.")

        if st.session_state.get("manual_job_description") and st.button("Clean OCR Text"):
            cleaned_text = clean_extracted_job_text(st.session_state.get("manual_job_description", ""))
            st.session_state["manual_pending_clean_text"] = cleaned_text
            st.session_state["manual_cleaned_extracted_text"] = cleaned_text
            st.rerun()
        render_extraction_reports(st.session_state.get("manual_extraction_reports", []) or [])

        current_text = st.session_state.get("manual_job_description", "")
        current_suggestions = current_manual_suggestions(current_text)
        apply_suggestions_to_empty_fields(current_suggestions)

        st.markdown("**Review and Save**")
        with st.form("manual_job_form"):
            row1_left, row1_right = st.columns(2)
            with row1_left:
                company = st.text_input("Company name", key="manual_company")
            with row1_right:
                title = st.text_input("Job title", key="manual_title")

            row2_left, row2_right = st.columns(2)
            with row2_left:
                location = st.text_input("Location", key="manual_location")
            with row2_right:
                source = st.selectbox("Job source", SOURCE_OPTIONS, key="manual_source")

            row3_left, row3_right = st.columns(2)
            with row3_left:
                url = st.text_input("Job URL", key="manual_url")
            with row3_right:
                salary_range = st.text_input("Salary range, optional", key="manual_salary_range")

            row4_left, row4_right = st.columns(2)
            with row4_left:
                visa_note = st.text_input("Work authorization / visa note, optional", key="manual_visa_note")
            with row4_right:
                status = st.selectbox("Status", STATUS_OPTIONS, key="manual_status")

            notes = st.text_area("Notes", height=90, key="manual_notes")
            job_description = st.text_area(
                "Full job description",
                height=300,
                key="manual_job_description",
                placeholder="Paste the full job description here, or extract text from an upload above.",
            )

            normalized_title = normalize_job_title(title)
            current_suggestions = current_manual_suggestions(job_description)
            quality_warnings = []
            if any([company.strip(), title.strip(), location.strip(), url.strip(), job_description.strip()]):
                quality_warnings = job_description_quality_warnings(
                    company=company,
                    title=normalized_title,
                    location=location,
                    url=url,
                    job_description=job_description,
                )
            if quality_warnings:
                with st.expander("Save-time quality checks", expanded=True):
                    for warning in quality_warnings:
                        st.warning(warning)

            submitted = st.form_submit_button("Save Target Job")

    with right_col:
        with st.container(border=True):
            render_compact_at_a_glance(
                current_suggestions,
                job_description=st.session_state.get("manual_job_description", ""),
                reports=st.session_state.get("manual_extraction_reports", []) or [],
            )

    if not submitted:
        return

    if not normalized_title:
        st.error("Job title is required.")
    elif not job_description.strip():
        st.error("Job description is required. Paste text manually or extract it from an upload.")
    elif not is_valid_url(url):
        st.error("Enter a valid http(s) Job URL, or leave it blank.")
    elif duplicate_manual_job_exists(company, normalized_title, url):
        st.error("Duplicate target job found with the same company, title, and URL.")
    else:
        try:
            upload_files = [(uploaded_file.name, uploaded_file.getvalue()) for uploaded_file in uploaded_files]
            record = save_manual_job(
                company=company,
                title=normalized_title,
                location=location,
                source=source,
                url=url,
                salary_range=salary_range,
                visa_note=visa_note,
                status=status,
                notes=notes,
                job_description=job_description,
                extracted_text=st.session_state.get("manual_extracted_text", ""),
                raw_extracted_text=st.session_state.get("manual_raw_extracted_text", ""),
                cleaned_extracted_text=st.session_state.get("manual_cleaned_extracted_text", ""),
                parser_suggestions=current_suggestions,
                upload_files=upload_files,
            )
            st.session_state["manual_generate_selected"] = manual_record_label(record)
            st.success("Target job saved. Continue to Generate Package when you are ready.")
            st.info("Open the Generate Package tab next. This saved job will be preselected there.")
            if SHOW_DEBUG_UI:
                with st.expander("Advanced: raw Markdown path", expanded=False):
                    st.write(f"Saved Markdown: `{record['markdown_path']}`")
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not save target job: {error}")


def render_saved_manual_jobs_tab() -> None:
    """Render compact saved manual job table and edit controls."""
    st.caption("Review saved targets, confirm company details, and update status.")
    records = sorted_manual_records()
    if not records:
        st.info("No target jobs saved yet.")
        return

    st.dataframe(
        [
            {
                "Company": record.get("company", ""),
                "Company status": verification_status_label(record),
                "Job title": display_title_from_value(record.get("title"), fallback="Sample Job"),
                "Location": normalize_location(str(record.get("location", ""))),
                "Source": record.get("source", ""),
                "Status": record.get("status", ""),
                "Created date": str(record.get("created_at", ""))[:10],
            }
            for record in records
        ],
        width="stretch",
        hide_index=True,
    )

    selected_record = select_manual_record(records, key="manual_saved_selected")
    if selected_record is None:
        return

    record_id = str(selected_record["id"])
    selected_url = str(selected_record.get("url", "") or "").strip()
    summary_left, summary_right = st.columns([0.68, 0.32], gap="large")
    with summary_left:
        st.markdown(f"**{selected_record.get('company', '') or '-'}**")
        st.write(display_title_from_value(selected_record.get("title"), fallback="Sample Job"))
        st.caption(
            f"{normalize_location(str(selected_record.get('location', ''))) or '-'} | "
            f"{selected_record.get('source', '') or '-'}"
        )
    with summary_right:
        st.write(f"Status: {selected_record.get('status', '-')}")
        st.write(f"Company: {verification_status_label(selected_record)}")
        if is_valid_url(selected_url):
            st.link_button("Open Job URL", selected_url, width="stretch")

    overview_tab, verification_tab, jd_tab, notes_tab = st.tabs(
        ["Overview", "Verification", "Full Job Description", "Notes / Status"]
    )
    with overview_tab:
        st.write(f"Company: {selected_record.get('company', '') or '-'}")
        st.write(f"Role: {display_title_from_value(selected_record.get('title'), fallback='Sample Job')}")
        st.write(f"Location: {normalize_location(str(selected_record.get('location', ''))) or '-'}")
        st.write(f"Source: {selected_record.get('source', '') or '-'}")
        st.write(f"Status: {selected_record.get('status', '-')}")
        notes = str(selected_record.get("notes", "") or "").strip()
        if notes:
            st.caption(notes)
        generate_package_for_manual_record(selected_record, button_key=f"manual_saved_generate_{record_id}")
    with verification_tab:
        render_manual_company_confirmation(selected_record, key_prefix=f"manual_saved_{record_id}")
    with jd_tab:
        render_manual_record_long_details(selected_record)
    with notes_tab:
        edit_left, edit_right = st.columns([1, 2])
        with edit_left:
            current_status = str(selected_record.get("status", "Saved"))
            status_index = STATUS_OPTIONS.index(current_status) if current_status in STATUS_OPTIONS else 0
            edited_status = st.selectbox(
                "Edit status",
                STATUS_OPTIONS,
                index=status_index,
                key=f"manual_edit_status_{record_id}",
            )
        with edit_right:
            edited_notes = st.text_area(
                "Edit notes",
                value=str(selected_record.get("notes", "")),
                height=100,
                key=f"manual_edit_notes_{record_id}",
            )
        if st.button("Update Saved Target Job", key=f"manual_update_{record_id}"):
            updated = update_manual_job(record_id, status=edited_status, notes=edited_notes)
            if updated:
                st.success("Saved target job updated.")
                st.rerun()
            else:
                st.error("Could not find that target job record.")


def render_manual_generate_package_tab() -> None:
    """Render package generation for a selected saved manual job."""
    st.caption("Choose a saved job and generate the tailored resume, cover letter, and analysis package.")
    selected_record = select_manual_record(sorted_manual_records(), key="manual_generate_selected")
    if selected_record is None:
        return
    st.write(f"Company: {selected_record.get('company', '')}")
    st.write(f"Company status: {verification_status_label(selected_record)}")
    st.write(f"Job title: {display_title_from_value(selected_record.get('title'), fallback='Sample Job')}")
    st.write(f"Location: {normalize_location(str(selected_record.get('location', ''))) or '-'}")
    generate_package_for_manual_record(selected_record, button_key=f"manual_generate_{selected_record['id']}")
    if st.session_state.get("manual_generated_summary"):
        with st.expander("Latest generated package output", expanded=False):
            st.json(st.session_state["manual_generated_summary"])
    if st.session_state.get("manual_generated_backend_output"):
        with st.expander("Latest technical output (advanced)", expanded=False):
            st.text(st.session_state["manual_generated_backend_output"])
    if st.session_state.get("manual_generated_error"):
        st.error(st.session_state["manual_generated_error"])


def render_manual_debug_tab() -> None:
    """Render collapsed OCR and parser debugging details."""
    st.info(
        "This section is for troubleshooting OCR, PDF extraction, and parsing issues. "
        "Most users do not need it during normal job entry."
    )
    with st.expander("Cleanup and current state", expanded=False):
        st.write("Current uploaded file names:")
        st.json(st.session_state.get("manual_source_upload_filenames", []) or [])
        st.write(f"Current upload signature: {st.session_state.get('manual_last_extracted_upload_signature', '-')}")
        st.write(
            "Parser suggestion source: "
            + ("current job description text" if st.session_state.get("manual_parser_suggestions") else "-")
        )
        st.write(f"Extracted text source: {st.session_state.get('manual_last_extracted_upload_signature', '-')}")
        st.write(f"Selected target job for generation: {st.session_state.get('manual_generate_selected', '-')}")
        st.write(f"Last cleanup timestamp: {st.session_state.get('manual_last_cleanup_timestamp', '-')}")
    with st.expander("Advanced: raw extracted text", expanded=False):
        st.text(st.session_state.get("manual_raw_extracted_text", ""))
    with st.expander("Advanced: cleaned extracted text", expanded=False):
        st.text(st.session_state.get("manual_cleaned_extracted_text", ""))
    with st.expander("Advanced: parser suggestions", expanded=False):
        st.json(st.session_state.get("manual_parser_suggestions", {}) or {})
    with st.expander("Advanced: parser evidence", expanded=False):
        suggestions = st.session_state.get("manual_parser_suggestions", {}) or {}
        for label, key in [
            ("Company", "company_evidence"),
            ("Job title", "job_title_evidence"),
            ("Location", "location_evidence"),
            ("Visa / work authorization", "visa_evidence"),
            ("Employment type", "employment_type_evidence"),
        ]:
            evidence = suggestions.get(key)
            if evidence:
                st.write(f"{label}: {evidence}")
    with st.expander("Advanced: extraction reports", expanded=False):
        st.json(st.session_state.get("manual_extraction_reports", []) or [])


def manual_job_target_tab() -> None:
    """Render the manual target job workflow in compact sub-tabs."""
    render_page_header("Add Target Job", "Save a role locally, then review fit and generate a package.")
    if demo_mode_enabled():
        st.info("Demo workspace uses bundled sample jobs. Select Personal to save a new target job locally.")
        return

    # Tab order follows the daily workflow: enter a job, generate a package, then
    # use saved jobs as history/library. Debug UI stays internal by default.
    tab_labels = ["Add / Extract Job", "Generate Package", "Saved Target Jobs"]
    if SHOW_DEBUG_UI:
        tab_labels.append("Advanced / Debug")
    tabs = st.tabs(tab_labels)
    tab_add, tab_generate, tab_saved = tabs[:3]
    with tab_add:
        render_manual_add_extract_tab()
    with tab_generate:
        render_manual_generate_package_tab()
    with tab_saved:
        render_saved_manual_jobs_tab()
    if SHOW_DEBUG_UI:
        with tabs[3]:
            render_manual_debug_tab()


JOB_CARD_METADATA_PREFIXES = {
    "company",
    "role",
    "location",
    "job url",
    "source",
    "source job id",
    "created at",
    "company raw",
    "company normalized",
    "company confidence",
    "company needs review",
    "company evidence",
    "company candidates",
    "company confirmed by user",
    "company confirmed at",
    "canonical job key",
    "first seen at",
    "last seen at",
    "first seen fetch run id",
    "last seen fetch run id",
    "latest fetch run id",
    "fetch run ids",
}


def clean_card_text(value: Any, fallback: str = "-") -> str:
    """Return one-line text that cannot be interpreted as Markdown headings."""
    text = " ".join(str(value or "").split())
    text = re.sub(r"^[#>*_`\\-\\s]+", "", text).strip()
    return text or fallback


def card_html(text: Any, class_name: str) -> str:
    """Render escaped card text with compact product typography."""
    return f'<div class="{class_name}">{html.escape(clean_card_text(text))}</div>'


def is_card_metadata_line(line: str) -> bool:
    """Return True for saved-job metadata lines that should not appear in cards."""
    if ":" not in line:
        return False
    key = line.split(":", 1)[0].strip().lower().lstrip("#").strip()
    return key in JOB_CARD_METADATA_PREFIXES


def clean_job_card_snippet(preview: str, limit: int = 180) -> str:
    """Build a short card snippet from job body text, not saved metadata."""
    body_lines: list[str] = []
    in_job_description = False
    for raw_line in str(preview or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        normalized = line.lower().strip("# ").strip()
        if normalized in {"job description", "description", "about the role"}:
            in_job_description = True
            continue
        if is_card_metadata_line(line):
            continue
        if line.startswith("#") and not in_job_description:
            continue
        line = re.sub(r"^[#>*_`\\-\\s]+", "", line).strip()
        if line:
            body_lines.append(line)

    snippet = " ".join(" ".join(body_lines).split())
    return snippet[:limit].rstrip()


def build_job_snippet(job: dict[str, Any], limit: int = 240) -> str:
    """Return a short readable snippet for a job card."""
    warnings_text = str(job.get("warnings_text", "") or "").strip()
    if warnings_text and warnings_text != "-":
        return clean_card_text(f"Review note: {warnings_text}")
    return clean_job_card_snippet(str(job.get("preview", "") or ""), limit)


def key_requirements_from_text(job_text: str) -> list[str]:
    """Extract a small requirements list from saved Markdown text."""
    requirements: list[str] = []
    in_requirements = False
    for line in job_text.splitlines():
        cleaned = line.strip()
        lower_cleaned = cleaned.lower().strip("#:")
        if any(marker in lower_cleaned for marker in ["requirements", "qualifications", "what you need"]):
            in_requirements = True
            continue
        if in_requirements and cleaned.startswith("#"):
            break
        if in_requirements and cleaned.startswith(("-", "*")):
            requirements.append(cleaned.lstrip("-* ").strip())
        if len(requirements) >= 6:
            break
    return requirements


def load_fit_resume_text() -> str:
    """Read resume text for fit analysis without requiring private data in demo mode."""
    path = current_workspace().resume_source_path
    if path is None or not path.exists():
        return ""
    return read_text_file(path)


def structured_fit_analysis(job: dict[str, Any], job_text: str) -> dict[str, Any]:
    """Return the same canonical analysis already attached to the loaded job."""
    existing = job.get("analysis_result")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    return analyze_job_for_dashboard(job, job_text)


def sanitize_fit_text(value: Any) -> str:
    """Hide local implementation filenames from user-facing fit text."""
    text = str(value)
    replacements = {
        "`resume_source.md`": "the candidate profile",
        "resume_source.md": "the candidate profile",
        "resume_source.example.md": "the demo candidate profile",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def render_keyword_list(label: str, keywords: list[Any], empty_text: str = "None") -> None:
    """Render keyword chips as short text."""
    cleaned = [sanitize_fit_text(keyword) for keyword in keywords if str(keyword).strip()]
    st.write(f"{label}: {', '.join(cleaned) if cleaned else empty_text}")


def render_fit_analysis_sections(job: dict[str, Any], job_text: str) -> None:
    """Render explainable job fit analysis for the selected job."""
    analysis = structured_fit_analysis(job, job_text)
    presentation = build_fit_presentation(apply_canonical_analysis(job, analysis))
    terms = presentation["terms"]
    st.markdown("**Fit Analysis**")
    eligibility = dict(analysis.get("eligibility", {}))
    scoring_confidence = dict(analysis.get("confidence", {}))
    level = confidence_level(scoring_confidence)
    score = analysis.get("score")
    if level == "low" or not analysis.get("analysis_available"):
        st.write(f"Role Fit: {presentation['role_fit']}")
    else:
        st.write(f"Role Fit Score: {int(score)}/100")
    st.write(f"Eligibility: {str(eligibility.get('status', 'manual_review')).replace('_', ' ').title()}")
    st.write(f"Scoring Confidence: {level.title()}")
    st.write(f"Recommendation: {analysis.get('recommendation', 'Manual Review')}")
    st.write(f"Primary reason: {sanitize_fit_text(analysis.get('main_reason', 'Review manually.'))}")
    st.write(f"Primary risk: {sanitize_fit_text(analysis.get('main_risk', 'Review the full job description manually.'))}")

    if level == "low":
        st.write(f"Recognized requirements: {terms['active_requirement_count']}")
        st.write(
            f"Matched requirements: {terms['matched_requirement_count']} of "
            f"{terms['active_requirement_count']}"
        )
        if score is not None and terms["active_requirement_count"]:
            st.caption(f"Coverage among recognized requirements: {int(score)}%")

    st.markdown("**Recognized Requirements**")
    if not terms["active_requirement_count"]:
        st.write("Requirements could not be extracted reliably.")
        st.write("Review the full job description manually.")
    else:
        render_keyword_list("Matched required terms", terms["matched_required"], empty_text="None")
        render_keyword_list("Matched preferred terms", terms["matched_preferred"], empty_text="None")
        render_keyword_list("Missing required terms", terms["missing_required"], empty_text="No missing required terms among the recognized requirements.")
        render_keyword_list("Missing preferred terms", terms["missing_preferred"], empty_text="No preferred terms were recognized or missing.")
        render_keyword_list("Partial required matches", terms["partial_required"], empty_text="None")
        render_keyword_list("Partial preferred matches", terms["partial_preferred"], empty_text="None")

    st.markdown("**Matched Strengths**")
    for item in list(analysis.get("matched_strengths", []))[:6] or ["Not available yet."]:
        st.write(f"- {sanitize_fit_text(item)}")

    st.markdown("**Missing or Weak Areas**")
    for item in list(analysis.get("weak_areas", []))[:6] or ["Not available yet."]:
        st.write(f"- {sanitize_fit_text(item)}")

    st.markdown("**Resume Suggestions**")
    for item in list(analysis.get("resume_suggestions", []))[:5] or ["Not available yet."]:
        st.write(f"- {sanitize_fit_text(item)}")

    st.markdown("**Evidence**")
    jd_evidence = list(analysis.get("jd_evidence", []))
    profile_evidence = list(analysis.get("profile_evidence", []))
    if jd_evidence:
        st.caption("Job description evidence")
        for item in jd_evidence[:3]:
            st.write(f"- {sanitize_fit_text(item)}")
    if profile_evidence and not demo_mode_enabled():
        st.caption("Resume/profile evidence")
        for item in profile_evidence[:3]:
            st.write(f"- {sanitize_fit_text(item)}")
    elif profile_evidence:
        st.caption("Demo profile evidence")
        for item in profile_evidence[:3]:
            st.write(f"- {sanitize_fit_text(item)}")

    if analysis.get("raw_analysis"):
        with st.expander("Full analysis", expanded=False):
            st.markdown(sanitize_fit_text(analysis["raw_analysis"]))


def render_generation_success(summary: dict[str, Any]) -> None:
    """Show a user-facing package generation checklist."""
    remember_generated_package(summary)
    try:
        database_path = current_workspace().tracker_database_path
        if database_path is not None:
            run_with_captured_output(update_status, int(summary["tracker_id"]), "ready", database_path)
    except Exception:
        pass

    st.success("Application package generated.")
    st.write("Resume generated")
    if summary.get("resume_docx_path"):
        st.write("Resume DOCX generated")
    st.write("Cover letter generated")
    if summary.get("cover_letter_docx_path"):
        st.write("Cover letter DOCX generated")
    st.write("Match report generated")
    if summary.get("tailoring_notes_path") or summary.get("cover_letter_notes_path"):
        st.write("Internal notes generated")
    st.info("Open the Application Package page to preview and export deliverables.")


def generate_review_job_package(
    job: dict[str, Any],
    button_key: str,
    primary: bool = False,
    label: str = "Generate Application Package",
) -> None:
    """Generate an application package for a reviewed job using existing logic."""
    selected_path = job["path"]
    metadata = parse_job_metadata(selected_path)
    company = str(job.get("company_normalized") or metadata.get("company") or job.get("company", "")).strip()
    role = str(metadata.get("role") or job.get("role", "")).strip()
    location = normalize_location(str(metadata.get("location") or job.get("normalized_location") or job.get("location", "")))
    job_url = str(metadata.get("job_url") or job.get("job_url", "")).strip()

    if not st.button(label, key=button_key, type="primary" if primary else "secondary", width="stretch"):
        return
    if not all([company, role, location, job_url]):
        st.error("Company, role, location, and job URL are required before generating a package.")
        return

    latest_company_fields = verification_from_markdown(selected_path)
    if normalize_company_name(company) != str(latest_company_fields.get("company_normalized", "")):
        st.error("Confirm the company name in the detail view before generating a cover letter.")
        return
    if not company_generation_allowed(latest_company_fields):
        st.error(
            "Company name needs confirmation before generating a cover letter. "
            "This prevents using the wrong company name in your application."
        )
        return

    try:
        summary, output = run_with_captured_output(
            create_application_package,
            job_description_path=selected_path,
            workspace=current_workspace(),
            company=company,
            role=role,
            location=location,
            job_url=job_url,
        )
        render_generation_success(summary)
        st.write(f"Overall score: {summary['match_score']}/100")
        st.write(f"Recommendation: {summary['recommendation']}")
        if summary.get("uk_review_notes"):
            st.warning("UK work authorization review")
            for note in summary["uk_review_notes"]:
                st.write(f"- {note}")
        if summary.get("export_warnings"):
            with st.expander("Validation warnings", expanded=False):
                for warning in summary["export_warnings"]:
                    st.write(f"- {warning}")
        if SHOW_DEBUG_UI and output:
            with st.expander("Advanced: package generation output", expanded=False):
                st.text(output)
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not generate the application package: {error}")


def set_review_job_selection(job: dict[str, Any], focus: str = "Overview") -> None:
    """Select a review job and focus the detail panel."""
    st.session_state["selected_review_job_path"] = str(job["path"])
    st.session_state["selected_review_job_label"] = job["label"]
    st.session_state["selected_review_tab"] = focus


def resolve_review_job_selection(
    shortlist: list[dict[str, Any]],
    selected_label: object,
    selected_path: object,
) -> dict[str, Any]:
    """Return a visible selected job, falling back safely from stale state."""
    jobs_by_label = {job["label"]: job for job in shortlist}
    jobs_by_path = {str(job["path"]): job for job in shortlist}
    if selected_label in jobs_by_label:
        return jobs_by_label[selected_label]
    if selected_path in jobs_by_path:
        return jobs_by_path[selected_path]
    return shortlist[0]


def render_review_action_buttons(job: dict[str, Any], tracker_rows: list[dict[str, Any]], key_prefix: str) -> None:
    """Render selected-job actions in the detail panel."""
    action_package, action_fit, action_track = st.columns([0.44, 0.28, 0.28])
    with action_package:
        if st.button("Prepare Package", key=f"{key_prefix}_package", type="primary", width="stretch"):
            set_review_job_selection(job, "Package")
            st.rerun()
    with action_fit:
        if st.button("View Fit", key=f"{key_prefix}_fit", width="stretch"):
            set_review_job_selection(job, "Fit")
            st.rerun()
    with action_track:
        if demo_mode_enabled():
            st.caption("Tracker disabled in Demo workspace.")
        elif st.button("Track", key=f"{key_prefix}_track", width="stretch"):
            try:
                tracker_id, output = save_job_to_tracker(job)
                st.success(f"Saved to tracker #{tracker_id}.")
                if SHOW_DEBUG_UI and output:
                    with st.expander("Advanced: tracker output", expanded=False):
                        st.text(output)
            except Exception as error:  # noqa: BLE001
                st.error(str(error))
    if not demo_mode_enabled():
        with st.expander("More actions", expanded=False):
            if st.button("Ignore", key=f"{key_prefix}_ignore"):
                try:
                    tracker_id, output = mark_job_not_interested(job, tracker_rows)
                    st.success(f"Marked tracker #{tracker_id} as not interested.")
                    if SHOW_DEBUG_UI and output:
                        with st.expander("Advanced: tracker output", expanded=False):
                            st.text(output)
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))


def render_job_result_cards(jobs: list[dict[str, Any]], tracker_rows: list[dict[str, Any]]) -> None:
    """Render shortlist jobs as compact list items."""
    st.markdown(
        """
        <style>
        .job-card-company {
            font-size: 0.84rem;
            font-weight: 700;
            line-height: 1.2;
        }
        .job-card-role {
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.25;
        }
        .job-card-meta,
        .job-card-status {
            color: rgba(49, 51, 63, 0.72);
            font-size: 0.8rem;
            line-height: 1.25;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    for index, job in enumerate(jobs, start=1):
        file_key = safe_slug(str(job["path"]))
        tracker_status = tracker_status_for_job(job, tracker_rows)
        package_status = job.get("package_status") or package_status_for_job(job, tracker_rows)
        fit_presentation = build_fit_presentation(job)
        with st.container(border=True):
            st.markdown(
                card_html(job["company"], "job-card-company")
                + card_html(get_job_display_title(job), "job-card-role")
                + card_html(
                    f"{job['normalized_location']} | {source_display_name(str(job['source']))}",
                    "job-card-meta",
                )
                + card_html(
                    f"{fit_presentation['card_status']} · {tracker_status} · {package_status}",
                    "job-card-status",
                ),
                unsafe_allow_html=True,
            )

            action_view, action_fit, action_package = st.columns(3)
            with action_view:
                if st.button("Select", key=f"view_details_{file_key}_{index}", width="stretch"):
                    set_review_job_selection(job, "Overview")
                    st.rerun()
            with action_fit:
                if st.button("Fit", key=f"view_fit_{file_key}_{index}", width="stretch"):
                    set_review_job_selection(job, "Fit")
                    st.rerun()
            with action_package:
                if st.button(
                    "Package",
                    key=f"view_package_{file_key}_{index}",
                    width="stretch",
                    disabled=demo_mode_enabled() and package_status != "Demo package",
                ):
                    set_review_job_selection(job, "Package")
                    st.rerun()


def job_descriptions_tab() -> None:
    """Render the job-description review and package generation workflow."""
    render_page_header("Review Jobs", "Use the job inbox to review fit, prepare packages, and track next steps.")
    if demo_mode_enabled():
        st.info(
            "Demo workspace uses fictional, read-only data. "
            "All Jobs is shown by default so you can compare different scoring outcomes."
        )

    all_jobs = load_screened_jobs()

    if not all_jobs:
        st.info("No jobs found yet. Start with Find Jobs or Add Target Job.")
        if st.button("Add Target Job", width="stretch"):
            go_to_page("Add Target Job")
        return

    fetch_runs = load_fetch_runs()
    fetch_runs_by_id = {str(run.get("fetch_run_id", "")): run for run in fetch_runs}
    tracker_rows = [] if demo_mode_enabled() else load_tracker_rows(sort_by="created_at", descending=True)
    is_demo = demo_mode_enabled()
    default_inbox_view = default_review_inbox_view(all_jobs, tracker_rows, demo=is_demo)
    workspace_state_key = "demo" if is_demo else "personal"
    workspace_changed = st.session_state.get("review_workspace_mode") != workspace_state_key
    if workspace_changed:
        st.session_state["review_workspace_mode"] = workspace_state_key
        st.session_state["review_inbox_view"] = default_inbox_view
        if is_demo:
            st.session_state["review_minimum_score"] = 0
            st.session_state["review_hide_hard_red_flags"] = False
            st.session_state["review_hide_degree_required"] = False
            st.session_state["review_hide_current_student_only"] = False

    def clear_review_filters() -> None:
        st.session_state["review_search_text"] = ""
        st.session_state["review_inbox_view"] = default_inbox_view
        st.session_state["review_sort_by"] = "Role Fit high to low"
        st.session_state["selected_region_key"] = "all"
        st.session_state["review_source_filter"] = "all"
        st.session_state["review_recommendation_filter"] = "all"
        st.session_state["review_minimum_score"] = 0 if is_demo else 50
        st.session_state["review_hide_hard_red_flags"] = not is_demo
        st.session_state["review_hide_degree_required"] = not is_demo
        st.session_state["review_hide_current_student_only"] = not is_demo

    def show_all_review_jobs() -> None:
        clear_review_filters()
        st.session_state["review_inbox_view"] = "All Jobs"
        st.session_state["review_minimum_score"] = 0

    inbox_view_options = ["Recommended", "Needs Review", "Package Ready", "Not Tracked", "Ignored", "All Jobs"]
    sort_options = ["Role Fit high to low", "Newest first", "Recommendation", "Company A-Z", "Package status", "Tracker status"]
    st.session_state.setdefault("review_inbox_view", default_inbox_view)
    if st.session_state["review_inbox_view"] not in inbox_view_options:
        st.session_state["review_inbox_view"] = default_inbox_view
    st.session_state.setdefault("review_sort_by", "Role Fit high to low")
    if st.session_state["review_sort_by"] not in sort_options:
        st.session_state["review_sort_by"] = "Role Fit high to low"
    source_options = dynamic_source_options(all_jobs)
    if st.session_state.get("review_source_filter", "all") not in source_options:
        st.session_state["review_source_filter"] = "all"
    recommendation_options = [
        "all",
        "Apply",
        "Apply / Maybe Apply",
        "Maybe Apply",
        "Manual Review",
        "Skip or Low Priority",
        "Skip / Not Eligible",
    ]
    if st.session_state.get("review_recommendation_filter", "all") not in recommendation_options:
        st.session_state["review_recommendation_filter"] = "all"
    st.session_state.setdefault("review_search_text", "")
    st.session_state.setdefault("region_search_query", "")
    st.session_state.setdefault("job_tab_shortlist_limit", DEFAULT_RECOMMENDATION_LIMIT)
    st.session_state.setdefault("review_minimum_score", 50)
    st.session_state.setdefault("review_hide_hard_red_flags", True)
    st.session_state.setdefault("review_hide_degree_required", True)
    st.session_state.setdefault("review_hide_current_student_only", True)

    inbox_view = st.segmented_control(
        "Job inbox view",
        inbox_view_options,
        key="review_inbox_view",
        selection_mode="single",
    ) or default_inbox_view

    search_col, sort_col, count_col = st.columns([0.42, 0.32, 0.26])
    with search_col:
        search_text = st.text_input("Search company or role", key="review_search_text")
    with sort_col:
        sort_by = st.selectbox(
            "Sort by",
            sort_options,
            key="review_sort_by",
        )
    with count_col:
        recommendation_limit = st.slider(
            "Number of recommendations",
            min_value=MIN_RECOMMENDATION_LIMIT,
            max_value=MAX_RECOMMENDATION_LIMIT,
            key="job_tab_shortlist_limit",
            help="How many ranked jobs to display after filtering and duplicate removal.",
        )
    options_by_key = build_region_options(all_jobs)
    with st.expander("Filters", expanded=False):
        filter_row1_left, filter_row1_right = st.columns(2)
        with filter_row1_left:
            region_search_query = st.text_input(
                "Search region",
                placeholder="Beijing, China, Remote",
                key="region_search_query",
            )
        region_option_keys = filtered_region_option_keys(options_by_key, region_search_query)
        if region_search_query.strip() and region_option_keys == ["all"]:
            st.info("No matching regions found.")

        selected_region_key = st.session_state.get("selected_region_key", "all")
        if selected_region_key not in region_option_keys:
            region_option_keys = [selected_region_key, *region_option_keys] if selected_region_key in options_by_key else region_option_keys
        selected_region_index = region_option_keys.index(selected_region_key) if selected_region_key in region_option_keys else 0
        with filter_row1_right:
            selected_region_key = st.selectbox(
                "Region",
                region_option_keys,
                index=selected_region_index,
                key="selected_region_key",
                format_func=lambda key: options_by_key.get(key, options_by_key["all"])["label"],
            )
        selected_region_option = options_by_key.get(selected_region_key, options_by_key["all"])
        save_recent_region_key(selected_region_key)

        filter_row2_left, filter_row2_right = st.columns(2)
        with filter_row2_left:
            source_filter = st.selectbox("Source", source_options, key="review_source_filter")
        with filter_row2_right:
            recommendation_filter = st.selectbox(
                "Recommendation",
                recommendation_options,
                key="review_recommendation_filter",
            )

        filter_row3_left, filter_row3_right = st.columns(2)
        with filter_row3_left:
            minimum_score = st.slider(
                "Minimum Role Fit score",
                min_value=0,
                max_value=100,
                key="review_minimum_score",
            )
        with filter_row3_right:
            st.caption(f"Showing up to {recommendation_limit} recommendations.")
            st.button(
                "Clear filters",
                key="review_clear_filters",
                on_click=clear_review_filters,
                width="stretch",
            )

        flag_left, flag_middle, flag_right = st.columns(3)
        with flag_left:
            hide_hard_red_flags = st.checkbox("Hide hard red flags", key="review_hide_hard_red_flags")
        with flag_middle:
            hide_degree_required = st.checkbox(
                "Hide PhD / Master's required roles",
                key="review_hide_degree_required",
            )
        with flag_right:
            hide_current_student_only = st.checkbox(
                "Hide current-student-only internships",
                key="review_hide_current_student_only",
            )

    filtered_jobs = []
    for job in all_jobs:
        tracker_status = tracker_status_for_job(job, tracker_rows)
        package_status = package_status_for_job(job, tracker_rows)
        if not review_inbox_view_matches(job, inbox_view, tracker_status, package_status):
            continue
        red_flags_text = " | ".join(job["red_flags"])
        if not job_matches_region_option(job, selected_region_option):
            continue
        if source_filter != "all" and source_display_name(str(job["source"])) != source_filter:
            continue
        if recommendation_filter != "all" and job["recommendation"] != recommendation_filter:
            continue
        if job.get("score") is None and minimum_score > 0:
            continue
        if job.get("score") is not None and int(job["score"]) < minimum_score:
            continue
        if hide_hard_red_flags and job["hard_red_flag"]:
            continue
        if hide_degree_required and any("PhD" in flag or "Master" in flag or "Graduate degree" in flag for flag in job["red_flags"]):
            continue
        if hide_current_student_only and any("enrolled" in flag.lower() or "school" in flag.lower() or "penultimate" in flag.lower() for flag in job["red_flags"]):
            continue
        if search_text.strip():
            needle = search_text.strip().lower()
            if needle not in f"{job['company']} {job['role']} {get_job_display_title(job)}".lower():
                continue
        fetch_run = fetch_runs_by_id.get(str(job.get("latest_fetch_run_id", "")), {})
        job["fetch_run_date"] = str(fetch_run.get("created_at", job.get("last_seen_at", "")))
        job["new_label"] = "Saved job"
        job["red_flags_text"] = red_flags_text or "-"
        job["warnings_text"] = " | ".join(job["warnings"]) or "-"
        job["tracker_status"] = tracker_status
        job["package_status"] = package_status
        filtered_jobs.append(job)

    filtered_jobs = sorted_review_jobs(filtered_jobs, sort_by)
    shortlist = filtered_jobs[:recommendation_limit]
    filter_summary = f"Role Fit {minimum_score}+"
    if selected_region_key != "all":
        filter_summary += f" · {region_label(selected_region_option, include_count=False)}"
    if source_filter != "all":
        filter_summary += f" · {source_filter}"
    if recommendation_filter != "all":
        filter_summary += f" · {recommendation_filter}"
    sort_summary = {
        "Role Fit high to low": "Role Fit",
        "Newest first": "newest",
        "Recommendation": "recommendation",
        "Company A-Z": "company",
        "Package status": "package status",
        "Tracker status": "tracker status",
    }.get(sort_by, sort_by.lower())
    summary_parts = [
        f"{len(shortlist)} jobs shown",
        inbox_view,
        f"Sorted by {sort_summary}",
        filter_summary,
    ]

    if not shortlist:
        st.info("No jobs match the current filters.")
        empty_left, empty_middle, empty_right = st.columns(3)
        with empty_left:
            st.button(
                "Clear filters",
                key="review_empty_clear_filters",
                on_click=clear_review_filters,
                width="stretch",
            )
        with empty_middle:
            st.button("Show All Jobs", on_click=show_all_review_jobs, width="stretch")
        with empty_right:
            if st.button("Add Target Job", width="stretch"):
                go_to_page("Add Target Job")
        return

    if len(shortlist) == 1 and len(all_jobs) > 1:
        st.caption("Only 1 job shown. Clear filters or switch to All Jobs to see more.")
        hint_left, hint_right = st.columns(2)
        with hint_left:
            st.button("Clear filters", on_click=clear_review_filters, width="stretch")
        with hint_right:
            st.button("Show All Jobs", on_click=show_all_review_jobs, width="stretch")

    shortlist_paths = [str(job["path"]) for job in shortlist]
    selected_label_state = st.session_state.get("selected_review_job_label")
    selected_path_state = st.session_state.get("selected_review_job_path", shortlist_paths[0])
    selected_job = resolve_review_job_selection(shortlist, selected_label_state, selected_path_state)
    if str(selected_job["path"]) != selected_path_state and selected_job["label"] != selected_label_state:
        set_review_job_selection(shortlist[0], st.session_state.get("selected_review_tab", "Overview"))
    st.session_state["selected_review_job_path"] = str(selected_job["path"])
    st.session_state["selected_review_job_label"] = selected_job["label"]
    selected_path = selected_job["path"]
    selected_text = read_text_file(selected_path)
    selected_tracker_row = tracker_row_for_job(selected_job, tracker_rows)

    sections = ["Overview", "Fit", "JD", "Package"]
    if st.session_state.get("selected_review_tab") not in sections:
        st.session_state["selected_review_tab"] = "Overview"

    left_col, right_col = st.columns([0.42, 0.58], gap="large")
    with left_col:
        st.caption(" · ".join(summary_parts))
        render_job_result_cards(shortlist, tracker_rows)
        with st.expander("Compact table view", expanded=False):
            st.dataframe(
                [
                    {
                        "Company": job["company"],
                        "Role": get_job_display_title(job),
                        "Location": job["normalized_location"],
                        "Source": source_display_name(str(job["source"])),
                        "Role Fit": build_fit_presentation(job)["role_fit"],
                        "Recommendation": job["recommendation"],
                        "Tracker": job.get("tracker_status") or tracker_status_for_job(job, tracker_rows),
                        "Package": job.get("package_status") or package_status_for_job(job, tracker_rows),
                    }
                    for job in shortlist
                ],
                width="stretch",
                hide_index=True,
            )

    with right_col:
        tracker_status = tracker_status_for_job(selected_job, tracker_rows)
        selected_presentation = build_fit_presentation(selected_job)
        header_left, header_right = st.columns([0.72, 0.28])
        with header_left:
            st.markdown(f"**{selected_job['company']}**")
            st.write(get_job_display_title(selected_job))
            st.caption(
                f"{selected_job['normalized_location']} | "
                f"{source_display_name(str(selected_job['source']))}"
            )
        with header_right:
            if confidence_level(selected_job.get("confidence")) == "low":
                st.caption("Role Fit")
                st.write("Insufficient evidence")
            elif selected_job.get("score") is not None:
                st.metric("Role Fit Score", f"{selected_job['score']}/100")
            else:
                st.caption("Role Fit: Not available")
            st.caption(selected_job["recommendation"])
            st.caption(f"Tracker: {tracker_status}")
            st.caption(f"Package: {selected_job.get('package_status') or package_status_for_job(selected_job, tracker_rows)}")

        selected_section = st.segmented_control(
            "Detail section",
            sections,
            default=st.session_state["selected_review_tab"],
            selection_mode="single",
            label_visibility="collapsed",
        )
        selected_section = selected_section or "Overview"
        if selected_section != st.session_state["selected_review_tab"]:
            st.session_state["selected_review_tab"] = selected_section
            st.rerun()

        if selected_section == "Overview":
            snippet = build_job_snippet(selected_job)
            if snippet:
                st.caption(snippet)
            selected_analysis = dict(selected_job.get("analysis_result", {}))
            st.write(f"Main reason: {selected_analysis.get('main_reason', selected_presentation['card_status'])}")
            main_risk = str(selected_analysis.get("main_risk", "")) or (
                selected_job["red_flags_text"] if selected_job["red_flags_text"] != "-" else selected_job["warnings_text"]
            )
            if main_risk != "-":
                st.write(f"Main risk: {main_risk}")
            render_review_action_buttons(
                selected_job,
                tracker_rows,
                key_prefix=f"overview_actions_{safe_slug(str(selected_path))}",
            )

            detailed_notes = " | ".join(
                note
                for note in [selected_job.get("red_flags_text", "-"), selected_job.get("warnings_text", "-")]
                if note and note != "-"
            )
            if detailed_notes and detailed_notes != main_risk:
                with st.expander("Review notes", expanded=False):
                    st.write(detailed_notes)

        elif selected_section == "Fit":
            render_fit_analysis_sections(selected_job, selected_text)
            requirements = key_requirements_from_text(selected_text)
            if requirements:
                st.markdown("**Key Requirements**")
                for requirement in requirements:
                    st.write(f"- {requirement}")
            package_dir = package_dir_for_job(selected_job, tracker_rows)
            if package_dir:
                suggestions = load_package_notes(package_dir)
                if suggestions:
                    with st.expander("Resume / Cover Letter suggestions", expanded=False):
                        st.markdown(sanitize_fit_text(suggestions))

        elif selected_section == "JD":
            if selected_job.get("job_url"):
                st.link_button("Open Job URL", str(selected_job["job_url"]))
            st.caption(f"Source: {source_display_name(str(selected_job['source']))}")
            st.text(selected_text or selected_job["preview"])
            if SHOW_DEBUG_UI:
                with st.expander("Advanced: job metadata", expanded=False):
                    st.write(f"Markdown path: `{relative_path(selected_path)}`")
                    st.write(f"New status: {selected_job.get('new_label', '-')}")
                    st.write(f"First seen: {selected_job.get('first_seen_at', '-')}")
                    st.write(f"Last seen: {selected_job.get('last_seen_at', '-')}")
                    st.write(f"Search run date: {selected_job.get('fetch_run_date', '-')}")
                    if selected_tracker_row:
                        st.write(f"Tracker id: {selected_tracker_row['id']}")

        elif selected_section == "Package":
            if demo_mode_enabled():
                st.info("Demo workspace does not generate new files. Open Application Package to view the sanitized sample package.")
                return

            metadata = parse_job_metadata(selected_path)
            file_key = safe_slug(str(selected_path))

            with st.expander("Package options", expanded=True):
                st.markdown("**Company verification**")
                selected_company_fields = render_markdown_company_confirmation(
                    selected_path,
                    key_prefix=f"job_desc_{safe_slug(str(selected_path))}",
                )
                default_company = str(selected_company_fields.get("company_normalized") or metadata.get("company", ""))
                default_role = metadata.get("role", "")
                default_location = normalize_location(metadata.get("location", ""))
                default_job_url = metadata.get("job_url", "")

                st.write(f"Confirmed company: {default_company or '-'}")
                row1_left, row1_right = st.columns(2)
                with row1_left:
                    company = st.text_input("Editable company", value=default_company, key=f"company_{file_key}")
                    location = st.text_input("Location override", value=default_location, key=f"location_{file_key}")
                with row1_right:
                    role = st.text_input("Role override", value=default_role, key=f"role_{file_key}")
                    job_url = st.text_input("Job URL override", value=default_job_url, key=f"job_url_{file_key}")

                if st.button("Generate Application Package", key=f"generate_{file_key}", type="primary"):
                    if not all([company.strip(), role.strip(), location.strip(), job_url.strip()]):
                        st.error("Please fill in company, role, location, and job URL before generating the package.")
                        return
                    latest_company_fields = verification_from_markdown(selected_path)
                    if normalize_company_name(company) != str(latest_company_fields.get("company_normalized", "")):
                        st.error("Confirm the edited company name before generating a cover letter.")
                        return
                    if not company_generation_allowed(latest_company_fields):
                        st.error(
                            "Company name needs confirmation before generating a cover letter. "
                            "This prevents using the wrong company name in your application."
                        )
                        return

                    try:
                        summary, output = run_with_captured_output(
                            create_application_package,
                            job_description_path=selected_path,
                            workspace=current_workspace(),
                            company=company.strip(),
                            role=role.strip(),
                            location=location.strip(),
                            job_url=job_url.strip(),
                        )
                        render_generation_success(summary)
                        st.write(f"Overall score: {summary['match_score']}/100")
                        st.write(f"Recommendation: {summary['recommendation']}")
                        st.write(f"Tracker id: {summary['tracker_id']}")
                        st.write(f"Resume DOCX: `{relative_path(summary['resume_docx_path'])}`")
                        st.write(f"Cover letter DOCX: `{relative_path(summary['cover_letter_docx_path'])}`")
                        if summary.get("uk_review_notes"):
                            st.warning("UK work authorization review")
                            for note in summary["uk_review_notes"]:
                                st.write(f"- {note}")
                        if summary.get("export_warnings"):
                            with st.expander("Validation warnings", expanded=False):
                                for warning in summary["export_warnings"]:
                                    st.write(f"- {warning}")
                        if SHOW_DEBUG_UI and output:
                            with st.expander("Advanced: package generation output", expanded=False):
                                st.text(output)
                    except Exception as error:  # noqa: BLE001
                        st.error(f"Could not generate the application package: {error}")


def tracker_tab() -> None:
    """Render the local SQLite tracker table and record actions."""
    render_page_header("Tracker")
    if demo_mode_enabled():
        st.info("Demo workspace does not read or update the Personal tracker database.")
        return

    status_options = sorted(VALID_STATUSES)
    selected_statuses = st.multiselect("Status filter", status_options, default=status_options)
    minimum_score = st.slider("Minimum match score", min_value=0, max_value=100, value=0)
    company_search = st.text_input("Company search", value="")
    sort_by = st.selectbox("Sort by", ["match_score", "created_at", "status"], index=1)
    descending = st.checkbox("Sort descending", value=True)

    records = load_tracker_rows(
        statuses=selected_statuses,
        minimum_score=minimum_score,
        company_search=company_search,
        sort_by=sort_by,
        descending=descending,
    )

    if records:
        st.table([
            {
                "id": row["id"],
                "status": row["status"],
                "stored_score": row["match_score"],
                "company": row["company"],
                "role": display_title_from_value(row["role"], fallback="Sample Job"),
                "location": row["location"],
                "created_at": row["created_at"],
                "applied_date": row["applied_date"],
            }
            for row in records
        ])
    else:
        st.info("No tracker records yet. Save a job to tracker to begin.")

    if not records:
        return

    selected_id = st.selectbox(
        "Select application id",
        [row["id"] for row in records],
        key="tracker_selected_id",
    )
    selected_row = next((row for row in records if row["id"] == selected_id), None)
    if selected_row is None:
        return

    st.markdown("**Selected application**")
    st.write(f"Company: {selected_row['company']}")
    st.write(f"Role: {display_title_from_value(selected_row['role'], fallback='Sample Job')}")
    st.write(f"Status: {selected_row['status']}")
    st.write(f"Stored score: {selected_row['match_score']}")
    st.write(f"Stored recommendation: {selected_row['recommendation']}")

    new_status = st.selectbox("Update status to", status_options, index=status_options.index(selected_row["status"]))
    if st.button("Update Status", key=f"update_{selected_id}"):
        try:
            database_path = current_workspace().tracker_database_path
            if database_path is None:
                raise WorkspaceError("Tracker is unavailable in Demo workspace.")
            _, output = run_with_captured_output(update_status, selected_id, new_status, database_path)
            st.success(f"Updated application #{selected_id} to {new_status}.")
            if output:
                st.text(output)
            st.rerun()
        except Exception as error:  # noqa: BLE001
            st.error(str(error))

    st.markdown("**Delete record**")
    confirm_delete = st.checkbox("I understand this only deletes the tracker record.", key=f"confirm_delete_{selected_id}")
    if st.button("Delete Record", key=f"delete_{selected_id}"):
        if not confirm_delete:
            st.warning("Check the confirmation box before deleting this tracker record.")
        else:
            try:
                database_path = current_workspace().tracker_database_path
                if database_path is None:
                    raise WorkspaceError("Tracker is unavailable in Demo workspace.")
                _, output = run_with_captured_output(delete_application, selected_id, database_path)
                st.success(f"Deleted tracker record #{selected_id}.")
                if output:
                    st.text(output)
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(str(error))


def package_viewer_tab() -> None:
    """Render a viewer for tracker-linked or manually provided package folders."""
    render_page_header(
        "Application Package",
        "Review generated resume and cover letter material before using it. Nothing is submitted automatically.",
    )
    if demo_mode_enabled():
        st.info("This sanitized sample package demonstrates the final files produced in Personal workspace.")
        package_dir = DEMO_PACKAGE_DIR if DEMO_PACKAGE_DIR.exists() else None
        tracker_row = None
        if package_dir is None:
            st.info("Demo sample package is unavailable.")
            return
    else:
        package_dir = None
        tracker_row = None

    all_records = load_tracker_rows(sort_by="created_at", descending=True)

    if not demo_mode_enabled():
        view_mode = st.radio(
            "View source",
            ["Tracker record", "Package folder path"],
            horizontal=True,
        )
    else:
        view_mode = "Demo package"

    if view_mode == "Tracker record":
        if not all_records:
            latest_dir = st.session_state.get("latest_generated_package_dir", "")
            if latest_dir:
                candidate = Path(latest_dir)
                if candidate.exists() and candidate.is_dir():
                    package_dir = candidate
                    st.info("Showing the latest generated package. Save jobs to tracker to browse by tracker record.")
                else:
                    st.info("No packages generated yet. Review a job and generate an application package.")
                    return
            else:
                st.info("No packages generated yet. Review a job and generate an application package.")
                return

        if all_records:
            tracker_id = st.selectbox(
                "Select tracker id",
                [row["id"] for row in all_records],
                key="package_viewer_tracker_id",
            )
            tracker_row = next((row for row in all_records if row["id"] == tracker_id), None)
            if tracker_row:
                package_dir = resolve_package_dir_from_tracker(tracker_row)
    elif view_mode == "Package folder path":
        folder_input = st.text_input(
            "Package folder path",
            value=str(current_workspace().generated_dir),
        )
        candidate = Path(folder_input).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        generated_root = current_workspace().generated_dir.resolve()
        if candidate.exists() and candidate.is_dir() and candidate.is_relative_to(generated_root):
            package_dir = candidate
        else:
            st.warning("Enter a package folder inside the Personal workspace generated directory.")

    if tracker_row:
        package_dir = package_dir or latest_package_for_company_role(
            tracker_row["company"],
            tracker_row["role"],
        )
        summary_left, summary_right = st.columns([0.64, 0.36], gap="large")
        with summary_left:
            st.markdown(f"**{tracker_row['company']}**")
            st.write(display_title_from_value(tracker_row["role"], fallback="Sample Job"))
            st.caption(
                f"Status: {tracker_row['status']} | Stored package/tracker score: "
                f"{tracker_row['match_score'] if tracker_row['match_score'] is not None else '-'}"
            )
        with summary_right:
            if tracker_row["job_url"]:
                st.link_button("Open Job URL", tracker_row["job_url"], width="stretch")
            notes = str(tracker_row["notes"] or "").strip()
            if notes:
                with st.expander("Tracker notes", expanded=False):
                    st.write(notes)
    elif package_dir is not None:
        notes = load_package_notes(package_dir)
        if notes:
            with st.expander("Package notes", expanded=False):
                st.markdown(notes)

    if package_dir is None:
        st.info("No packages generated yet. Review a job and generate an application package.")
        return

    analysis_path = package_dir / "analysis.md"
    resume_md_path = package_dir / "tailored_resume.md"
    cover_letter_md_path = package_dir / "cover_letter.md"
    cover_letter_docx_path = package_dir / "cover_letter.docx"
    resume_docx_path = first_existing_package_file(package_dir, ["tailored_resume.docx", "resume.docx"])
    internal_notes_paths = existing_package_files(package_dir, INTERNAL_NOTES_FILE_ORDER)
    internal_notes = "\n\n".join(read_text_file(path) for path in internal_notes_paths)

    package_key = safe_slug(relative_path(package_dir)) or "selected_package"
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    flash_key = f"package_flash_{package_key}"
    flash_message = st.session_state.pop(flash_key, "")
    if flash_message:
        st.success(flash_message)

    st.markdown("**Package Actions**")
    package_left, package_right = st.columns([0.48, 0.52], gap="large")
    with package_left:
        render_readiness_checklist(
            resume_md_path,
            resume_docx_path,
            cover_letter_md_path,
            cover_letter_docx_path,
            analysis_path,
            internal_notes_paths,
        )
        st.caption("Review generated files before sharing them with employers.")

        action_left, action_right = st.columns(2)
        with action_left:
            if resume_docx_path is None and resume_md_path.exists():
                if demo_mode_enabled():
                    st.caption("Resume DOCX unavailable in this sample.")
                elif st.button("Generate Resume DOCX", key=f"generate_resume_docx_{package_key}"):
                    try:
                        generated_path, warnings = generate_resume_docx_for_package(package_dir)
                        if generated_path:
                            st.session_state[flash_key] = "Resume DOCX generated."
                            if warnings:
                                with st.expander("Advanced: resume DOCX warnings", expanded=False):
                                    for warning in warnings:
                                        st.write(f"- {warning}")
                            st.rerun()
                        else:
                            st.info("Resume DOCX could not be generated because the tailored resume source is missing.")
                    except Exception as error:  # noqa: BLE001
                        st.error(f"Could not generate Resume DOCX: {error}")
            elif resume_docx_path is None:
                st.info("Resume DOCX needs a tailored resume source before it can be generated.")
        with action_right:
            if not cover_letter_docx_path.exists() and cover_letter_md_path.exists():
                if demo_mode_enabled():
                    st.caption("Cover Letter DOCX unavailable in this sample.")
                elif st.button("Generate Cover Letter DOCX", key=f"generate_cover_letter_docx_{package_key}"):
                    try:
                        generated_path, warnings = generate_cover_letter_docx_for_package(package_dir)
                        if generated_path:
                            st.session_state[flash_key] = "Cover Letter DOCX generated."
                            if warnings:
                                with st.expander("Advanced: cover letter DOCX warnings", expanded=False):
                                    for warning in warnings:
                                        st.write(f"- {warning}")
                            st.rerun()
                        else:
                            st.info("Cover Letter DOCX could not be generated because the cover letter source is missing.")
                    except Exception as error:  # noqa: BLE001
                        st.error(f"Could not generate Cover Letter DOCX: {error}")
            elif not cover_letter_docx_path.exists():
                st.info("Cover Letter DOCX needs a cover letter source before it can be generated.")

    with package_right:
        st.warning("Generated materials may contain personal information. Review before sharing.")
        download_left, download_right = st.columns(2)
        with download_left:
            if resume_docx_path:
                st.download_button(
                    "Download Resume DOCX",
                    data=resume_docx_path.read_bytes(),
                    file_name=resume_docx_path.name,
                    mime=docx_mime,
                    key=f"download_resume_docx_{package_key}",
                    width="stretch",
                )
            elif not demo_mode_enabled():
                st.info("Resume DOCX not generated yet")

            if analysis_path.exists():
                st.download_button(
                    "Download Match Report",
                    data=analysis_path.read_bytes(),
                    file_name=analysis_path.name,
                    mime="text/markdown",
                    key=f"download_match_report_{package_key}",
                    width="stretch",
                )
            else:
                st.info("Match report not available yet")

        with download_right:
            if cover_letter_docx_path.exists():
                st.download_button(
                    "Download Cover Letter DOCX",
                    data=cover_letter_docx_path.read_bytes(),
                    file_name=cover_letter_docx_path.name,
                    mime=docx_mime,
                    key=f"download_cover_letter_docx_{package_key}",
                    width="stretch",
                )
            elif not demo_mode_enabled():
                st.info("Cover Letter DOCX not generated yet")

            if internal_notes_paths:
                if len(internal_notes_paths) == 1:
                    internal_notes_data = internal_notes_paths[0].read_bytes()
                    internal_notes_file_name = internal_notes_paths[0].name
                    internal_notes_mime = "text/markdown"
                else:
                    internal_notes_buffer = io.BytesIO()
                    with zipfile.ZipFile(internal_notes_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for notes_path in internal_notes_paths:
                            zip_file.write(notes_path, arcname=notes_path.name)
                    internal_notes_data = internal_notes_buffer.getvalue()
                    internal_notes_file_name = "internal_notes.zip"
                    internal_notes_mime = "application/zip"
                st.download_button(
                    "Download Internal Notes",
                    data=internal_notes_data,
                    file_name=internal_notes_file_name,
                    mime=internal_notes_mime,
                    key=f"download_internal_notes_{package_key}",
                    width="stretch",
                )
            else:
                st.info("Internal notes not available yet")

        zip_bytes, zip_paths = build_application_package_zip(package_dir)
        if zip_paths:
            st.download_button(
                "Download Full Application Package ZIP",
                data=zip_bytes,
                file_name=package_zip_filename(package_dir),
                mime="application/zip",
                key=f"download_full_package_zip_{package_key}",
                width="stretch",
            )
            st.caption("ZIP includes generated resume, cover letter, match report, and internal notes from this package.")
        else:
            st.info("Full package ZIP is available after generated files exist.")

    st.markdown("**Preview**")
    render_markdown_file(resume_md_path, "Preview Resume")
    render_markdown_file(cover_letter_md_path, "Preview Cover Letter")
    st.caption("Package reports are stored output from generation time; current live analysis appears in Review Jobs.")
    render_markdown_file(analysis_path, "Preview Stored Match Report")
    if internal_notes:
        with st.expander("Preview Internal Notes", expanded=False):
            st.markdown(internal_notes)

    st.markdown("**Application status**")
    status_left, status_middle, status_right = st.columns(3)
    with status_left:
        if demo_mode_enabled():
            st.info("Mark as Applied is disabled in Demo workspace.")
        elif tracker_row:
            if st.button("Mark as Applied"):
                try:
                    database_path = current_workspace().tracker_database_path
                    if database_path is None:
                        raise WorkspaceError("Tracker is unavailable in Demo workspace.")
                    _, output = run_with_captured_output(
                        update_status, int(tracker_row["id"]), "applied", database_path
                    )
                    st.success("Marked as applied.")
                    if output:
                        with st.expander("Advanced: tracker output", expanded=False):
                            st.text(output)
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
        else:
            st.info("Select a tracker record to mark an application as applied.")
    with status_middle:
        if st.button("Open Tracker", key="package_open_tracker"):
            go_to_page("Tracker")
    with status_right:
        if not demo_mode_enabled():
            st.info("Resume DOCX export is only shown when a real DOCX exists in the selected package.")

    with st.expander("Advanced: package files", expanded=False):
        st.write(f"Package folder: `{relative_path(package_dir)}`")
        for label, path in [
            ("tailored_resume.docx", package_dir / "tailored_resume.docx"),
            ("resume.docx", package_dir / "resume.docx"),
            ("tailored_resume.md", resume_md_path),
            ("cover_letter.docx", cover_letter_docx_path),
            ("cover_letter.md", cover_letter_md_path),
            ("analysis.md", analysis_path),
            ("cover_letter_notes.md", package_dir / "cover_letter_notes.md"),
            ("tailoring_notes.md", package_dir / "tailoring_notes.md"),
        ]:
            if path.exists():
                st.write(f"{label}: `{relative_path(path)}`")
            else:
                st.write(f"{label}: not found")

    if cover_letter_docx_path.exists():
        st.caption(f"Cover letter DOCX: `{relative_path(cover_letter_docx_path)}`")


def safety_notes_tab() -> None:
    """Render static safety reminders."""
    render_page_header("Settings")
    with st.expander("Privacy & Safety", expanded=True):
        st.markdown(
            """
            - Local-first workflow: saved jobs, tracker records, and generated packages stay on this machine.
            - No automatic submissions: the app does not submit applications or answer external forms for you.
            - Human-in-the-loop: manually review every resume, cover letter, and application answer before using it.
            - Safe generation: resume and cover letter drafts should rephrase real experience from your source resume, not invent facts.
            """
        )
    with st.expander("Developer / Advanced notes", expanded=False):
        st.markdown(
            """
            - Manually confirm visa and work authorization questions before applying.
            - `.env` should never be committed to Git.
            - API keys should never be shared.
            - Internal debug UI is disabled by default for the public app.
            """
        )


def render_candidate_workspace_setup(workspace: Workspace) -> None:
    """Collect candidate files before enabling Personal workflows."""
    render_page_header(
        "Candidate Workspace Setup",
        "Add your candidate source to initialize the private local workspace.",
    )
    st.write("Candidate source is required. Upload a PDF, DOCX, Markdown, or TXT file.")
    resume_upload = st.file_uploader(
        "Candidate source",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_RESUME_EXTENSIONS)],
        help="Files are parsed locally and stored as canonical Markdown. Text-based PDFs only; no OCR.",
        key="workspace_resume_upload",
    )
    experience_upload = st.file_uploader(
        "Experience bank (optional)",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXPERIENCE_BANK_EXTENSIONS)],
        key="workspace_experience_upload",
    )
    template_upload = st.file_uploader(
        "Cover-letter template (optional)",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS)],
        key="workspace_template_upload",
    )
    if workspace.ready:
        st.caption("Submitting replaces the candidate source and any optional file selected here.")
    if not st.button("Save Personal workspace", type="primary", disabled=resume_upload is None):
        return

    try:
        assert resume_upload is not None
        updated = initialize_personal_workspace(
            resume_filename=resume_upload.name,
            resume_content=resume_upload.getvalue(),
            experience_bank=(experience_upload.name, experience_upload.getvalue()) if experience_upload else None,
            cover_letter_template=(template_upload.name, template_upload.getvalue()) if template_upload else None,
        )
        if not updated.ready:
            raise WorkspaceError("The Personal workspace could not be validated after setup.")
        st.session_state["workspace_setup_open"] = False
        format_label = (updated.candidate_original_extension or "source").lstrip(".").upper()
        extraction_label = (updated.candidate_extraction_method or "local extraction").replace("_", " ")
        details = f"Accepted {format_label}; extracted locally with {extraction_label}."
        if updated.candidate_pdf_page_count is not None:
            details += f" PDF pages: {updated.candidate_pdf_page_count}."
        st.success("Personal workspace configured. Candidate files remain local and ignored by Git.")
        st.caption(details)
        st.rerun()
    except WorkspaceError as error:
        st.error(str(error))


def render_global_styles() -> None:
    """Apply safe, compact spacing for portfolio-friendly screenshots."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1.5rem !important;
            padding-bottom: 2rem !important;
        }
        .app-title-safe-area {
            padding-top: 2.25rem;
            padding-bottom: 0.75rem;
            overflow: visible !important;
        }
        .app-title-text {
            font-size: 2.35rem;
            font-weight: 750;
            line-height: 1.25;
            letter-spacing: -0.02em;
            margin: 0;
            padding: 0;
            overflow: visible !important;
        }
        .page-header {
            margin-top: 0.35rem;
            margin-bottom: 0.65rem;
        }
        .page-title {
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.25;
            margin: 0 0 0.35rem 0;
        }
        .page-subtitle {
            color: rgba(49, 51, 63, 0.72);
            font-size: 0.92rem;
            line-height: 1.35;
            margin: 0;
        }
        h2 {
            margin-top: 0.55rem !important;
            margin-bottom: 0.4rem !important;
        }
        h3 {
            margin-top: 0.45rem !important;
            margin-bottom: 0.35rem !important;
        }
        div[data-testid="stRadio"] {
            margin-top: 0rem !important;
            margin-bottom: 0.2rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render the workflow reminder in the sidebar."""
    st.sidebar.title("Your Job Search Flow")
    st.sidebar.selectbox(
        "Workspace",
        ["Demo", "Personal"],
        index=0,
        key="workspace_mode",
        help="Demo is sanitized and read-only. Personal uses only your ignored local workspace.",
    )
    workspace = current_workspace()
    if workspace.mode == "personal":
        st.sidebar.caption("Configured" if workspace.ready else "Setup required")
        st.sidebar.write(f"Candidate source: {'Ready' if workspace.resume_source_path else 'Missing'}")
        st.sidebar.write(f"Experience bank: {'Provided' if workspace.experience_bank_path else 'Generic fallback'}")
        st.sidebar.write(
            f"Cover-letter template: {'Provided' if workspace.cover_letter_template_path else 'Generic template'}"
        )
        if workspace.ready and st.sidebar.button("Replace candidate files"):
            st.session_state["workspace_setup_open"] = True
            st.rerun()
    else:
        st.sidebar.caption("Sanitized, read-only sample workspace")
    st.sidebar.markdown(
        """
        1. Set target roles
        2. Find jobs
        3. Review matches
        4. Generate application packages
        5. Track applications
        """
    )
    st.sidebar.caption("Local-first. Human-reviewed. No automatic submissions.")


def main() -> None:
    """Streamlit entry point."""
    st.set_page_config(page_title="Job Application Copilot", layout="wide")
    render_global_styles()
    render_sidebar()
    st.markdown(
        """
        <div class="app-title-safe-area">
          <div class="app-title-text">Job Application Copilot</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    workspace = current_workspace()
    if workspace.mode == "personal" and (
        not workspace.ready or st.session_state.get("workspace_setup_open", False)
    ):
        render_candidate_workspace_setup(workspace)
        return
    if workspace.mode == "personal":
        manual_jobs_module.MANUAL_SAVED_JOBS_DIR = workspace.jobs_dir

    if st.session_state.get("active_page") not in PAGE_NAMES:
        st.session_state["active_page"] = "Dashboard"

    current_index = PAGE_NAMES.index(st.session_state["active_page"])
    selected_page = st.radio(
        "Navigation",
        PAGE_NAMES,
        index=current_index,
        horizontal=True,
        label_visibility="collapsed",
    )

    if selected_page != st.session_state["active_page"]:
        st.session_state["active_page"] = selected_page
        st.rerun()

    active_page = st.session_state["active_page"]

    if active_page == "Dashboard":
        dashboard_tab()
    elif active_page == "Find Jobs":
        fetch_jobs_tab()
    elif active_page == "Add Target Job":
        manual_job_target_tab()
    elif active_page == "Review Jobs":
        job_descriptions_tab()
    elif active_page == "Application Package":
        package_viewer_tab()
    elif active_page == "Tracker":
        tracker_tab()
    elif active_page == "Settings":
        safety_notes_tab()


if __name__ == "__main__":
    main()
