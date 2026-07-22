"""Streamlit dashboard for the local Job Application Toolkit.

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
DEFAULT_RECOMMENDATION_LIMIT = 12
MIN_RECOMMENDATION_LIMIT = 5
MAX_RECOMMENDATION_LIMIT = 30
SHOW_DEBUG_UI = False
DASHBOARD_SCORING_VERSION = "canonical-v5-full-jd-gate"
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
CONFIDENCE_RANK = {
    "High": 3,
    "Medium": 2,
    "Low": 1,
}
SHORTLIST_REGION_OPTIONS = [
    "all",
    "Remote",
    "United States",
    "Canada",
    "Australia",
]

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from analyze_job import analyze_job_structured, extract_job_description_body  # noqa: E402
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
    parse_job_metadata_from_package,
)
from fetch_history import latest_successful_fetch_run, load_fetch_runs  # noqa: E402
from fetch_jobs import jsearch_configured  # noqa: E402
from jd_enrichment import enrich_saved_job_description  # noqa: E402
import manual_jobs as manual_jobs_module  # noqa: E402
from manual_jobs import confirm_manual_job_company  # noqa: E402
from ml.inference import predict_relevance_batch, suppress_collapsed_relevance_signals  # noqa: E402
from ml.jd_quality import classify_jd_quality  # noqa: E402
from output_paths import safe_slug, timestamp_slug  # noqa: E402
from tracker import add_application, update_status  # noqa: E402
from dashboard_fit import (  # noqa: E402
    apply_canonical_analysis,
    build_fit_presentation,
    confidence_level,
    eligibility_status,
    summarize_analysis_requirements,
)
from dashboard_fetch import (  # noqa: E402
    ADZUNA_SUPPORTED_COUNTRIES,
    DEFAULT_FETCH_LIMIT_PER_SOURCE,
    MAX_FETCH_LIMIT_PER_SOURCE,
    REGION_CONFIG,
    REGION_OPTIONS,
    FetchPageServices,
    fetch_jobs_tab as render_fetch_jobs_page,
)
from dashboard_home import (  # noqa: E402
    HomePageServices,
    dashboard_tab as render_home_page,
)
from dashboard_cover_letter import (  # noqa: E402
    CoverLetterPageServices,
    package_viewer_tab as render_cover_letter_page,
)
from dashboard_packages import (  # noqa: E402
    INTERNAL_PACKAGE_FILES,
    build_application_package_zip,
    existing_package_files,
    package_zip_filename,
    readiness_status,
)
from dashboard_manual import (  # noqa: E402
    ManualPageServices,
    manual_job_target_tab as render_manual_job_target_page,
)
from dashboard_regions import (  # noqa: E402
    build_region_options,
    default_region_option_keys,
    dynamic_source_options,
    filtered_region_option_keys,
    infer_high_level_region,
    infer_location_from_path,
    job_matches_region_option,
    load_recent_region_keys,
    normalize_location,
    region_label,
    region_option_key,
    region_search_blob,
    source_display_name,
)
from dashboard_review import (  # noqa: E402
    RECOMMENDATION_RANK,
    is_current_recommendation,
    is_ignored_tracker_status,
    is_strong_match,
    job_evidence_label,
    job_needs_full_jd,
    parse_local_datetime,
    review_inbox_view_matches,
    review_job_next_action,
    review_job_sort_key,
    sorted_review_jobs,
    tracker_age_days,
    tracker_follow_up_due,
    tracker_next_action,
)
from dashboard_review_page import (  # noqa: E402
    ReviewPageServices,
    job_descriptions_tab as render_review_jobs_page,
    render_job_result_cards as render_review_job_cards,
    render_review_action_buttons as render_review_job_actions,
    resolve_review_job_selection,
    set_review_job_selection,
)
from dashboard_shell import (  # noqa: E402
    PAGE_NAMES,
    render_global_styles as render_shell_styles,
    render_sidebar as render_shell_sidebar,
    run_app,
    switch_workspace_mode as switch_shell_workspace_mode,
)
from dashboard_settings import (  # noqa: E402
    SettingsPageServices,
    render_candidate_workspace_setup as render_workspace_setup_page,
    safety_notes_tab as render_settings_page,
)
from dashboard_titles import (  # noqa: E402
    display_title_from_value,
    first_role_heading,
    get_job_display_title,
    is_placeholder_job_title,
    looks_like_internal_slug,
    read_markdown_field,
    resolve_canonical_job_title,
)
from dashboard_tracker import (  # noqa: E402
    TrackerPageServices,
    tracker_tab as render_tracker_page,
)
from workspace import (  # noqa: E402
    Workspace,
    WorkspaceError,
    generic_cover_letter_template,
    resolve_workspace,
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
    """Resolve the selected workspace, defaulting fresh sessions to Personal."""
    return resolve_workspace(str(st.session_state.get("workspace_mode", "Personal")))


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


def save_recent_region_key(region_key: str) -> None:
    """Persist a small MRU list for the compact default region dropdown."""
    if demo_mode_enabled() or not region_key or region_key == "all":
        return
    UI_STATE_DIR.mkdir(parents=True, exist_ok=True)
    recent_keys = [key for key in load_recent_region_keys() if key != region_key]
    recent_keys.insert(0, region_key)
    RECENT_REGIONS_PATH.write_text(json.dumps(recent_keys[:10], indent=2), encoding="utf-8")


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


def build_dashboard_job_record(path: Path) -> dict[str, Any]:
    """Build one ranked dashboard row from a saved job Markdown file."""
    job_text = read_text_file(path)
    jd_quality = classify_jd_quality(job_text)
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
    description_source = read_markdown_field(job_text, "Description Source", "")
    jd_fetch_status = read_markdown_field(job_text, "JD Fetch Status", "")
    description_word_count = len(extract_job_description_body(job_text).split())
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
        "description_source": description_source,
        "jd_fetch_status": jd_fetch_status,
        "description_word_count": description_word_count,
        "jd_quality": jd_quality,
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


def job_description_preference(job: dict[str, Any]) -> tuple[int, int, int]:
    """Prefer duplicate records using the canonical JD-quality result."""
    quality = dict(job.get("jd_quality", {}) or {})
    readiness = (
        2
        if quality.get("reliable_scoring_ready", False)
        else 1
        if quality.get("provisional_scoring_ready", False)
        else 0
    )
    return (
        readiness,
        int(quality.get("quality_score", 0) or 0),
        int(job.get("description_word_count", 0) or 0),
    )


def deduplicate_dashboard_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate saved jobs while retaining the strongest available JD evidence."""
    unique_jobs: list[dict[str, Any]] = []
    url_indexes: dict[str, int] = {}
    fallback_indexes: dict[tuple[str, str, str], int] = {}

    for job in jobs:
        job_url, company, role, location = job_duplicate_key(job)
        fallback = (company, role, location)
        existing_index = url_indexes.get(job_url) if job_url else None
        if existing_index is None and all(fallback):
            existing_index = fallback_indexes.get(fallback)

        if existing_index is None:
            existing_index = len(unique_jobs)
            unique_jobs.append(job)
        elif job_description_preference(job) > job_description_preference(unique_jobs[existing_index]):
            unique_jobs[existing_index] = job

        if job_url:
            url_indexes[job_url] = existing_index
        if all(fallback):
            fallback_indexes[fallback] = existing_index

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
    relevance_pairs: list[tuple[str, str]] = []
    for job in unique_records:
        job_text = read_text_file(Path(job["path"]))
        analysis = analyze_job_for_dashboard(job, job_text, candidate_text)
        analyzed = apply_canonical_analysis(job, analysis)
        relevance_pairs.append((candidate_text, extract_job_description_body(job_text)))
        presentation = build_fit_presentation(analyzed)
        analyzed["label"] = (
            f"{analyzed['company']} | {get_job_display_title(analyzed)} | "
            f"{analyzed['location']} | {presentation['role_fit']}"
        )
        analyzed_records.append(analyzed)
    relevance_signals = suppress_collapsed_relevance_signals(
        predict_relevance_batch(relevance_pairs)
    )
    for analyzed, relevance_signal in zip(analyzed_records, relevance_signals):
        analyzed["ml_relevance"] = relevance_signal
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
    """Count generated cover-letter bundle folders without opening user content."""
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
        return "Demo cover letter" if bundled_job_path.resolve() == job_path.resolve() else "Demo only"
    return "Cover letter ready" if package_dir_for_job(job, tracker_rows) else "No cover letter"


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
    """Store the latest generated bundle selection for the Cover Letter page."""
    tracker_id = summary.get("tracker_id")
    package_dir = summary.get("package_dir")
    if tracker_id:
        st.session_state["package_viewer_tracker_id"] = int(tracker_id)
    if package_dir:
        st.session_state["latest_generated_package_dir"] = str(package_dir)
    st.session_state["latest_generated_package_summary"] = {
        "cover_letter": bool(summary.get("cover_letter_path")),
        "match_report": bool(summary.get("analysis_path")),
        "internal_notes": bool(summary.get("cover_letter_notes_path")),
        "tracker_id": tracker_id,
    }


def load_package_notes(package_dir: Path) -> str:
    """Read the most useful internal notes file available for a package."""
    for name in ["cover_letter_notes.md", "analysis.md"]:
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


def render_readiness_checklist(
    cover_letter_md_path: Path,
    cover_letter_docx_path: Path,
    analysis_path: Path,
    internal_notes_paths: list[Path],
) -> None:
    """Show package readiness without exposing raw paths in the main flow."""
    st.markdown("**Application Materials**")
    st.table(
        [
            {
                "Material": "Uploaded Resume",
                "Status": "Used unchanged",
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
                "Source": source_display_name(str(job.get("source", ""))),
                "Saved": "Yes" if job.get("path") else "No",
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
            st.caption("Saved locally · Fit and evidence quality are calculated in Review Jobs")
            if job.get("job_url"):
                st.link_button("Open original posting", str(job["job_url"]))
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


def render_action_callout(action: str, *, caution: bool = False) -> None:
    """Render a consistent, compact next-action callout."""
    message = f"Next best action: {action}"
    if caution:
        st.warning(message)
    else:
        st.info(message)


def dashboard_tab() -> None:
    """Render the home page through its explicitly injected services."""
    render_home_page(
        HomePageServices(
            count_generated_packages=count_generated_packages,
            demo_mode_enabled=demo_mode_enabled,
            load_screened_jobs=load_screened_jobs,
            load_tracker_rows=load_tracker_rows,
            render_page_header=render_page_header,
        )
    )


def fetch_jobs_tab() -> None:
    """Render Find Jobs through the extracted discovery module."""
    render_fetch_jobs_page(
        FetchPageServices(
            demo_mode_enabled=demo_mode_enabled,
            go_to_page=go_to_page,
            relocate_fetched_jobs_to_workspace=relocate_fetched_jobs_to_workspace,
            render_fetch_history_section=render_fetch_history_section,
            render_fetch_run_job_cards=render_fetch_run_job_cards,
            render_fetch_run_job_table=render_fetch_run_job_table,
            render_page_header=render_page_header,
            run_with_captured_output=run_with_captured_output,
            default_recommendation_limit=DEFAULT_RECOMMENDATION_LIMIT,
            min_recommendation_limit=MIN_RECOMMENDATION_LIMIT,
            max_recommendation_limit=MAX_RECOMMENDATION_LIMIT,
            show_debug_ui=SHOW_DEBUG_UI,
        )
    )

def manual_job_target_tab() -> None:
    """Render Add Target Job through the extracted manual workflow module."""
    render_manual_job_target_page(
        ManualPageServices(
            company_generation_allowed=company_generation_allowed,
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            relative_path=relative_path,
            render_manual_company_confirmation=render_manual_company_confirmation,
            render_page_header=render_page_header,
            run_with_captured_output=run_with_captured_output,
        )
    )


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
    st.markdown("**Decision summary**")
    eligibility = dict(analysis.get("eligibility", {}))
    scoring_confidence = dict(analysis.get("confidence", {}))
    jd_quality = dict(analysis.get("jd_quality", {}) or job.get("jd_quality", {}) or {})
    level = confidence_level(scoring_confidence)
    score = analysis.get("score")
    decision_cols = st.columns(3)
    decision_cols[0].metric("Recommendation", analysis.get("recommendation", "Manual Review"))
    decision_cols[1].metric("Eligibility", str(eligibility.get("status", "manual_review")).replace("_", " ").title())
    decision_cols[2].metric("Confidence", level.title())
    st.write(f"Why: {sanitize_fit_text(analysis.get('main_reason', 'Review manually.'))}")
    st.write(f"Risk: {sanitize_fit_text(analysis.get('main_risk', 'Review the full job description manually.'))}")
    role_alignment = dict(analysis.get("role_alignment", {}) or {})
    if role_alignment.get("detected"):
        role_focus = sanitize_fit_text(role_alignment.get("focus", "Not detected"))
        role_support = "Supported" if role_alignment.get("score") == 100 else "Candidate evidence not found"
        st.caption(f"Role focus: {role_focus} · {role_support}")
    if jd_quality:
        st.caption(
            f"JD quality: {jd_quality.get('display_label', 'Needs review')} · "
            f"{jd_quality.get('next_action', 'Verify the complete posting.')}"
        )

    learned_signal = dict(job.get("ml_relevance", {}) or {})
    if (
        learned_signal.get("available")
        and learned_signal.get("displayable", True)
        and jd_quality.get("reliable_scoring_ready", False)
    ):
        probability = float(learned_signal.get("probability", 0.0))
        st.info(
            f"Experimental local relevance signal: {probability:.0%}. "
            "It is trained on synthetic candidate/job pairs and is shown only as a second opinion; "
            "it does not change Role Fit, eligibility, ranking, or recommendation."
        )

    if level == "low":
        if int(scoring_confidence.get("candidate_evidence_count", 0) or 0) > 0:
            st.info(
                "Candidate evidence is loaded, but this job posting yielded too few "
                "recognized requirements for a reliable score. The provisional Role Fit "
                "has been evidence-calibrated; the coverage value below describes only the "
                "extracted terms. Paste the full job description before making an application decision."
            )
        evidence_cols = st.columns(3)
        evidence_cols[0].metric("Recognized", terms["active_requirement_count"])
        evidence_cols[1].metric("Matched", terms["matched_requirement_count"])
        coverage_score = presentation.get("coverage_score")
        evidence_cols[2].metric("Observed coverage", f"{int(coverage_score)}%" if coverage_score is not None else "—")

    strengths_col, gaps_col = st.columns(2, gap="large")
    with strengths_col:
        st.markdown("**Supported strengths**")
        for item in list(analysis.get("matched_strengths", []))[:5] or ["No supported strengths extracted yet."]:
            st.write(f"- {sanitize_fit_text(item)}")
    with gaps_col:
        st.markdown("**Gaps / weak evidence**")
        for item in list(analysis.get("weak_areas", []))[:5] or ["No major gap detected in the recognized requirements."]:
            st.write(f"- {sanitize_fit_text(item)}")

    semantic_evidence = dict(analysis.get("semantic_evidence", {}) or {})
    semantic_matches = list(semantic_evidence.get("matches", []) or [])
    if semantic_matches:
        with st.expander("Requirement-to-resume evidence map", expanded=True):
            st.caption(
                f"{semantic_evidence.get('accepted_count', 0)} of "
                f"{semantic_evidence.get('requirement_count', 0)} requirements have evidence above the "
                f"{float(semantic_evidence.get('threshold', 0.0)):.0%} threshold · "
                f"{semantic_evidence.get('method', 'local evidence retrieval')}"
            )
            for match in semantic_matches[:8]:
                demand = str(match.get("demand", "required")).title()
                st.markdown(f"**{demand}: {sanitize_fit_text(match.get('requirement', 'Requirement'))}**")
                if match.get("accepted"):
                    st.write(f"Resume evidence: “{sanitize_fit_text(match.get('evidence', ''))}”")
                    st.caption(
                        f"Similarity {float(match.get('similarity', 0.0)):.0%} · "
                        f"{match.get('match_type', 'Evidence support')} · "
                        f"Section: {sanitize_fit_text(match.get('section_evidence', 'Resume'))}"
                    )
                else:
                    st.caption("Insufficient evidence · no resume statement passed the acceptance threshold")

    with st.expander("Recognized requirement details", expanded=False):
        if not terms["active_requirement_count"]:
            st.write("Requirements could not be extracted reliably. Review the full job description manually.")
        else:
            render_keyword_list("Matched required", terms["matched_required"], empty_text="None")
            render_keyword_list("Matched preferred", terms["matched_preferred"], empty_text="None")
            render_keyword_list("Missing required", terms["missing_required"], empty_text="None among recognized terms")
            render_keyword_list("Missing preferred", terms["missing_preferred"], empty_text="None")
            render_keyword_list("Partial required", terms["partial_required"], empty_text="None")
            render_keyword_list("Partial preferred", terms["partial_preferred"], empty_text="None")

    with st.expander("Resume tailoring suggestions", expanded=False):
        for item in list(analysis.get("resume_suggestions", []))[:5] or ["Not available yet."]:
            st.write(f"- {sanitize_fit_text(item)}")

    jd_evidence = list(analysis.get("jd_evidence", []))
    profile_evidence = list(analysis.get("profile_evidence", []))
    with st.expander("Source evidence", expanded=False):
        if jd_evidence:
            st.caption("Job description evidence")
            for item in jd_evidence[:3]:
                st.write(f"- {sanitize_fit_text(item)}")
        if profile_evidence:
            st.caption("Candidate-profile evidence" if not demo_mode_enabled() else "Demo-profile evidence")
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

    st.success("Cover letter bundle generated.")
    st.write("Uploaded resume used unchanged as the factual source")
    st.write("Cover letter generated")
    if summary.get("cover_letter_docx_path"):
        st.write("Cover letter DOCX generated")
    st.write("Match report generated")
    if summary.get("cover_letter_notes_path"):
        st.write("Evidence trace and gap audit generated")
    st.info("Open the Cover Letter page to preview and export the draft.")


def generate_review_job_package(
    job: dict[str, Any],
    button_key: str,
    primary: bool = False,
    label: str = "Generate Cover Letter",
) -> None:
    """Generate a cover-letter bundle for a reviewed job."""
    selected_path = job["path"]
    metadata = parse_job_metadata(selected_path)
    company = str(job.get("company_normalized") or metadata.get("company") or job.get("company", "")).strip()
    role = str(metadata.get("role") or job.get("role", "")).strip()
    location = normalize_location(str(metadata.get("location") or job.get("normalized_location") or job.get("location", "")))
    job_url = str(metadata.get("job_url") or job.get("job_url", "")).strip()

    if not st.button(label, key=button_key, type="primary" if primary else "secondary", width="stretch"):
        return
    if not all([company, role, location, job_url]):
        st.error("Company, role, location, and job URL are required before generating a cover letter.")
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
            with st.expander("Advanced: cover-letter generation output", expanded=False):
                st.text(output)
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not generate the cover letter: {error}")


def review_page_services() -> ReviewPageServices:
    """Build the explicit dependency surface for Review Jobs."""
    return ReviewPageServices(
        build_job_snippet=build_job_snippet,
        card_html=card_html,
        company_generation_allowed=company_generation_allowed,
        current_workspace=current_workspace,
        default_review_inbox_view=default_review_inbox_view,
        demo_mode_enabled=demo_mode_enabled,
        go_to_page=go_to_page,
        enrich_saved_job_description=enrich_saved_job_description,
        jsearch_configured=jsearch_configured,
        key_requirements_from_text=key_requirements_from_text,
        load_package_notes=load_package_notes,
        load_screened_jobs=load_screened_jobs,
        load_tracker_rows=load_tracker_rows,
        mark_job_not_interested=mark_job_not_interested,
        package_dir_for_job=package_dir_for_job,
        package_status_for_job=package_status_for_job,
        read_text_file=read_text_file,
        relative_path=relative_path,
        render_action_callout=render_action_callout,
        render_fit_analysis_sections=render_fit_analysis_sections,
        render_generation_success=render_generation_success,
        render_markdown_company_confirmation=render_markdown_company_confirmation,
        render_page_header=render_page_header,
        run_with_captured_output=run_with_captured_output,
        sanitize_fit_text=sanitize_fit_text,
        save_job_to_tracker=save_job_to_tracker,
        save_recent_region_key=save_recent_region_key,
        tracker_row_for_job=tracker_row_for_job,
        tracker_status_for_job=tracker_status_for_job,
        default_recommendation_limit=DEFAULT_RECOMMENDATION_LIMIT,
        min_recommendation_limit=MIN_RECOMMENDATION_LIMIT,
        max_recommendation_limit=MAX_RECOMMENDATION_LIMIT,
        show_debug_ui=SHOW_DEBUG_UI,
    )


def render_review_action_buttons(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    key_prefix: str,
) -> None:
    """Compatibility wrapper for the extracted review action renderer."""
    render_review_job_actions(job, tracker_rows, key_prefix, review_page_services())


def render_job_result_cards(
    jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
) -> None:
    """Compatibility wrapper for the extracted review card renderer."""
    render_review_job_cards(jobs, tracker_rows, review_page_services())


def job_descriptions_tab() -> None:
    """Render Review Jobs through the extracted page module."""
    render_review_jobs_page(review_page_services())

def tracker_tab() -> None:
    """Render Tracker through the extracted page module."""
    render_tracker_page(
        TrackerPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            load_tracker_rows=load_tracker_rows,
            render_action_callout=render_action_callout,
            render_page_header=render_page_header,
            run_with_captured_output=run_with_captured_output,
        )
    )


def package_viewer_tab() -> None:
    """Render Cover Letter through the extracted page module."""
    render_cover_letter_page(
        CoverLetterPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            generate_cover_letter_docx_for_package=generate_cover_letter_docx_for_package,
            go_to_page=go_to_page,
            latest_package_for_company_role=latest_package_for_company_role,
            load_package_notes=load_package_notes,
            load_tracker_rows=load_tracker_rows,
            read_text_file=read_text_file,
            relative_path=relative_path,
            render_action_callout=render_action_callout,
            render_markdown_file=render_markdown_file,
            render_page_header=render_page_header,
            render_readiness_checklist=render_readiness_checklist,
            resolve_package_dir_from_tracker=resolve_package_dir_from_tracker,
            run_with_captured_output=run_with_captured_output,
        )
    )


def settings_page_services() -> SettingsPageServices:
    """Build explicit shared services for Settings and workspace setup."""
    return SettingsPageServices(
        current_workspace=current_workspace,
        demo_mode_enabled=demo_mode_enabled,
        list_job_description_files=list_job_description_files,
        load_tracker_rows=load_tracker_rows,
        render_page_header=render_page_header,
    )


def safety_notes_tab() -> None:
    """Render Settings through the extracted page module."""
    render_settings_page(settings_page_services())


def render_candidate_workspace_setup(workspace: Workspace) -> None:
    """Render Personal workspace setup through the extracted page module."""
    render_workspace_setup_page(workspace, settings_page_services())


def render_global_styles() -> None:
    """Apply the shared Streamlit shell styles."""
    render_shell_styles(st)


def switch_workspace_mode(mode: str) -> None:
    """Compatibility wrapper for workspace-mode state transitions."""
    switch_shell_workspace_mode(st.session_state, mode)


def render_sidebar() -> None:
    """Compatibility wrapper for the shared Personal-first sidebar."""
    render_shell_sidebar(
        st,
        current_workspace=current_workspace,
        list_job_description_files=list_job_description_files,
        count_generated_packages=count_generated_packages,
        load_tracker_rows=load_tracker_rows,
        demo_mode_enabled=demo_mode_enabled,
    )


def main() -> None:
    """Streamlit entry point."""
    run_app(
        st,
        current_workspace=current_workspace,
        list_job_description_files=list_job_description_files,
        count_generated_packages=count_generated_packages,
        load_tracker_rows=load_tracker_rows,
        demo_mode_enabled=demo_mode_enabled,
        render_candidate_workspace_setup=render_candidate_workspace_setup,
        manual_jobs_module=manual_jobs_module,
        page_renderers={
            "Dashboard": dashboard_tab,
            "Find Jobs": fetch_jobs_tab,
            "Add Target Job": manual_job_target_tab,
            "Review Jobs": job_descriptions_tab,
            "Cover Letter": package_viewer_tab,
            "Tracker": tracker_tab,
            "Settings": safety_notes_tab,
        },
    )


if __name__ == "__main__":
    main()
