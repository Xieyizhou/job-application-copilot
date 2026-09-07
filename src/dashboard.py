"""Streamlit dashboard for the local Job Application Toolkit.

This dashboard wraps the existing local scripts. It does not submit
applications, scrape websites, or expose API credentials.
"""

from __future__ import annotations

from dashboard_regions import _normalize_match_text as normalize_text

from scoring_extraction import is_uk_job


from dashboard_ui import sanitize_fit_text

from document_text import read_text_file

import contextlib
import io
import json
import re
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
DASHBOARD_SCORING_VERSION = "canonical-v11-evidence-label-guards"
HARD_RED_FLAG_PATTERNS = {
    "PhD required": ["phd required", "ph.d. required", "doctorate required"],
    "PhD internship/candidate": [
        "phd internship",
        "ph.d. internship",
        "phd candidate",
        "ph.d. candidate",
    ],
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

from scoring_report import analyze_job_structured  # noqa: E402
from export_documents import (  # noqa: E402
    export_cover_letter_to_docx,
    parse_job_metadata_from_package,
)
from fetch_jobs import jsearch_configured  # noqa: E402
from saved_job_deletion import archive_all_saved_jobs, archive_saved_job  # noqa: E402
from jd_enrichment import (  # noqa: E402
    enrich_saved_job_description,
    enrich_saved_job_description_from_url,
    replace_saved_job_description,
)
from ats_jd_completion import (  # noqa: E402
    complete_public_ats_jobs,
    public_ats_completion_candidates,
)
import manual_jobs as manual_jobs_module  # noqa: E402
from ml.inference import predict_relevance_batch, suppress_collapsed_relevance_signals  # noqa: E402
from output_paths import safe_slug, timestamp_slug  # noqa: E402
from tracker import add_application, update_status  # noqa: E402
from dashboard_fit import (  # noqa: E402
    confidence_level,
    eligibility_status,
)
from dashboard_fit_sections import render_fit_analysis_sections as render_compact_fit_sections  # noqa: E402
from dashboard_analysis_service import (  # noqa: E402
    analyze_dashboard_job as service_analyze_dashboard_job,
    unavailable_dashboard_analysis,
)
from dashboard_company_verification import (  # noqa: E402
    render_manual_company_confirmation,
    render_markdown_company_confirmation,
)
from dashboard_fetch import (  # noqa: E402
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
)
from dashboard_manual import (  # noqa: E402
    ManualPageServices,
    manual_job_target_tab as render_manual_job_target_page,
)
from dashboard_regions import (  # noqa: E402
    load_recent_region_keys,
)
from dashboard_repository import (  # noqa: E402
    build_dashboard_job_record as repository_build_dashboard_job_record,
    list_job_description_files as repository_list_job_description_files,
    load_screened_jobs as repository_load_screened_jobs,
    load_tracker_rows as repository_load_tracker_rows,
)
from dashboard_review import (  # noqa: E402
    RECOMMENDATION_RANK,
    review_inbox_view_matches,
)
from dashboard_review_page import (  # noqa: E402
    ReviewPageServices,
    job_descriptions_tab as render_review_jobs_page,
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
from dashboard_titles import resolve_canonical_job_title  # noqa: E402
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
from scoring_types import DashboardJob, TrackerRow  # noqa: E402


def enqueue_web_candidate_shadow(
    job_text: str,
    candidate_text: str,
    *,
    workspace_mode: str,
) -> None:
    """Load the optional ML shadow runtime only when Personal mode invokes it."""
    from ml.web_candidate_shadow import enqueue_web_candidate_shadow as enqueue

    enqueue(job_text, candidate_text, workspace_mode=workspace_mode)


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
    workspace = current_workspace()
    return repository_list_job_description_files(
        demo_mode=workspace.mode == "demo",
        demo_job_dir=DEMO_JOB_DIR,
        jobs_dir=workspace.jobs_dir,
        is_job_description=clean_job_description_path,
        search_text=search_text,
    )


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


def warnings_for_job(job_text: str) -> list[str]:
    """Return dashboard review warnings for incomplete or high-risk postings."""
    warnings = []
    if len(job_text.split()) < 120:
        warnings.append("API description may be an incomplete snippet")
    for phrase in ["work authorization", "visa", "citizenship", "sponsorship"]:
        if phrase in job_text.lower():
            warnings.append(f"Review {phrase} requirement manually")
    if is_uk_job(job_text):
        warnings.append("UK HPI review: user may be eligible to apply, but should not claim current UK work authorization.")
        if asks_for_uk_work_authorization(job_text):
            warnings.append("Confirm whether the employer accepts candidates planning to use the HPI visa route")
    return warnings


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


def analyze_job_for_dashboard(
    job: DashboardJob,
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
    workspace = current_workspace() if use_cache or candidate_text is None else None
    if candidate_text is None:
        candidate_path = workspace.resume_source_path if workspace else None
        if candidate_path is None or not candidate_path.is_file():
            return unavailable_dashboard_analysis("Candidate source is missing or unreadable.")
        candidate_text = read_text_file(candidate_path)
    cache = st.session_state.setdefault("dashboard_analysis_cache", {}) if use_cache else {}
    analysis = service_analyze_dashboard_job(
        job,
        job_text,
        candidate_text,
        analyzer=analyze_job_structured,
        scoring_version=DASHBOARD_SCORING_VERSION,
        cache=cache if use_cache else None,
        workspace_mode=workspace.mode if workspace else "provided",
        workspace_root=str(workspace.root) if workspace else "provided",
    )
    if workspace is not None and workspace.mode == "personal":
        enqueue_web_candidate_shadow(job_text, candidate_text, workspace_mode=workspace.mode)
    return analysis


def build_dashboard_job_record(path: Path) -> DashboardJob:
    """Build one ranked dashboard row from a saved job Markdown file."""
    return repository_build_dashboard_job_record(
        path,
        read_text=read_text_file,
        detect_red_flags=detect_dashboard_red_flags,
        collect_warnings=warnings_for_job,
    )


def dashboard_rank_key(job: DashboardJob) -> tuple[int, int, int, int]:
    """Sort by recommendation, score, confidence, then fewer red flags."""
    return (
        RECOMMENDATION_RANK.get(job["recommendation"], 0),
        int(job.get("score") or 0),
        CONFIDENCE_RANK.get(confidence_level(job.get("confidence")).title(), 0),
        -len(job["red_flags"]),
    )


def load_screened_jobs(search_text: str = "") -> list[DashboardJob]:
    """Load jobs and attach canonical full-analysis results in memory."""
    _ = search_text
    candidate_path = current_workspace().resume_source_path
    return repository_load_screened_jobs(
        list_job_description_files(),
        candidate_path=candidate_path,
        read_text=read_text_file,
        build_record=build_dashboard_job_record,
        analyze_job=analyze_job_for_dashboard,
        rank_key=dashboard_rank_key,
        predict_relevance=predict_relevance_batch,
        suppress_relevance=suppress_collapsed_relevance_signals,
    )


def load_tracker_rows(
    statuses: list[str] | None = None,
    minimum_score: int = 0,
    company_search: str = "",
    sort_by: str = "created_at",
    descending: bool = True,
) -> list[TrackerRow]:
    """Load tracker rows with simple local filtering and sorting."""
    return repository_load_tracker_rows(
        current_workspace().tracker_database_path,
        statuses=statuses,
        minimum_score=minimum_score,
        company_search=company_search,
        sort_by=sort_by,
        descending=descending,
    )


def resolve_package_dir_from_tracker(row: TrackerRow | dict[str, Any]) -> Path | None:
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


def tracker_args_for_job(job: DashboardJob) -> SimpleNamespace:
    """Prepare canonical current-analysis values for a new tracker row."""
    eligibility = eligibility_status(job).replace("_", " ").title()
    confidence = confidence_level(job.get("confidence")).title()
    score = job.get("score")
    return SimpleNamespace(
        company=str(job.get("company", "")).strip() or "Unknown company",
        role=resolve_canonical_job_title(job),
        location=str(job.get("normalized_location", "") or job.get("location", "")).strip(),
        job_url=str(job.get("job_url", "")).strip(),
        match_score=int(score) if job.get("analysis_available") and score is not None else None,
        recommendation=str(job.get("recommendation", "Manual Review")).strip(),
        status="saved",
        resume_file="",
        cover_letter_file="",
        notes=(f"Saved from Review Jobs. Eligibility: {eligibility}. Scoring confidence: {confidence}. No application was submitted."),
    )


def save_job_to_tracker(job: DashboardJob) -> tuple[int | None, str]:
    """Save a reviewed job lead to the local tracker without generating documents."""
    args = tracker_args_for_job(job)
    workspace = current_workspace()
    workspace.require_writable()
    assert workspace.tracker_database_path is not None
    return run_with_captured_output(add_application, args, workspace.tracker_database_path)


def tracker_row_for_job(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
) -> TrackerRow | None:
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


def tracker_status_for_job(job: DashboardJob, tracker_rows: list[TrackerRow]) -> str:
    """Return the current tracker status for display on job cards."""
    if demo_mode_enabled():
        return "Demo only"
    row = tracker_row_for_job(job, tracker_rows)
    return str(row.get("status", "") or "Not tracked") if row else "Not tracked"


def package_dir_for_job(job: DashboardJob, tracker_rows: list[TrackerRow]) -> Path | None:
    """Find a generated package for display without changing package behavior."""
    row = tracker_row_for_job(job, tracker_rows)
    if row:
        package_dir = resolve_package_dir_from_tracker(row)
        if package_dir:
            return package_dir
    return latest_package_for_company_role(str(job.get("company", "")), resolve_canonical_job_title(job))


def package_status_for_job(job: DashboardJob, tracker_rows: list[TrackerRow]) -> str:
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


def default_review_inbox_view(
    jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
    *,
    demo: bool,
) -> str:
    """Choose the initial Review Jobs view without changing filtering rules."""
    if demo:
        return "All"
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
            "Needs attention",
            tracker_status_for_job(job, tracker_rows),
            package_status_for_job(job, tracker_rows),
        )
        for job in jobs
    ):
        return "Needs attention"
    return "All"


def mark_job_not_interested(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
) -> tuple[int | None, str]:
    """Archive a tracked job, creating a tracker row first if needed."""
    row = tracker_row_for_job(job, tracker_rows)
    if row is None:
        tracker_id, output = save_job_to_tracker(job)
    else:
        tracker_id = int(row["id"])
        output = ""
    if tracker_id is None:
        raise WorkspaceError("Tracker record could not be created.")
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


def generate_cover_letter_docx_for_package(package_dir: Path) -> tuple[Path | None, list[str]]:
    """Regenerate cover_letter.docx for a selected package when possible."""
    cover_letter_md_path = package_dir / "cover_letter.md"
    if not cover_letter_md_path.exists():
        return None, ["Cover letter source is missing."]
    cover_letter_docx_path = package_dir / "cover_letter.docx"
    workspace = current_workspace()
    warnings = export_cover_letter_to_docx(
        cover_letter_md_path,
        cover_letter_docx_path,
        parse_job_metadata_from_package(package_dir),
        generic_cover_letter_template(workspace),
        workspace.candidate_profile,
    )
    return cover_letter_docx_path, warnings


def dashboard_tab() -> None:
    """Compose the page with its stateful operations."""
    render_home_page(
        HomePageServices(
            count_generated_packages=count_generated_packages,
            demo_mode_enabled=demo_mode_enabled,
            go_to_page=go_to_page,
            load_screened_jobs=load_screened_jobs,
        )
    )


def fetch_jobs_tab() -> None:
    """Compose the page with its stateful operations."""
    render_fetch_jobs_page(
        FetchPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            go_to_page=go_to_page,
            relocate_fetched_jobs_to_workspace=relocate_fetched_jobs_to_workspace,
            run_with_captured_output=run_with_captured_output,
            default_recommendation_limit=DEFAULT_RECOMMENDATION_LIMIT,
            min_recommendation_limit=MIN_RECOMMENDATION_LIMIT,
            max_recommendation_limit=MAX_RECOMMENDATION_LIMIT,
            show_debug_ui=SHOW_DEBUG_UI,
        )
    )


def manual_job_target_tab() -> None:
    """Compose the page with its stateful operations."""
    render_manual_job_target_page(
        ManualPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            go_to_page=go_to_page,
            relative_path=relative_path,
            render_manual_company_confirmation=render_manual_company_confirmation,
            run_with_captured_output=run_with_captured_output,
            switch_workspace_mode=switch_workspace_mode,
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


def structured_fit_analysis(job: DashboardJob, job_text: str) -> dict[str, Any]:
    """Return the same canonical analysis already attached to the loaded job."""
    existing = job.get("analysis_result")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    return analyze_job_for_dashboard(job, job_text)


def render_fit_analysis_sections(job: DashboardJob, job_text: str) -> None:
    """Render decision evidence with diagnostics collapsed by default."""
    render_compact_fit_sections(
        job,
        job_text,
        analyze=structured_fit_analysis,
        sanitize=sanitize_fit_text,
        demo_mode=demo_mode_enabled(),
        show_debug=SHOW_DEBUG_UI,
    )


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


def job_descriptions_tab() -> None:
    """Compose the page with its stateful operations."""
    render_review_jobs_page(
        ReviewPageServices(
            archive_all_saved_jobs=archive_all_saved_jobs,
            archive_saved_job=archive_saved_job,
            complete_public_ats_jobs=complete_public_ats_jobs,
            current_workspace=current_workspace,
            default_review_inbox_view=default_review_inbox_view,
            demo_mode_enabled=demo_mode_enabled,
            go_to_page=go_to_page,
            enrich_saved_job_description=enrich_saved_job_description,
            enrich_saved_job_description_from_url=enrich_saved_job_description_from_url,
            jsearch_configured=jsearch_configured,
            load_package_notes=load_package_notes,
            load_screened_jobs=load_screened_jobs,
            load_tracker_rows=load_tracker_rows,
            package_dir_for_job=package_dir_for_job,
            package_status_for_job=package_status_for_job,
            public_ats_completion_candidates=public_ats_completion_candidates,
            relative_path=relative_path,
            render_fit_analysis_sections=render_fit_analysis_sections,
            render_generation_success=render_generation_success,
            render_markdown_company_confirmation=render_markdown_company_confirmation,
            replace_saved_job_description=replace_saved_job_description,
            run_with_captured_output=run_with_captured_output,
            save_recent_region_key=save_recent_region_key,
            tracker_row_for_job=tracker_row_for_job,
            tracker_status_for_job=tracker_status_for_job,
            show_debug_ui=SHOW_DEBUG_UI,
        )
    )


def tracker_tab() -> None:
    """Compose the page with its stateful operations."""
    render_tracker_page(
        TrackerPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            load_tracker_rows=load_tracker_rows,
            run_with_captured_output=run_with_captured_output,
        )
    )


def package_viewer_tab() -> None:
    """Compose the page with its stateful operations."""
    render_cover_letter_page(
        CoverLetterPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            generate_cover_letter_docx_for_package=generate_cover_letter_docx_for_package,
            latest_package_for_company_role=latest_package_for_company_role,
            load_tracker_rows=load_tracker_rows,
            relative_path=relative_path,
            resolve_package_dir_from_tracker=resolve_package_dir_from_tracker,
            run_with_captured_output=run_with_captured_output,
        )
    )


def safety_notes_tab() -> None:
    """Compose the page with its stateful operations."""
    render_settings_page(
        SettingsPageServices(
            current_workspace=current_workspace,
            demo_mode_enabled=demo_mode_enabled,
            list_job_description_files=list_job_description_files,
            load_tracker_rows=load_tracker_rows,
        )
    )


def render_candidate_workspace_setup(workspace: Workspace) -> None:
    """Render Personal workspace setup through the extracted page module."""
    render_workspace_setup_page(workspace)


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
