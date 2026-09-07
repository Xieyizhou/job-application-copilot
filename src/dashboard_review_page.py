"""Review Jobs page for evidence-first comparison and cover-letter preparation."""
from __future__ import annotations
from dataclasses import dataclass
import html
from pathlib import Path
import re
from typing import Any, Callable, cast

import streamlit as st
from apply_package import create_application_package, parse_job_metadata
from company_verification import normalize_company_name, verification_from_markdown
from dashboard_desktop_styles import render_review_workspace_styles
from dashboard_jd_recovery import render_full_jd_recovery, render_jd_workspace_styles
from dashboard_ats_completion import render_ats_completion_action
from dashboard_saved_job_management import (
    render_saved_job_delete_notice,
    render_saved_job_management,
)
from dashboard_jd_document import format_job_description_body
from structured_jd import structure_job_description
from dashboard_structured_jd import render_structured_job
from dashboard_review_chrome import render_saved_jobs_header, render_selected_job_context
from dashboard_regions import (
    job_matches_region_option,
    normalize_location,
    source_display_name,
)
from dashboard_review import review_fit_band, review_inbox_view_matches, sorted_review_jobs
from dashboard_review_filters import (
    initialize_review_state,
    render_review_filter_controls,
)
from dashboard_review_components import (
    render_review_overview_section as render_compact_review_overview,
    render_selected_review_header as render_compact_review_header,
)
from dashboard_review_selector import render_review_job_table
from dashboard_titles import get_job_display_title
from fetch_history import load_fetch_runs
from generate_cover_letter import CoverLetterGenerationError
from output_paths import safe_slug
from scoring_types import DashboardJob, TrackerRow
REVIEW_SECTION_LABELS = {"Overview": "Decision", "Fit": "Evidence", "JD": "Job description", "Cover Letter": "Cover letter"}
@dataclass(frozen=True)
class ReviewPageServices:
    """Shared dashboard operations required by the Review Jobs page."""

    company_generation_allowed: Callable[[dict[str, Any]], bool]
    complete_public_ats_jobs: Callable[..., dict[str, Any]]
    archive_all_saved_jobs: Callable[..., list[Path]]
    archive_saved_job: Callable[..., Path]
    current_workspace: Callable[[], Any]
    default_review_inbox_view: Callable[..., str]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    enrich_saved_job_description: Callable[..., dict[str, Any]]
    enrich_saved_job_description_from_url: Callable[..., dict[str, Any]]
    jsearch_configured: Callable[[], bool]
    key_requirements_from_text: Callable[[str], list[str]]
    load_package_notes: Callable[..., str]
    load_screened_jobs: Callable[..., list[DashboardJob]]
    load_tracker_rows: Callable[..., list[TrackerRow]]
    package_dir_for_job: Callable[..., Any]
    package_status_for_job: Callable[..., str]
    public_ats_completion_candidates: Callable[..., list[Path]]
    read_text_file: Callable[..., str]
    relative_path: Callable[..., str]
    render_fit_analysis_sections: Callable[..., None]
    render_generation_success: Callable[..., None]
    render_markdown_company_confirmation: Callable[..., dict[str, Any]]
    render_page_header: Callable[[str, str | None], None]
    replace_saved_job_description: Callable[..., dict[str, Any]]
    run_with_captured_output: Callable[..., tuple[Any, str]]
    sanitize_fit_text: Callable[[Any], str]
    save_recent_region_key: Callable[[str], None]
    tracker_row_for_job: Callable[..., Any]
    tracker_status_for_job: Callable[..., str]
    max_recommendation_limit: int
    show_debug_ui: bool = False
def set_review_job_selection(job: DashboardJob, focus: str = "Overview") -> None:
    """Select a review job and focus the detail panel."""
    if focus not in REVIEW_SECTION_LABELS:
        focus = "Overview"
    st.session_state["selected_review_job_path"] = str(job["path"])
    st.session_state["selected_review_job_label"] = job["label"]
    st.session_state["selected_review_tab"] = focus
    st.session_state["review_detail_tabs"] = REVIEW_SECTION_LABELS[focus]


def resolve_review_job_selection(
    shortlist: list[DashboardJob],
    selected_label: object,
    selected_path: object,
) -> DashboardJob:
    """Return the uniquely selected job, falling back safely from stale state."""
    jobs_by_label = {job["label"]: job for job in shortlist}
    jobs_by_path = {str(job["path"]): job for job in shortlist}
    if isinstance(selected_path, str) and selected_path in jobs_by_path:
        return jobs_by_path[selected_path]
    if isinstance(selected_label, str) and selected_label in jobs_by_label:
        return jobs_by_label[selected_label]
    return shortlist[0]


def review_job_matches_filters(
    job: DashboardJob,
    filters: dict[str, Any],
    tracker_status: str,
    package_status: str,
) -> bool:
    """Return whether one job satisfies the current review filters."""
    if not review_inbox_view_matches(job, filters["inbox_view"], tracker_status, package_status):
        return False
    if not job_matches_region_option(job, filters["selected_region"]):
        return False
    if (
        filters["source_filter"] != "all"
        and source_display_name(str(job["source"])) != filters["source_filter"]
    ):
        return False
    if filters.get("fit_band", "All") != "All" and review_fit_band(job) != filters["fit_band"]:
        return False
    if (
        filters["recommendation_filter"] != "all"
        and job["recommendation"] != filters["recommendation_filter"]
    ):
        return False
    tracker_filter = filters.get("tracker_filter", "all")
    tracker_is_ignored = str(tracker_status).lower() in {
        "archived",
        "ignored",
        "not interested",
        "rejected",
        "skip",
    }
    if tracker_filter == "Not tracked" and tracker_status not in {"Not tracked", "Demo only"}:
        return False
    if tracker_filter == "Tracked" and tracker_status in {"Not tracked", "Demo only"}:
        return False
    if tracker_filter == "Ignored" and not tracker_is_ignored:
        return False
    if tracker_filter != "Ignored" and tracker_is_ignored:
        return False
    score = job.get("score")
    if score is None and filters["minimum_score"] > 0:
        return False
    if score is not None and int(score) < filters["minimum_score"]:
        return False
    if filters["hide_hard"] and job["hard_red_flag"]:
        return False
    if filters["hide_degree"] and any(
        "PhD" in flag or "Master" in flag or "Graduate degree" in flag for flag in job["red_flags"]
    ):
        return False
    if filters["hide_student"] and any(
        marker in flag.lower()
        for flag in job["red_flags"]
        for marker in ("enrolled", "school", "penultimate")
    ):
        return False
    needle = filters["search_text"].strip().lower()
    return (
        not needle
        or needle in f"{job['company']} {job['role']} {get_job_display_title(job)}".lower()
    )


def filtered_review_jobs(
    all_jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
    fetch_runs_by_id: dict[str, dict[str, Any]],
    filters: dict[str, Any],
    services: ReviewPageServices,
) -> list[DashboardJob]:
    """Filter and enrich jobs without mutating the loaded records."""
    filtered: list[DashboardJob] = []
    for original in all_jobs:
        tracker_status = services.tracker_status_for_job(original, tracker_rows)
        package_status = services.package_status_for_job(original, tracker_rows)
        if not review_job_matches_filters(original, filters, tracker_status, package_status):
            continue
        job = cast(DashboardJob, dict(original))
        fetch_run = fetch_runs_by_id.get(str(job.get("latest_fetch_run_id", "")), {})
        job["fetch_run_date"] = str(fetch_run.get("created_at", job.get("last_seen_at", "")))
        job["new_label"] = "Saved job"
        job["red_flags_text"] = " | ".join(job["red_flags"]) or "-"
        job["warnings_text"] = " | ".join(job["warnings"]) or "-"
        job["tracker_status"] = tracker_status
        job["package_status"] = package_status
        filtered.append(job)
    return sorted_review_jobs(filtered, filters["sort_by"])


def render_empty_review_state(filters: dict[str, Any]) -> None:
    """Render recovery actions when filters yield no jobs."""
    st.caption("No jobs match the current filters.")
    st.button(
        "Reset filters",
        key="review_empty_clear_filters",
        type="tertiary",
        on_click=filters["clear_filters"],
    )


def resolve_selected_review_job(
    shortlist: list[DashboardJob],
    services: ReviewPageServices,
) -> DashboardJob:
    """Resolve and persist the selected job, text, and tracker row."""
    paths = [str(job["path"]) for job in shortlist]
    label_state = st.session_state.get("selected_review_job_label")
    path_state = st.session_state.get("selected_review_job_path", paths[0])
    job = resolve_review_job_selection(shortlist, label_state, path_state)
    if str(job["path"]) != path_state and job["label"] != label_state:
        set_review_job_selection(
            shortlist[0], st.session_state.get("selected_review_tab", "Overview")
        )
    st.session_state["selected_review_job_path"] = str(job["path"])
    st.session_state["selected_review_job_label"] = job["label"]
    return job


def render_selected_review_header(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
    services: ReviewPageServices,
) -> dict[str, Any]:
    """Compatibility wrapper for the four-field decision header."""
    return render_compact_review_header(job, tracker_rows, services)


def render_review_overview_section(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
    selected_path: Path,
    context: dict[str, Any],
    services: ReviewPageServices,
) -> None:
    """Compatibility wrapper for the decision-first overview."""
    render_compact_review_overview(
        job,
        selected_path,
        context,
        set_review_job_selection,
    )


def render_review_fit_section(
    job: DashboardJob,
    job_text: str,
    tracker_rows: list[TrackerRow],
    services: ReviewPageServices,
) -> None:
    """Render explainable fit evidence and saved suggestions."""
    services.render_fit_analysis_sections(job, job_text)
    package_dir = services.package_dir_for_job(job, tracker_rows)
    if package_dir:
        suggestions = services.load_package_notes(package_dir)
        if suggestions:
            with st.expander("Resume / Cover Letter suggestions", expanded=False):
                st.markdown(services.sanitize_fit_text(suggestions))


def extract_job_description_body(job_text: str, fallback: str = "") -> str:
    """Return the readable JD body without the local record metadata header."""
    parts = re.split(r"^##\s+Job Description\s*$", job_text, maxsplit=1, flags=re.MULTILINE)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip()
    return fallback.strip() or job_text.strip()


def render_review_jd_section(
    job: DashboardJob,
    job_text: str,
    selected_path: Path,
    selected_tracker_row: TrackerRow | None,
    jd_quality: dict[str, Any],
    services: ReviewPageServices,
) -> None:
    """Render a readable JD document with compact recovery actions."""
    render_jd_workspace_styles()
    source_label = source_display_name(str(job["source"]))
    quality_label = str(jd_quality.get("display_label", "Needs review"))
    st.markdown(
        '<div class="jd-document-head"><strong>Job description</strong>'
        f'<span>{html.escape(source_label)} · {html.escape(quality_label)}</span></div>',
        unsafe_allow_html=True,
    )
    render_full_jd_recovery(
        selected_path,
        jd_quality,
        key_prefix=f"jd_section_{safe_slug(str(selected_path))}",
        services=services,
        job_url=str(job.get("job_url", "")),
    )
    structured = structure_job_description(job_text)
    body = extract_job_description_body(job_text, str(job.get("preview", "")))
    body = format_job_description_body(body, get_job_display_title(job))
    render_structured_job(structured, body)
    if services.show_debug_ui:
        with st.expander("Advanced: job metadata", expanded=False):
            st.write(f"Markdown path: {services.relative_path(selected_path)}")
            st.write(f"New status: {job.get('new_label', '-')}")
            st.write(f"First seen: {job.get('first_seen_at', '-')}")
            st.write(f"Last seen: {job.get('last_seen_at', '-')}")
            st.write(f"Search run date: {job.get('fetch_run_date', '-')}")
            if selected_tracker_row:
                st.write(f"Tracker id: {selected_tracker_row['id']}")


def render_cover_letter_result(
    summary: dict[str, Any], output: str, services: ReviewPageServices
) -> None:
    """Render one successful cover-letter generation result."""
    services.render_generation_success(summary)
    st.write(f"Overall score: {summary['match_score']}/100")
    st.write(f"Recommendation: {summary['recommendation']}")
    st.write(f"Tracker id: {summary['tracker_id']}")
    st.write(f"Cover letter DOCX: {services.relative_path(summary['cover_letter_docx_path'])}")
    if summary.get("uk_review_notes"):
        st.warning("UK work authorization review")
        for note in summary["uk_review_notes"]:
            st.write(f"- {note}")
    if summary.get("export_warnings"):
        with st.expander("Validation warnings", expanded=False):
            for warning in summary["export_warnings"]:
                st.write(f"- {warning}")
    if services.show_debug_ui and output:
        with st.expander("Advanced: cover-letter generation output", expanded=False):
            st.text(output)


def render_review_cover_letter_section(
    selected_path: Path,
    jd_quality: dict[str, Any],
    services: ReviewPageServices,
) -> None:
    """Render verified cover-letter options and generation action."""
    if services.demo_mode_enabled():
        st.caption("Demo is read-only; the bundled sample draft is ready to review.")
        if st.button(
            "Open Sample Cover Letter",
            type="primary",
            width="content",
        ):
            services.go_to_page("Cover Letter")
        return
    if not bool(jd_quality.get("reliable_scoring_ready", False)):
        st.markdown("**Cover letter needs a complete job description**")
        st.caption(
            "Complete and verify the saved posting first so the draft is grounded in the "
            "employer's actual requirements."
        )
        if st.button(
            "Go to Job description",
            key=f"cover_letter_to_jd_{safe_slug(str(selected_path))}",
            icon=":material/arrow_forward:",
            type="primary",
        ):
            st.session_state["selected_review_tab"] = "JD"
            st.session_state["review_detail_tabs"] = REVIEW_SECTION_LABELS["JD"]
            st.rerun()
        return
    metadata = parse_job_metadata(selected_path)
    file_key = safe_slug(str(selected_path))
    latest_fields = verification_from_markdown(selected_path)
    default_company = str(latest_fields.get("company_normalized") or metadata.get("company", ""))
    needs_confirmation = not services.company_generation_allowed(latest_fields)
    st.caption(
        f"{default_company or 'Employer not confirmed'} · "
        f"{metadata.get('role', 'Role not confirmed')} · "
        f"{normalize_location(metadata.get('location', '')) or 'Location not confirmed'}"
    )
    with st.expander("Edit employer details", expanded=needs_confirmation):
        company_fields = services.render_markdown_company_confirmation(
            selected_path, key_prefix=f"job_desc_{safe_slug(str(selected_path))}"
        )
        default_company = str(company_fields.get("company_normalized") or default_company)
        left, right = st.columns(2)
        with left:
            company = st.text_input(
                "Editable company", value=default_company, key=f"company_{file_key}"
            )
            location = st.text_input(
                "Location override",
                value=normalize_location(metadata.get("location", "")),
                key=f"location_{file_key}",
            )
        with right:
            role = st.text_input(
                "Role override", value=metadata.get("role", ""), key=f"role_{file_key}"
            )
            job_url = st.text_input(
                "Job URL override", value=metadata.get("job_url", ""), key=f"job_url_{file_key}"
            )
    if not st.button("Generate Cover Letter", key=f"generate_{file_key}", type="primary"):
        return
    if not all([company.strip(), role.strip(), location.strip(), job_url.strip()]):
        st.error(
            "Please fill in company, role, location, and job URL before generating the cover letter."
        )
        return
    latest_fields = verification_from_markdown(selected_path)
    if normalize_company_name(company) != str(latest_fields.get("company_normalized", "")):
        st.error("Confirm the edited company name before generating a cover letter.")
        return
    if not services.company_generation_allowed(latest_fields):
        st.error("Company name needs confirmation before generating a cover letter.")
        return
    try:
        summary, output = services.run_with_captured_output(
            create_application_package,
            job_description_path=selected_path,
            workspace=services.current_workspace(),
            company=company.strip(),
            role=role.strip(),
            location=location.strip(),
            job_url=job_url.strip(),
        )
        render_cover_letter_result(summary, output, services)
    except CoverLetterGenerationError as error:
        st.warning(str(error))
        for reason in error.reasons:
            st.write(f"- {reason}")
        if error.gaps:
            with st.expander("Requirements still missing reliable evidence", expanded=False):
                for gap in error.gaps:
                    st.write(f"- {gap}")
    except Exception as error:  # noqa: BLE001
        st.error(f"Could not generate the cover letter: {error}")


def render_selected_review_detail(
    job: DashboardJob,
    tracker_rows: list[TrackerRow],
    services: ReviewPageServices,
) -> None:
    """Render the selected job header and one independently testable section."""
    selected_path = Path(job["path"])
    selected_text = services.read_text_file(selected_path)
    selected_tracker_row = services.tracker_row_for_job(job, tracker_rows)
    section_labels = REVIEW_SECTION_LABELS
    if st.session_state.get("selected_review_tab") not in section_labels:
        st.session_state["selected_review_tab"] = "Overview"
    render_selected_job_context(job)
    context = render_selected_review_header(job, tracker_rows, services)
    selected_label = section_labels[st.session_state["selected_review_tab"]]
    if st.session_state.get("review_detail_tabs") not in section_labels.values():
        st.session_state["review_detail_tabs"] = selected_label
    selected_section_label = (
        st.segmented_control(
            "Detail section",
            list(section_labels.values()),
            selection_mode="single",
            key="review_detail_tabs",
            label_visibility="collapsed",
            width="content",
        )
        or selected_label
    )
    section = next(key for key, label in section_labels.items() if label == selected_section_label)
    if section != st.session_state["selected_review_tab"]:
        st.session_state["selected_review_tab"] = section
        st.rerun()
    with st.container(border=False, key="review_detail_panel"):
        if section == "Overview":
            render_review_overview_section(job, tracker_rows, selected_path, context, services)
        elif section == "Fit":
            render_review_fit_section(job, selected_text, tracker_rows, services)
        elif section == "JD":
            render_review_jd_section(
                job,
                selected_text,
                selected_path,
                selected_tracker_row,
                context["jd_quality"],
                services,
            )
        else:
            render_review_cover_letter_section(selected_path, context["jd_quality"], services)


def job_descriptions_tab(services: ReviewPageServices) -> None:
    """Render the job-description review and package generation workflow."""
    st.markdown('<div class="desktop-workspace-marker"></div>', unsafe_allow_html=True)
    render_review_workspace_styles()
    render_saved_job_delete_notice()
    all_jobs = services.load_screened_jobs()
    if not all_jobs:
        services.render_page_header(
            "Review Jobs",
            "Compare evidence, risks, and next actions—not just a single score.",
        )
        st.info("No jobs found yet. Start with Find Jobs or Add Target Job.")
        if st.button("Add Target Job", width="stretch"):
            services.go_to_page("Add Target Job")
        return
    tracker_rows = (
        []
        if services.demo_mode_enabled()
        else services.load_tracker_rows(sort_by="created_at", descending=True)
    )
    fetch_runs_by_id = {str(run.get("fetch_run_id", "")): run for run in load_fetch_runs()}
    left_panel, detail_panel = st.columns(
        [0.28, 0.72], gap="medium", vertical_alignment="top"
    )
    with left_panel, st.container(key="review_job_list_panel"):
        render_saved_jobs_header(services.go_to_page)
        render_ats_completion_action(all_jobs, services)
        state = initialize_review_state(all_jobs, tracker_rows, services)
        filters = render_review_filter_controls(all_jobs, state, services)
        filtered_jobs = filtered_review_jobs(
            all_jobs, tracker_rows, fetch_runs_by_id, filters, services
        )
        shortlist = filtered_jobs
        if not shortlist:
            render_empty_review_state(filters)
            return
        selected_job = resolve_selected_review_job(shortlist, services)
        render_saved_job_management(all_jobs, selected_job, services)
        selected_job = render_review_job_table(
            shortlist, selected_job, demo=state["is_demo"], on_select=set_review_job_selection
        )
    with detail_panel, st.container(key="review_detail_shell"):
        render_selected_review_detail(selected_job, tracker_rows, services)
