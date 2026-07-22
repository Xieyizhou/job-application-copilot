"""Review Jobs page for evidence-first comparison and cover-letter preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from apply_package import create_application_package, parse_job_metadata
from company_verification import normalize_company_name, verification_from_markdown
from dashboard_fit import build_fit_presentation, confidence_level, eligibility_status
from dashboard_regions import (
    build_region_options,
    dynamic_source_options,
    filtered_region_option_keys,
    job_matches_region_option,
    normalize_location,
    region_label,
    source_display_name,
)
from dashboard_review import (
    job_evidence_label,
    job_needs_full_jd,
    review_inbox_view_matches,
    review_job_next_action,
    sorted_review_jobs,
)
from dashboard_titles import get_job_display_title
from fetch_history import load_fetch_runs
from output_paths import safe_slug


@dataclass(frozen=True)
class ReviewPageServices:
    """Shared dashboard operations required by the Review Jobs page."""

    build_job_snippet: Callable[[dict[str, Any]], str]
    card_html: Callable[[Any, str], str]
    company_generation_allowed: Callable[[dict[str, Any]], bool]
    current_workspace: Callable[[], Any]
    default_review_inbox_view: Callable[..., str]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    key_requirements_from_text: Callable[[str], list[str]]
    load_package_notes: Callable[..., str]
    load_screened_jobs: Callable[..., list[dict[str, Any]]]
    load_tracker_rows: Callable[..., list[dict[str, Any]]]
    mark_job_not_interested: Callable[..., tuple[Any, str]]
    package_dir_for_job: Callable[..., Any]
    package_status_for_job: Callable[..., str]
    read_text_file: Callable[..., str]
    relative_path: Callable[..., str]
    render_action_callout: Callable[..., None]
    render_fit_analysis_sections: Callable[..., None]
    render_generation_success: Callable[..., None]
    render_markdown_company_confirmation: Callable[..., dict[str, Any]]
    render_page_header: Callable[[str, str | None], None]
    run_with_captured_output: Callable[..., tuple[Any, str]]
    sanitize_fit_text: Callable[[Any], str]
    save_job_to_tracker: Callable[..., tuple[Any, str]]
    save_recent_region_key: Callable[[str], None]
    tracker_row_for_job: Callable[..., Any]
    tracker_status_for_job: Callable[..., str]
    default_recommendation_limit: int
    min_recommendation_limit: int
    max_recommendation_limit: int
    show_debug_ui: bool = False


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


def render_review_action_buttons(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    key_prefix: str,
    services: ReviewPageServices,
) -> None:
    """Render selected-job actions in the detail panel."""
    action_package, action_fit, action_track = st.columns([0.44, 0.28, 0.28])
    with action_package:
        if st.button("Prepare Cover Letter", key=f"{key_prefix}_package", type="primary", width="stretch"):
            set_review_job_selection(job, "Cover Letter")
            st.rerun()
    with action_fit:
        if st.button("View Fit", key=f"{key_prefix}_fit", width="stretch"):
            set_review_job_selection(job, "Fit")
            st.rerun()
    with action_track:
        if services.demo_mode_enabled():
            st.caption("Tracker disabled in Demo workspace.")
        elif st.button("Track", key=f"{key_prefix}_track", width="stretch"):
            try:
                tracker_id, output = services.save_job_to_tracker(job)
                st.success(f"Saved to tracker #{tracker_id}.")
                if services.show_debug_ui and output:
                    with st.expander("Advanced: tracker output", expanded=False):
                        st.text(output)
            except Exception as error:  # noqa: BLE001
                st.error(str(error))
    if not services.demo_mode_enabled():
        with st.expander("More actions", expanded=False):
            if st.button("Ignore", key=f"{key_prefix}_ignore"):
                try:
                    tracker_id, output = services.mark_job_not_interested(job, tracker_rows)
                    st.success(f"Marked tracker #{tracker_id} as not interested.")
                    if services.show_debug_ui and output:
                        with st.expander("Advanced: tracker output", expanded=False):
                            st.text(output)
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))


def render_job_result_cards(
    jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    services: ReviewPageServices,
) -> None:
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
        tracker_status = services.tracker_status_for_job(job, tracker_rows)
        package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
        fit_presentation = build_fit_presentation(job)
        next_action = review_job_next_action(job, tracker_status, package_status)
        with st.container(border=True):
            st.markdown(
                services.card_html(job["company"], "job-card-company")
                + services.card_html(get_job_display_title(job), "job-card-role")
                + services.card_html(
                    f"{job['normalized_location']} | {source_display_name(str(job['source']))}",
                    "job-card-meta",
                )
                + services.card_html(
                    f"{fit_presentation['card_status']} · {tracker_status} · {package_status}",
                    "job-card-status",
                ),
                unsafe_allow_html=True,
            )
            st.caption(job_evidence_label(job))
            st.caption(f"Next: {next_action}")

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
                    "Cover Letter",
                    key=f"view_package_{file_key}_{index}",
                    width="stretch",
                    disabled=services.demo_mode_enabled() and package_status != "Demo cover letter",
                ):
                    set_review_job_selection(job, "Cover Letter")
                    st.rerun()


def job_descriptions_tab(services: ReviewPageServices) -> None:
    """Render the job-description review and package generation workflow."""
    services.render_page_header("Review Jobs", "Compare evidence, risks, and next actions—not just a single score.")
    if services.demo_mode_enabled():
        st.info(
            "Demo workspace uses fictional, read-only data. "
            "All Jobs is shown by default so you can compare different scoring outcomes."
        )

    all_jobs = services.load_screened_jobs()

    if not all_jobs:
        st.info("No jobs found yet. Start with Find Jobs or Add Target Job.")
        if st.button("Add Target Job", width="stretch"):
            services.go_to_page("Add Target Job")
        return

    fetch_runs = load_fetch_runs()
    fetch_runs_by_id = {str(run.get("fetch_run_id", "")): run for run in fetch_runs}
    tracker_rows = [] if services.demo_mode_enabled() else services.load_tracker_rows(sort_by="created_at", descending=True)
    is_demo = services.demo_mode_enabled()
    default_inbox_view = services.default_review_inbox_view(all_jobs, tracker_rows, demo=is_demo)
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

    inbox_view_options = ["Recommended", "Needs Review", "Cover Letter Ready", "Not Tracked", "Ignored", "All Jobs"]
    sort_options = ["Role Fit high to low", "Newest first", "Recommendation", "Company A-Z", "Cover letter status", "Tracker status"]
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
    st.session_state.setdefault("job_tab_shortlist_limit", services.default_recommendation_limit)
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
            min_value=services.min_recommendation_limit,
            max_value=services.max_recommendation_limit,
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
        services.save_recent_region_key(selected_region_key)

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
        tracker_status = services.tracker_status_for_job(job, tracker_rows)
        package_status = services.package_status_for_job(job, tracker_rows)
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
        "Cover letter status": "cover letter status",
        "Tracker status": "tracker status",
    }.get(sort_by, sort_by.lower())
    summary_parts = [
        f"{len(shortlist)} jobs shown",
        inbox_view,
        f"Sorted by {sort_summary}",
        filter_summary,
    ]

    overview_metrics = st.columns(4)
    overview_metrics[0].metric("Matching filters", len(filtered_jobs))
    overview_metrics[1].metric(
        "Reliable scores",
        sum(1 for job in filtered_jobs if confidence_level(job.get("confidence")) in {"medium", "high"}),
    )
    overview_metrics[2].metric(
        "Need full JD",
        sum(1 for job in filtered_jobs if job_needs_full_jd(job)),
    )
    overview_metrics[3].metric(
        "Cover letters ready",
        sum(1 for job in filtered_jobs if str(job.get("package_status", "")).lower() not in {"", "no cover letter", "not generated"}),
    )

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
                services.go_to_page("Add Target Job")
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
    selected_text = services.read_text_file(selected_path)
    selected_tracker_row = services.tracker_row_for_job(selected_job, tracker_rows)

    sections = ["Overview", "Fit", "JD", "Cover Letter"]
    if st.session_state.get("selected_review_tab") not in sections:
        st.session_state["selected_review_tab"] = "Overview"

    left_col, right_col = st.columns([0.42, 0.58], gap="large")
    with left_col:
        st.caption(" · ".join(summary_parts))
        render_job_result_cards(shortlist, tracker_rows, services)
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
                        "Tracker": job.get("tracker_status") or services.tracker_status_for_job(job, tracker_rows),
                        "Cover Letter": job.get("package_status") or services.package_status_for_job(job, tracker_rows),
                    }
                    for job in shortlist
                ],
                width="stretch",
                hide_index=True,
            )

    with right_col:
        tracker_status = services.tracker_status_for_job(selected_job, tracker_rows)
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
            st.caption("Recommendation")
            st.markdown(f"**{selected_job['recommendation']}**")
            st.caption(f"Tracker: {tracker_status}")
            st.caption(f"Cover letter: {selected_job.get('package_status') or services.package_status_for_job(selected_job, tracker_rows)}")

        selected_confidence = dict(selected_job.get("confidence", {}) or {})
        selected_package_status = selected_job.get("package_status") or services.package_status_for_job(selected_job, tracker_rows)
        selected_next_action = review_job_next_action(selected_job, tracker_status, selected_package_status)
        learned_signal = dict(selected_job.get("ml_relevance", {}) or {})
        selected_jd_quality = dict(selected_job.get("jd_quality", {}) or {})
        show_learned_signal = bool(
            learned_signal.get("available")
            and learned_signal.get("displayable", True)
            and selected_jd_quality.get("reliable_scoring_ready", False)
        )
        fit_metrics = st.columns(5 if show_learned_signal else 4)
        fit_metrics[0].metric(
            "Role Fit",
            f"{selected_job['score']}/100" if selected_job.get("score") is not None else "—",
            help="Evidence-calibrated fit used for ranking.",
        )
        selected_coverage = selected_presentation.get("coverage_score")
        fit_metrics[1].metric(
            "Observed coverage",
            f"{int(selected_coverage)}%" if selected_coverage is not None else "—",
            help="Coverage among recognized requirements only.",
        )
        fit_metrics[2].metric(
            "Evidence",
            f"{int(selected_confidence.get('active_requirement_count', 0) or 0)} reqs",
        )
        fit_metrics[3].metric("Confidence", confidence_level(selected_confidence).title())
        if show_learned_signal:
            fit_metrics[4].metric(
                "Local ML signal",
                f"{float(learned_signal.get('probability', 0.0)):.0%}",
                help=(
                    "Experimental auxiliary relevance estimate trained on synthetic pairs. "
                    "It does not affect Role Fit, eligibility, sorting, or recommendation."
                ),
            )
        if selected_jd_quality:
            st.caption(
                f"JD quality: {selected_jd_quality.get('display_label', 'Needs review')}"
            )
        services.render_action_callout(
            selected_next_action,
            caution=confidence_level(selected_confidence) == "low" or eligibility_status(selected_job) == "failed",
        )

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
            snippet = services.build_job_snippet(selected_job)
            if snippet:
                st.caption(snippet)
            selected_analysis = dict(selected_job.get("analysis_result", {}))
            st.write(f"Main reason: {selected_analysis.get('main_reason', selected_presentation['card_status'])}")
            main_risk = str(selected_analysis.get("main_risk", "")) or (
                selected_job["red_flags_text"] if selected_job["red_flags_text"] != "-" else selected_job["warnings_text"]
            )
            if main_risk != "-":
                st.write(f"Main risk: {main_risk}")
            st.caption(job_evidence_label(selected_job))
            render_review_action_buttons(
                selected_job,
                tracker_rows,
                key_prefix=f"overview_actions_{safe_slug(str(selected_path))}",
                services=services,
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
            services.render_fit_analysis_sections(selected_job, selected_text)
            requirements = services.key_requirements_from_text(selected_text)
            if requirements:
                st.markdown("**Key Requirements**")
                for requirement in requirements:
                    st.write(f"- {requirement}")
            package_dir = services.package_dir_for_job(selected_job, tracker_rows)
            if package_dir:
                suggestions = services.load_package_notes(package_dir)
                if suggestions:
                    with st.expander("Resume / Cover Letter suggestions", expanded=False):
                        st.markdown(services.sanitize_fit_text(suggestions))

        elif selected_section == "JD":
            if selected_job.get("job_url"):
                st.link_button("Open Job URL", str(selected_job["job_url"]))
            st.caption(f"Source: {source_display_name(str(selected_job['source']))}")
            st.text(selected_text or selected_job["preview"])
            if services.show_debug_ui:
                with st.expander("Advanced: job metadata", expanded=False):
                    st.write(f"Markdown path: `{services.relative_path(selected_path)}`")
                    st.write(f"New status: {selected_job.get('new_label', '-')}")
                    st.write(f"First seen: {selected_job.get('first_seen_at', '-')}")
                    st.write(f"Last seen: {selected_job.get('last_seen_at', '-')}")
                    st.write(f"Search run date: {selected_job.get('fetch_run_date', '-')}")
                    if selected_tracker_row:
                        st.write(f"Tracker id: {selected_tracker_row['id']}")

        elif selected_section == "Cover Letter":
            if services.demo_mode_enabled():
                st.info("Demo workspace does not generate new files. Open Cover Letter to view the sanitized sample draft.")
                return

            metadata = parse_job_metadata(selected_path)
            file_key = safe_slug(str(selected_path))

            with st.expander("Cover letter options", expanded=True):
                st.markdown("**Company verification**")
                selected_company_fields = services.render_markdown_company_confirmation(
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

                if st.button("Generate Cover Letter", key=f"generate_{file_key}", type="primary"):
                    if not all([company.strip(), role.strip(), location.strip(), job_url.strip()]):
                        st.error("Please fill in company, role, location, and job URL before generating the cover letter.")
                        return
                    latest_company_fields = verification_from_markdown(selected_path)
                    if normalize_company_name(company) != str(latest_company_fields.get("company_normalized", "")):
                        st.error("Confirm the edited company name before generating a cover letter.")
                        return
                    if not services.company_generation_allowed(latest_company_fields):
                        st.error(
                            "Company name needs confirmation before generating a cover letter. "
                            "This prevents using the wrong company name in your application."
                        )
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
                        services.render_generation_success(summary)
                        st.write(f"Overall score: {summary['match_score']}/100")
                        st.write(f"Recommendation: {summary['recommendation']}")
                        st.write(f"Tracker id: {summary['tracker_id']}")
                        st.write(f"Cover letter DOCX: `{services.relative_path(summary['cover_letter_docx_path'])}`")
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
                    except Exception as error:  # noqa: BLE001
                        st.error(f"Could not generate the cover letter: {error}")
