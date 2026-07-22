"""Review Jobs page for evidence-first comparison and cover-letter preparation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
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
    enrich_saved_job_description: Callable[..., dict[str, Any]]
    jsearch_configured: Callable[[], bool]
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
        needs_full_jd = job_needs_full_jd(job)
        action_label = "Find Full JD" if needs_full_jd else "Prepare Cover Letter"
        if st.button(action_label, key=f"{key_prefix}_package", type="primary", width="stretch"):
            set_review_job_selection(job, "JD" if needs_full_jd else "Cover Letter")
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


def render_full_jd_recovery(
    selected_path: Path,
    quality: dict[str, Any],
    *,
    key_prefix: str,
    services: ReviewPageServices,
) -> bool:
    """Render one concise recovery action and return current readiness."""
    if quality.get("reliable_scoring_ready", False):
        return True
    st.warning(
        f"A full JD is required before generating a cover letter. "
        f"Current quality: {quality.get('display_label', 'Needs review')}."
    )
    st.caption(str(quality.get("next_action", "Add the complete responsibilities and requirements.")))
    if not services.jsearch_configured():
        st.info("Configure JSEARCH_API_KEY to find the full posting automatically, or replace this record with the complete JD.")
        return False
    if st.button("Find and verify full JD", key=f"{key_prefix}_find_full_jd", type="primary"):
        try:
            result = services.enrich_saved_job_description(selected_path)
        except Exception as error:  # noqa: BLE001
            st.error(f"Full-JD lookup failed: {error}")
            return False
        if result.get("updated"):
            st.success(str(result["message"]))
            st.rerun()
        else:
            st.warning(str(result.get("message", "No safe full-JD match was found.")))
    return False


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


def initialize_review_state(
    all_jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    services: ReviewPageServices,
) -> dict[str, Any]:
    """Initialize workspace-specific review defaults and return option models."""
    is_demo = services.demo_mode_enabled()
    default_inbox_view = services.default_review_inbox_view(all_jobs, tracker_rows, demo=is_demo)
    workspace_state_key = "demo" if is_demo else "personal"
    if st.session_state.get("review_workspace_mode") != workspace_state_key:
        st.session_state["review_workspace_mode"] = workspace_state_key
        st.session_state["review_inbox_view"] = default_inbox_view
        if is_demo:
            st.session_state.update(
                review_minimum_score=0,
                review_hide_hard_red_flags=False,
                review_hide_degree_required=False,
                review_hide_current_student_only=False,
            )
    inbox_options = ["Recommended", "Needs Review", "Cover Letter Ready", "Not Tracked", "Ignored", "All Jobs"]
    sort_options = ["Role Fit high to low", "Newest first", "Recommendation", "Company A-Z", "Cover letter status", "Tracker status"]
    recommendation_options = [
        "all", "Apply", "Apply / Maybe Apply", "Maybe Apply", "Manual Review",
        "Skip or Low Priority", "Skip / Not Eligible",
    ]
    source_options = dynamic_source_options(all_jobs)
    defaults = {
        "review_inbox_view": default_inbox_view,
        "review_sort_by": "Role Fit high to low",
        "review_source_filter": "all",
        "review_recommendation_filter": "all",
        "review_search_text": "",
        "region_search_query": "",
        "job_tab_shortlist_limit": services.default_recommendation_limit,
        "review_minimum_score": 50,
        "review_hide_hard_red_flags": True,
        "review_hide_degree_required": True,
        "review_hide_current_student_only": True,
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    if st.session_state["review_inbox_view"] not in inbox_options:
        st.session_state["review_inbox_view"] = default_inbox_view
    if st.session_state["review_sort_by"] not in sort_options:
        st.session_state["review_sort_by"] = "Role Fit high to low"
    if st.session_state["review_source_filter"] not in source_options:
        st.session_state["review_source_filter"] = "all"
    if st.session_state["review_recommendation_filter"] not in recommendation_options:
        st.session_state["review_recommendation_filter"] = "all"
    return {
        "is_demo": is_demo,
        "default_inbox_view": default_inbox_view,
        "inbox_options": inbox_options,
        "sort_options": sort_options,
        "source_options": source_options,
        "recommendation_options": recommendation_options,
    }


def reset_review_filters(default_inbox_view: str, is_demo: bool, *, show_all: bool = False) -> None:
    """Reset review widgets to workspace defaults."""
    st.session_state.update(
        review_search_text="",
        review_inbox_view="All Jobs" if show_all else default_inbox_view,
        review_sort_by="Role Fit high to low",
        selected_region_key="all",
        review_source_filter="all",
        review_recommendation_filter="all",
        review_minimum_score=0 if show_all or is_demo else 50,
        review_hide_hard_red_flags=False if show_all else not is_demo,
        review_hide_degree_required=False if show_all else not is_demo,
        review_hide_current_student_only=False if show_all else not is_demo,
    )


def render_review_filter_controls(
    all_jobs: list[dict[str, Any]],
    state: dict[str, Any],
    services: ReviewPageServices,
) -> dict[str, Any]:
    """Render review filters and return their current values plus reset callbacks."""
    clear_filters = lambda: reset_review_filters(state["default_inbox_view"], state["is_demo"])
    show_all = lambda: reset_review_filters(state["default_inbox_view"], state["is_demo"], show_all=True)
    inbox_view = st.segmented_control(
        "Job inbox view", state["inbox_options"], key="review_inbox_view", selection_mode="single"
    ) or state["default_inbox_view"]
    search_col, sort_col, count_col = st.columns([0.42, 0.32, 0.26])
    with search_col:
        search_text = st.text_input("Search company or role", key="review_search_text")
    with sort_col:
        sort_by = st.selectbox("Sort by", state["sort_options"], key="review_sort_by")
    with count_col:
        recommendation_limit = st.slider(
            "Number of recommendations", min_value=services.min_recommendation_limit,
            max_value=services.max_recommendation_limit, key="job_tab_shortlist_limit",
            help="How many ranked jobs to display after filtering and duplicate removal.",
        )
    options_by_key = build_region_options(all_jobs)
    with st.expander("Filters", expanded=False):
        first_left, first_right = st.columns(2)
        with first_left:
            region_query = st.text_input("Search region", placeholder="Beijing, China, Remote", key="region_search_query")
        region_keys = filtered_region_option_keys(options_by_key, region_query)
        if region_query.strip() and region_keys == ["all"]:
            st.info("No matching regions found.")
        selected_key = st.session_state.get("selected_region_key", "all")
        if selected_key not in region_keys and selected_key in options_by_key:
            region_keys = [selected_key, *region_keys]
        with first_right:
            selected_region_key = st.selectbox(
                "Region", region_keys, index=region_keys.index(selected_key) if selected_key in region_keys else 0,
                key="selected_region_key", format_func=lambda key: options_by_key.get(key, options_by_key["all"])["label"],
            )
        selected_region = options_by_key.get(selected_region_key, options_by_key["all"])
        services.save_recent_region_key(selected_region_key)
        second_left, second_right = st.columns(2)
        with second_left:
            source_filter = st.selectbox("Source", state["source_options"], key="review_source_filter")
        with second_right:
            recommendation_filter = st.selectbox(
                "Recommendation", state["recommendation_options"], key="review_recommendation_filter"
            )
        third_left, third_right = st.columns(2)
        with third_left:
            minimum_score = st.slider("Minimum Role Fit score", min_value=0, max_value=100, key="review_minimum_score")
        with third_right:
            st.caption(f"Showing up to {recommendation_limit} recommendations.")
            st.button("Clear filters", key="review_clear_filters", on_click=clear_filters, width="stretch")
        flag_left, flag_middle, flag_right = st.columns(3)
        with flag_left:
            hide_hard = st.checkbox("Hide hard red flags", key="review_hide_hard_red_flags")
        with flag_middle:
            hide_degree = st.checkbox("Hide PhD / Master's required roles", key="review_hide_degree_required")
        with flag_right:
            hide_student = st.checkbox("Hide current-student-only internships", key="review_hide_current_student_only")
    return {
        "inbox_view": inbox_view, "search_text": search_text, "sort_by": sort_by,
        "recommendation_limit": recommendation_limit, "selected_region_key": selected_region_key,
        "selected_region": selected_region, "source_filter": source_filter,
        "recommendation_filter": recommendation_filter, "minimum_score": minimum_score,
        "hide_hard": hide_hard, "hide_degree": hide_degree, "hide_student": hide_student,
        "clear_filters": clear_filters, "show_all": show_all,
    }


def review_job_matches_filters(
    job: dict[str, Any],
    filters: dict[str, Any],
    tracker_status: str,
    package_status: str,
) -> bool:
    """Return whether one job satisfies the current review filters."""
    if not review_inbox_view_matches(job, filters["inbox_view"], tracker_status, package_status):
        return False
    if not job_matches_region_option(job, filters["selected_region"]):
        return False
    if filters["source_filter"] != "all" and source_display_name(str(job["source"])) != filters["source_filter"]:
        return False
    if filters["recommendation_filter"] != "all" and job["recommendation"] != filters["recommendation_filter"]:
        return False
    if job.get("score") is None and filters["minimum_score"] > 0:
        return False
    if job.get("score") is not None and int(job["score"]) < filters["minimum_score"]:
        return False
    if filters["hide_hard"] and job["hard_red_flag"]:
        return False
    if filters["hide_degree"] and any("PhD" in flag or "Master" in flag or "Graduate degree" in flag for flag in job["red_flags"]):
        return False
    if filters["hide_student"] and any(
        marker in flag.lower() for flag in job["red_flags"] for marker in ("enrolled", "school", "penultimate")
    ):
        return False
    needle = filters["search_text"].strip().lower()
    return not needle or needle in f"{job['company']} {job['role']} {get_job_display_title(job)}".lower()


def filtered_review_jobs(
    all_jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    fetch_runs_by_id: dict[str, dict[str, Any]],
    filters: dict[str, Any],
    services: ReviewPageServices,
) -> list[dict[str, Any]]:
    """Filter and enrich jobs without mutating the loaded records."""
    filtered: list[dict[str, Any]] = []
    for original in all_jobs:
        tracker_status = services.tracker_status_for_job(original, tracker_rows)
        package_status = services.package_status_for_job(original, tracker_rows)
        if not review_job_matches_filters(original, filters, tracker_status, package_status):
            continue
        job = dict(original)
        fetch_run = fetch_runs_by_id.get(str(job.get("latest_fetch_run_id", "")), {})
        job.update(
            fetch_run_date=str(fetch_run.get("created_at", job.get("last_seen_at", ""))),
            new_label="Saved job",
            red_flags_text=" | ".join(job["red_flags"]) or "-",
            warnings_text=" | ".join(job["warnings"]) or "-",
            tracker_status=tracker_status,
            package_status=package_status,
        )
        filtered.append(job)
    return sorted_review_jobs(filtered, filters["sort_by"])


def review_summary_parts(shortlist: list[dict[str, Any]], filters: dict[str, Any]) -> list[str]:
    """Build a compact summary of active filters and ordering."""
    filter_summary = f"Role Fit {filters['minimum_score']}+"
    if filters["selected_region_key"] != "all":
        filter_summary += f" · {region_label(filters['selected_region'], include_count=False)}"
    if filters["source_filter"] != "all":
        filter_summary += f" · {filters['source_filter']}"
    if filters["recommendation_filter"] != "all":
        filter_summary += f" · {filters['recommendation_filter']}"
    sort_label = {
        "Role Fit high to low": "Role Fit", "Newest first": "newest",
        "Recommendation": "recommendation", "Company A-Z": "company",
        "Cover letter status": "cover letter status", "Tracker status": "tracker status",
    }.get(filters["sort_by"], filters["sort_by"].lower())
    return [f"{len(shortlist)} jobs shown", filters["inbox_view"], f"Sorted by {sort_label}", filter_summary]


def render_review_metrics(filtered_jobs: list[dict[str, Any]]) -> None:
    """Render the four operational review counters."""
    metrics = st.columns(4)
    metrics[0].metric("Matching filters", len(filtered_jobs))
    metrics[1].metric(
        "Reliable scores", sum(1 for job in filtered_jobs if confidence_level(job.get("confidence")) in {"medium", "high"})
    )
    metrics[2].metric("Need full JD", sum(1 for job in filtered_jobs if job_needs_full_jd(job)))
    metrics[3].metric(
        "Cover letters ready",
        sum(1 for job in filtered_jobs if str(job.get("package_status", "")).lower() not in {"", "no cover letter", "not generated"}),
    )


def render_empty_review_state(filters: dict[str, Any], services: ReviewPageServices) -> None:
    """Render recovery actions when filters yield no jobs."""
    st.info("No jobs match the current filters.")
    left, middle, right = st.columns(3)
    with left:
        st.button("Clear filters", key="review_empty_clear_filters", on_click=filters["clear_filters"], width="stretch")
    with middle:
        st.button("Show All Jobs", on_click=filters["show_all"], width="stretch")
    with right:
        if st.button("Add Target Job", width="stretch"):
            services.go_to_page("Add Target Job")


def resolve_selected_review_job(shortlist: list[dict[str, Any]], services: ReviewPageServices) -> dict[str, Any]:
    """Resolve and persist the selected job, text, and tracker row."""
    paths = [str(job["path"]) for job in shortlist]
    label_state = st.session_state.get("selected_review_job_label")
    path_state = st.session_state.get("selected_review_job_path", paths[0])
    job = resolve_review_job_selection(shortlist, label_state, path_state)
    if str(job["path"]) != path_state and job["label"] != label_state:
        set_review_job_selection(shortlist[0], st.session_state.get("selected_review_tab", "Overview"))
    st.session_state["selected_review_job_path"] = str(job["path"])
    st.session_state["selected_review_job_label"] = job["label"]
    return job


def render_review_list_column(
    shortlist: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    summary_parts: list[str],
    services: ReviewPageServices,
) -> None:
    """Render cards plus the optional compact comparison table."""
    st.caption(" · ".join(summary_parts))
    render_job_result_cards(shortlist, tracker_rows, services)
    with st.expander("Compact table view", expanded=False):
        st.dataframe(
            [
                {
                    "Company": job["company"], "Role": get_job_display_title(job),
                    "Location": job["normalized_location"], "Source": source_display_name(str(job["source"])),
                    "Role Fit": build_fit_presentation(job)["role_fit"], "Recommendation": job["recommendation"],
                    "Tracker": job.get("tracker_status") or services.tracker_status_for_job(job, tracker_rows),
                    "Cover Letter": job.get("package_status") or services.package_status_for_job(job, tracker_rows),
                }
                for job in shortlist
            ],
            width="stretch", hide_index=True,
        )


def render_selected_review_header(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    services: ReviewPageServices,
) -> dict[str, Any]:
    """Render selected-job identity, fit metrics, and next action."""
    tracker_status = services.tracker_status_for_job(job, tracker_rows)
    presentation = build_fit_presentation(job)
    header_left, header_right = st.columns([0.72, 0.28])
    with header_left:
        st.markdown(f"**{job['company']}**")
        st.write(get_job_display_title(job))
        st.caption(f"{job['normalized_location']} | {source_display_name(str(job['source']))}")
    with header_right:
        st.caption("Recommendation")
        st.markdown(f"**{job['recommendation']}**")
        st.caption(f"Tracker: {tracker_status}")
        st.caption(f"Cover letter: {job.get('package_status') or services.package_status_for_job(job, tracker_rows)}")
    confidence = dict(job.get("confidence", {}) or {})
    package_status = job.get("package_status") or services.package_status_for_job(job, tracker_rows)
    learned_signal = dict(job.get("ml_relevance", {}) or {})
    jd_quality = dict(job.get("jd_quality", {}) or {})
    show_learned = bool(
        learned_signal.get("available") and learned_signal.get("displayable", True)
        and jd_quality.get("reliable_scoring_ready", False)
    )
    metrics = st.columns(5 if show_learned else 4)
    metrics[0].metric("Role Fit", f"{job['score']}/100" if job.get("score") is not None else "—", help="Evidence-calibrated fit used for ranking.")
    coverage = presentation.get("coverage_score")
    metrics[1].metric("Observed coverage", f"{int(coverage)}%" if coverage is not None else "—", help="Coverage among recognized requirements only.")
    metrics[2].metric("Evidence", f"{int(confidence.get('active_requirement_count', 0) or 0)} reqs")
    metrics[3].metric("Confidence", confidence_level(confidence).title())
    if show_learned:
        metrics[4].metric(
            "Local ML signal", f"{float(learned_signal.get('probability', 0.0)):.0%}",
            help="Experimental auxiliary estimate. It does not affect Role Fit, eligibility, sorting, or recommendation.",
        )
    if jd_quality:
        st.caption(f"JD quality: {jd_quality.get('display_label', 'Needs review')}")
    services.render_action_callout(
        review_job_next_action(job, tracker_status, package_status),
        caution=confidence_level(confidence) == "low" or eligibility_status(job) == "failed",
    )
    return {
        "tracker_status": tracker_status, "presentation": presentation,
        "confidence": confidence, "package_status": package_status, "jd_quality": jd_quality,
    }


def render_review_overview_section(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    selected_path: Path,
    context: dict[str, Any],
    services: ReviewPageServices,
) -> None:
    """Render the concise decision overview and primary actions."""
    snippet = services.build_job_snippet(job)
    if snippet:
        st.caption(snippet)
    analysis = dict(job.get("analysis_result", {}))
    st.write(f"Main reason: {analysis.get('main_reason', context['presentation']['card_status'])}")
    main_risk = str(analysis.get("main_risk", "")) or (
        job["red_flags_text"] if job["red_flags_text"] != "-" else job["warnings_text"]
    )
    if main_risk != "-":
        st.write(f"Main risk: {main_risk}")
    st.caption(job_evidence_label(job))
    render_review_action_buttons(
        job, tracker_rows, key_prefix=f"overview_actions_{safe_slug(str(selected_path))}", services=services
    )
    notes = " | ".join(
        value for value in [job.get("red_flags_text", "-"), job.get("warnings_text", "-")]
        if value and value != "-"
    )
    if notes and notes != main_risk:
        with st.expander("Review notes", expanded=False):
            st.write(notes)


def render_review_fit_section(
    job: dict[str, Any], job_text: str, tracker_rows: list[dict[str, Any]], services: ReviewPageServices
) -> None:
    """Render explainable fit evidence and saved suggestions."""
    services.render_fit_analysis_sections(job, job_text)
    requirements = services.key_requirements_from_text(job_text)
    if requirements:
        st.markdown("**Key Requirements**")
        for requirement in requirements:
            st.write(f"- {requirement}")
    package_dir = services.package_dir_for_job(job, tracker_rows)
    if package_dir:
        suggestions = services.load_package_notes(package_dir)
        if suggestions:
            with st.expander("Resume / Cover Letter suggestions", expanded=False):
                st.markdown(services.sanitize_fit_text(suggestions))


def render_review_jd_section(
    job: dict[str, Any],
    job_text: str,
    selected_path: Path,
    selected_tracker_row: dict[str, Any] | None,
    jd_quality: dict[str, Any],
    services: ReviewPageServices,
) -> None:
    """Render saved JD text, recovery action, and optional provenance."""
    if job.get("job_url"):
        st.link_button("Open Job URL", str(job["job_url"]))
    st.caption(f"Source: {source_display_name(str(job['source']))}")
    st.text(job_text or job["preview"])
    render_full_jd_recovery(
        selected_path, jd_quality, key_prefix=f"jd_section_{safe_slug(str(selected_path))}", services=services
    )
    if services.show_debug_ui:
        with st.expander("Advanced: job metadata", expanded=False):
            st.write(f"Markdown path: {services.relative_path(selected_path)}")
            st.write(f"New status: {job.get('new_label', '-')}")
            st.write(f"First seen: {job.get('first_seen_at', '-')}")
            st.write(f"Last seen: {job.get('last_seen_at', '-')}")
            st.write(f"Search run date: {job.get('fetch_run_date', '-')}")
            if selected_tracker_row:
                st.write(f"Tracker id: {selected_tracker_row['id']}")


def render_cover_letter_result(summary: dict[str, Any], output: str, services: ReviewPageServices) -> None:
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
        st.info("Demo workspace does not generate new files. Open Cover Letter to view the sanitized sample draft.")
        return
    if not render_full_jd_recovery(
        selected_path, jd_quality, key_prefix=f"cover_letter_{safe_slug(str(selected_path))}", services=services
    ):
        return
    metadata = parse_job_metadata(selected_path)
    file_key = safe_slug(str(selected_path))
    with st.expander("Cover letter options", expanded=True):
        st.markdown("**Company verification**")
        company_fields = services.render_markdown_company_confirmation(
            selected_path, key_prefix=f"job_desc_{safe_slug(str(selected_path))}"
        )
        default_company = str(company_fields.get("company_normalized") or metadata.get("company", ""))
        st.write(f"Confirmed company: {default_company or '-'}")
        left, right = st.columns(2)
        with left:
            company = st.text_input("Editable company", value=default_company, key=f"company_{file_key}")
            location = st.text_input("Location override", value=normalize_location(metadata.get("location", "")), key=f"location_{file_key}")
        with right:
            role = st.text_input("Role override", value=metadata.get("role", ""), key=f"role_{file_key}")
            job_url = st.text_input("Job URL override", value=metadata.get("job_url", ""), key=f"job_url_{file_key}")
        if not st.button("Generate Cover Letter", key=f"generate_{file_key}", type="primary"):
            return
        if not all([company.strip(), role.strip(), location.strip(), job_url.strip()]):
            st.error("Please fill in company, role, location, and job URL before generating the cover letter.")
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
                create_application_package, job_description_path=selected_path,
                workspace=services.current_workspace(), company=company.strip(), role=role.strip(),
                location=location.strip(), job_url=job_url.strip(),
            )
            render_cover_letter_result(summary, output, services)
        except Exception as error:  # noqa: BLE001
            st.error(f"Could not generate the cover letter: {error}")


def render_selected_review_detail(
    job: dict[str, Any],
    tracker_rows: list[dict[str, Any]],
    services: ReviewPageServices,
) -> None:
    """Render the selected job header and one independently testable section."""
    selected_path = Path(job["path"])
    selected_text = services.read_text_file(selected_path)
    selected_tracker_row = services.tracker_row_for_job(job, tracker_rows)
    sections = ["Overview", "Fit", "JD", "Cover Letter"]
    if st.session_state.get("selected_review_tab") not in sections:
        st.session_state["selected_review_tab"] = "Overview"
    context = render_selected_review_header(job, tracker_rows, services)
    section = st.segmented_control(
        "Detail section", sections, default=st.session_state["selected_review_tab"],
        selection_mode="single", label_visibility="collapsed",
    ) or "Overview"
    if section != st.session_state["selected_review_tab"]:
        st.session_state["selected_review_tab"] = section
        st.rerun()
    if section == "Overview":
        render_review_overview_section(job, tracker_rows, selected_path, context, services)
    elif section == "Fit":
        render_review_fit_section(job, selected_text, tracker_rows, services)
    elif section == "JD":
        render_review_jd_section(job, selected_text, selected_path, selected_tracker_row, context["jd_quality"], services)
    else:
        render_review_cover_letter_section(selected_path, context["jd_quality"], services)


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
    tracker_rows = [] if services.demo_mode_enabled() else services.load_tracker_rows(sort_by="created_at", descending=True)
    fetch_runs_by_id = {str(run.get("fetch_run_id", "")): run for run in load_fetch_runs()}
    state = initialize_review_state(all_jobs, tracker_rows, services)
    filters = render_review_filter_controls(all_jobs, state, services)
    filtered_jobs = filtered_review_jobs(all_jobs, tracker_rows, fetch_runs_by_id, filters, services)
    shortlist = filtered_jobs[: filters["recommendation_limit"]]
    render_review_metrics(filtered_jobs)
    if not shortlist:
        render_empty_review_state(filters, services)
        return
    if len(shortlist) == 1 and len(all_jobs) > 1:
        st.caption("Only 1 job shown. Clear filters or switch to All Jobs to see more.")
        left, right = st.columns(2)
        with left:
            st.button("Clear filters", on_click=filters["clear_filters"], width="stretch")
        with right:
            st.button("Show All Jobs", on_click=filters["show_all"], width="stretch")
    selected_job = resolve_selected_review_job(shortlist, services)
    left_col, right_col = st.columns([0.42, 0.58], gap="large")
    with left_col:
        render_review_list_column(shortlist, tracker_rows, review_summary_parts(shortlist, filters), services)
    with right_col:
        render_selected_review_detail(selected_job, tracker_rows, services)
