"""Live job-discovery page for the local toolkit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from fetch_jobs import jsearch_configured
from company_ats import load_ats_boards
from dashboard_fetch_preferences import (
    FETCH_SOURCES,
    load_fetch_sources,
    save_fetch_sources,
)
from dashboard_fetch_controls import (
    REGION_CONFIG,
    REGION_OPTIONS,
    render_advanced_fetch_options,
    render_fetch_search_styles,
    render_region_fields,
)
from dashboard_fetch_runner import FetchSearchOutcome, run_job_search
from dashboard_regions import source_display_name
from dashboard_search_profile import search_profile_from_path


__all__ = ["REGION_CONFIG", "REGION_OPTIONS"]


DEFAULT_FETCH_LIMIT_PER_SOURCE = 20
MAX_FETCH_LIMIT_PER_SOURCE = 20
MISSING_JSEARCH_MESSAGE = (
    "For automatic full job descriptions, add JSEARCH_API_KEY to `.env`. "
    "Adzuna and Jooble can still discover jobs, but their responses contain snippets."
)
UNSUPPORTED_ADZUNA_MESSAGE = (
    "Adzuna is not available for this region. Jooble can still search this location."
)


def _workspace_root(services: FetchPageServices) -> Path:
    workspace = services.current_workspace()
    return Path(getattr(workspace, "root", Path("data/local_workspace")))


def _remember_fetch_sources(submitted: bool, sources: list[str], remembered: list[str]) -> None:
    if submitted and sources != remembered:
        save_fetch_sources(sources)

@dataclass(frozen=True)
class FetchPageServices:
    """Shared dashboard operations required by the job-discovery page."""

    demo_mode_enabled: Callable[[], bool]
    current_workspace: Callable[[], Any]
    go_to_page: Callable[[str], None]
    relocate_fetched_jobs_to_workspace: Callable[..., list[Any]]
    render_fetch_history_section: Callable[[], None]
    render_fetch_run_job_cards: Callable[[list[dict[str, Any]], str], None]
    render_fetch_run_job_table: Callable[[list[dict[str, Any]], str], None]
    render_page_header: Callable[[str, str | None], None]
    run_with_captured_output: Callable[..., tuple[Any, str]]
    default_recommendation_limit: int
    min_recommendation_limit: int
    max_recommendation_limit: int
    show_debug_ui: bool = False


@dataclass(frozen=True)
class FetchSearchRequest:
    """Validated search-form values passed to provider orchestration."""

    submitted: bool
    query: str
    sources: list[str]
    recommendation_limit: int
    fetch_limit_per_source: int
    adzuna_country: str
    adzuna_location: str
    jooble_location: str
    adzuna_supported: bool


@dataclass(frozen=True)
class FetchResultSummary:
    """Display-focused totals derived from one provider search."""

    returned: int
    new: int
    already_seen: int
    skipped: int
    full_descriptions: int
    issues: int
    new_jobs: list[dict[str, Any]]
    seen_jobs: list[dict[str, Any]]
    skipped_jobs: list[dict[str, Any]]


def initialize_fetch_query(services: FetchPageServices) -> str:
    """Initialize the editable query from the current candidate source."""
    workspace = services.current_workspace()
    profile = search_profile_from_path(workspace.resume_source_path)
    profile_key = ":".join(
        [workspace.mode, str(workspace.resume_source_path), profile.query, *profile.keywords]
    )
    if st.session_state.get("fetch_profile_key") != profile_key:
        st.session_state["fetch_profile_key"] = profile_key
        st.session_state["fetch_query"] = profile.query
    if profile.source_ready:
        matched_keywords = ", ".join(keyword.title() for keyword in profile.keywords)
        detail = f" Resume signals: {matched_keywords}." if matched_keywords else ""
        help_text = f"Suggested from the uploaded resume; you can edit it.{detail}"
    else:
        help_text = "Upload a resume for an automatic suggestion, or enter a query manually."
    return st.text_input("Target role / query", key="fetch_query", help=help_text)


def render_fetch_region_fields(services: FetchPageServices) -> tuple[str, str, str, bool]:
    """Render region fields and return provider-specific locations."""
    return render_region_fields(st, show_debug_ui=services.show_debug_ui)


def render_fetch_options_form(
    services: FetchPageServices,
    *,
    adzuna_supported: bool,
) -> tuple[bool, list[str], int, int]:
    """Render provider and result-limit controls."""
    with st.form("fetch_jobs_form"):
        full_jd_source_ready = jsearch_configured()
        ats_source_ready = any(board.enabled for board in load_ats_boards(_workspace_root(services)))
        default_sources = (
            ["company_ats"] if ats_source_ready else ["jsearch"] if full_jd_source_ready else ["adzuna", "jooble"]
        )
        remembered_sources = load_fetch_sources(default_sources)
        sources = st.multiselect(
            "Sources",
            FETCH_SOURCES,
            default=remembered_sources,
            format_func=source_display_name,
            key="fetch_sources",
            placeholder="Choose one or more job sources",
        )
        if not full_jd_source_ready:
            st.info(MISSING_JSEARCH_MESSAGE)
        if not ats_source_ready:
            st.caption("Company ATS · Full JD becomes available after adding an employer board in Settings.")
        if "company_ats" in sources and not ats_source_ready:
            st.warning("Add and enable at least one employer ATS board in Settings first.")
        if "adzuna" in sources and not adzuna_supported:
            st.warning(UNSUPPORTED_ADZUNA_MESSAGE)
        recommendation_limit, fetch_limit_per_source = render_advanced_fetch_options(
            st,
            minimum_recommendations=services.min_recommendation_limit,
            maximum_recommendations=services.max_recommendation_limit,
            default_recommendations=services.default_recommendation_limit,
            default_per_source=DEFAULT_FETCH_LIMIT_PER_SOURCE,
            maximum_per_source=MAX_FETCH_LIMIT_PER_SOURCE,
        )
        submitted = st.form_submit_button(
            "Find Jobs",
            type="primary",
            icon=":material/search:",
            width="content",
        )
        _remember_fetch_sources(submitted, sources, remembered_sources)
    return submitted, sources, recommendation_limit, fetch_limit_per_source


def render_fetch_search_form(services: FetchPageServices) -> FetchSearchRequest:
    """Render search controls and return one immutable request."""
    with st.container(border=True, key="fetch_search_panel"):
        role_column, region_column = st.columns(2, gap="large")
        with role_column:
            query = initialize_fetch_query(services)
        with region_column:
            (
                adzuna_country,
                adzuna_location,
                jooble_location,
                adzuna_supported,
            ) = render_fetch_region_fields(services)
        submitted, sources, recommendation_limit, fetch_limit_per_source = (
            render_fetch_options_form(services, adzuna_supported=adzuna_supported)
        )
    return FetchSearchRequest(
        submitted=submitted,
        query=query,
        sources=sources,
        recommendation_limit=recommendation_limit,
        fetch_limit_per_source=fetch_limit_per_source,
        adzuna_country=adzuna_country,
        adzuna_location=adzuna_location,
        jooble_location=jooble_location,
        adzuna_supported=adzuna_supported,
    )


def fetch_request_is_valid(
    request: FetchSearchRequest,
    services: FetchPageServices,
) -> bool:
    """Fail closed on read-only or incomplete search requests."""
    if services.demo_mode_enabled():
        st.info(
            "Demo workspace does not call external job APIs. "
            "Select Personal and add API keys in `.env` for live fetch."
        )
        return False
    if not request.sources:
        st.error("Select at least one source.")
        return False
    if not request.query.strip():
        st.error("Enter a target role or search query.")
        return False
    if "company_ats" in request.sources and not any(
        board.enabled for board in load_ats_boards(_workspace_root(services))
    ):
        st.error("Company ATS is selected, but no employer boards are enabled in Settings.")
        return False
    return True


def execute_fetch_search(
    request: FetchSearchRequest,
    services: FetchPageServices,
) -> FetchSearchOutcome:
    """Execute a validated request through provider orchestration."""
    return run_job_search(
        sources=request.sources,
        query=request.query.strip(),
        country=request.adzuna_country,
        adzuna_location=request.adzuna_location,
        jooble_location=request.jooble_location,
        adzuna_supported=request.adzuna_supported,
        limit_per_source=request.fetch_limit_per_source,
        run_with_captured_output=services.run_with_captured_output,
        relocate_fetched_jobs_to_workspace=services.relocate_fetched_jobs_to_workspace,
    )


def summarize_fetch_outcome(outcome: FetchSearchOutcome) -> FetchResultSummary:
    """Aggregate provider runs without merging independent signals."""
    runs = outcome.runs
    skipped = sum(int(run.get("skipped_jobs_count", 0) or 0) for run in runs)
    return FetchResultSummary(
        returned=sum(int(run.get("total_jobs_returned", 0) or 0) for run in runs),
        new=sum(int(run.get("new_jobs_count", 0) or 0) for run in runs),
        already_seen=sum(int(run.get("duplicate_jobs_count", 0) or 0) for run in runs),
        skipped=skipped,
        full_descriptions=sum(
            int(run.get("full_descriptions_count", 0) or 0) for run in runs
        ),
        issues=len(outcome.errors) + skipped,
        new_jobs=[job for run in runs for job in (run.get("new_jobs", []) or [])],
        seen_jobs=[
            job for run in runs for job in (run.get("previously_seen_jobs", []) or [])
        ],
        skipped_jobs=[
            job for run in runs for job in (run.get("skipped_jobs", []) or [])
        ],
    )


def render_fetch_status(outcome: FetchSearchOutcome) -> None:
    """Render completion state and provider errors."""
    if outcome.runs and outcome.errors:
        st.warning("Search completed for some sources. One or more sources could not be searched.")
    elif outcome.runs:
        st.success("Search complete.")
    else:
        st.error("Search failed. Check API keys or use Demo workspace.")
    if any(
        ".env" in error or "API_KEY" in error or "APP_ID" in error or "APP_KEY" in error
        for error in outcome.errors
    ):
        st.info(
            "Live job search requires API keys. "
            "You can use Demo workspace or add keys to `.env`."
        )
    if outcome.errors:
        with st.expander("Source issues", expanded=not outcome.runs):
            for error in outcome.errors:
                st.error(error)


def render_fetch_metrics(
    outcome: FetchSearchOutcome,
    summary: FetchResultSummary,
) -> None:
    """Render one compact row of search totals."""
    result_metrics = st.columns(6)
    result_metrics[0].metric("Returned", summary.returned)
    result_metrics[1].metric("New", summary.new)
    result_metrics[2].metric("Already seen", summary.already_seen)
    result_metrics[3].metric("Skipped previews", summary.skipped)
    result_metrics[4].metric("Full JDs", summary.full_descriptions)
    result_metrics[5].metric("Source issues", len(outcome.errors))


def render_fetch_job_results(
    outcome: FetchSearchOutcome,
    summary: FetchResultSummary,
    services: FetchPageServices,
) -> None:
    """Render new jobs or the no-new-results recovery actions."""
    if summary.new == 0 and outcome.runs:
        if summary.skipped and not summary.already_seen:
            st.info(
                "No scoring-ready jobs were saved.\n\n"
                f"{summary.skipped} preview-only result(s) were skipped because they "
                "did not include a full JD or a recoverable original posting.\n\n"
                "Try JSearch · Full JD, change the query, or add a target job manually."
            )
            st.markdown("**Preview-only results · not added to Saved Jobs**")
            services.render_fetch_run_job_cards(
                summary.skipped_jobs,
                "No preview-only results in this search.",
            )
        else:
            st.info(
                "No new jobs found.\n\n"
                "Returned jobs were already seen or were preview-only results.\n\n"
                "Try broadening the query, changing region, or reviewing saved jobs."
            )
        next_left, next_right = st.columns(2)
        with next_left:
            if st.button("Review Saved Jobs", width="stretch"):
                services.go_to_page("Review Jobs")
        with next_right:
            if st.button("Add Target Job Manually", width="stretch"):
                services.go_to_page("Add Target Job")
        return
    for run in outcome.runs:
        new_jobs = run.get("new_jobs", []) or []
        if not new_jobs:
            continue
        st.markdown(f"**New jobs from {source_display_name(str(run.get('source', '')))}**")
        services.render_fetch_run_job_cards(new_jobs, "No new jobs in this search.")
    if summary.new_jobs:
        with st.expander("Compact table view", expanded=False):
            services.render_fetch_run_job_table(
                summary.new_jobs,
                "No new jobs in this search.",
            )
    if summary.skipped_jobs:
        with st.expander("Preview-only results · not saved", expanded=False):
            services.render_fetch_run_job_cards(
                summary.skipped_jobs,
                "No preview-only results in this search.",
            )


def render_fetch_details(
    outcome: FetchSearchOutcome,
    summary: FetchResultSummary,
    services: FetchPageServices,
) -> None:
    """Render optional per-source and previously-seen details."""
    if not outcome.runs:
        return
    with st.expander("Search details", expanded=False):
        st.dataframe(
            [
                {
                    "Source": source_display_name(str(run.get("source", ""))),
                    "Returned": int(run.get("total_jobs_returned", 0) or 0),
                    "New": int(run.get("new_jobs_count", 0) or 0),
                    "Already seen": int(run.get("duplicate_jobs_count", 0) or 0),
                    "Saved": len(run.get("new_jobs", []) or []),
                    "Full JDs": int(run.get("full_descriptions_count", 0) or 0),
                    "Skipped previews": int(run.get("skipped_jobs_count", 0) or 0),
                }
                for run in outcome.runs
            ],
            width="stretch",
            hide_index=True,
        )
        if summary.seen_jobs:
            st.markdown("**Already seen jobs**")
            services.render_fetch_run_job_table(
                summary.seen_jobs,
                "No already seen jobs in this search.",
            )
        if services.show_debug_ui and outcome.backend_outputs:
            st.markdown("**Developer fetch output**")
            st.text("\n\n".join(outcome.backend_outputs))


def render_fetch_results(
    outcome: FetchSearchOutcome,
    request: FetchSearchRequest,
    services: FetchPageServices,
) -> None:
    """Render all user-facing output for a completed search."""
    summary = summarize_fetch_outcome(outcome)
    render_fetch_status(outcome)
    render_fetch_metrics(outcome, summary)
    st.session_state["recommendation_limit"] = request.recommendation_limit
    render_fetch_job_results(outcome, summary, services)
    render_fetch_details(outcome, summary, services)


def render_fetch_debug(
    request: FetchSearchRequest,
    backend_outputs: list[str],
    services: FetchPageServices,
) -> None:
    """Render diagnostics only when explicitly enabled."""
    if not services.show_debug_ui:
        return
    with st.expander("Developer search details", expanded=False):
        st.markdown("**Source mapping**")
        st.write(f"Adzuna country: `{request.adzuna_country}`")
        st.write(f"Adzuna location: `{request.adzuna_location}`")
        st.write(f"Jooble location: `{request.jooble_location}`")
        st.markdown("**Rate-limit notes**")
        st.warning(
            "Do not repeatedly open many Adzuna links in a short time. "
            "If Adzuna shows 'Too Many Requests', wait 10-30 minutes and avoid refreshing."
        )
        if backend_outputs:
            st.markdown("**Internal fetch metadata**")
            st.text("\n\n".join(backend_outputs))
        st.markdown("**Raw fetch history**")
        services.render_fetch_history_section()


def fetch_jobs_tab(services: FetchPageServices) -> None:
    """Render the fetch-jobs workflow."""
    render_fetch_search_styles(st)
    services.render_page_header(
        "Find Jobs",
        "Search supported job sources and save roles for review.",
    )
    if services.demo_mode_enabled():
        st.caption("Live search is unavailable in the read-only Demo workspace.")
    st.caption(
        "Sources with complete descriptions are searched first; discovery sources may return previews."
    )
    request = render_fetch_search_form(services)
    backend_outputs: list[str] = []
    if request.submitted:
        if not fetch_request_is_valid(request, services):
            return
        outcome = execute_fetch_search(request, services)
        backend_outputs = outcome.backend_outputs
        render_fetch_results(outcome, request, services)
    render_fetch_debug(request, backend_outputs, services)
