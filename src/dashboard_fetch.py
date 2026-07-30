"""Live job-discovery page for the local toolkit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from fetch_jobs import jsearch_configured
from dashboard_fetch_runner import FetchSearchOutcome, run_job_search
from dashboard_regions import source_display_name
from dashboard_search_profile import search_profile_from_path


DEFAULT_FETCH_LIMIT_PER_SOURCE = 20
MAX_FETCH_LIMIT_PER_SOURCE = 20
REGION_OPTIONS = [
    "Remote",
    "Singapore",
    "United Kingdom",
    "United States",
    "Canada",
    "Australia",
    "Custom",
]
ADZUNA_SUPPORTED_COUNTRIES = {
    "sg", "gb", "us", "ca", "au", "nz", "de", "fr", "it", "nl", "pl", "br", "za", "in"
}
REGION_CONFIG = {
    "Remote": {"adzuna_country": "us", "adzuna_location": "Remote", "jooble_location": "Remote"},
    "Singapore": {
        "adzuna_country": "sg",
        "adzuna_location": "Singapore",
        "jooble_location": "Singapore",
    },
    "United Kingdom": {
        "adzuna_country": "gb",
        "adzuna_location": "United Kingdom",
        "jooble_location": "United Kingdom",
    },
    "United States": {
        "adzuna_country": "us",
        "adzuna_location": "United States",
        "jooble_location": "United States",
    },
    "Canada": {"adzuna_country": "ca", "adzuna_location": "Canada", "jooble_location": "Canada"},
    "Australia": {
        "adzuna_country": "au",
        "adzuna_location": "Australia",
        "jooble_location": "Australia",
    },
    "Custom": {"adzuna_country": "us", "adzuna_location": "", "jooble_location": ""},
}


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
    query = st.text_input("Target role / query", key="fetch_query")
    if profile.source_ready:
        matched_keywords = ", ".join(keyword.title() for keyword in profile.keywords)
        detail = f" Resume signals: {matched_keywords}." if matched_keywords else ""
        st.caption(f"Suggested from the uploaded resume; you can edit it before searching.{detail}")
    else:
        st.caption(
            "Upload a resume to receive an automatic role suggestion, "
            "or enter a query manually."
        )
    return query


def render_fetch_region_fields(services: FetchPageServices) -> tuple[str, str, str, bool]:
    """Render region fields and return provider-specific locations."""
    region = st.selectbox("Region", REGION_OPTIONS, index=0, key="fetch_region")
    region_config = REGION_CONFIG[region]
    adzuna_country = region_config["adzuna_country"]
    adzuna_location = region_config["adzuna_location"]
    jooble_location = region_config["jooble_location"]
    if region == "Custom":
        location_text = st.text_input("Custom Location", key="fetch_custom_location")
        if services.show_debug_ui:
            adzuna_country = st.text_input(
                "Developer: Adzuna country",
                value=adzuna_country,
            )
        adzuna_location = location_text
        jooble_location = location_text
    return (
        adzuna_country,
        adzuna_location,
        jooble_location,
        adzuna_country.lower() in ADZUNA_SUPPORTED_COUNTRIES,
    )


def render_fetch_options_form(
    services: FetchPageServices,
    *,
    adzuna_supported: bool,
) -> tuple[bool, list[str], int, int]:
    """Render provider and result-limit controls."""
    with st.form("fetch_jobs_form"):
        recommendation_limit = st.slider(
            "Number of recommendations",
            min_value=services.min_recommendation_limit,
            max_value=services.max_recommendation_limit,
            value=services.default_recommendation_limit,
            help="How many ranked jobs to display after filtering and duplicate removal.",
        )
        full_jd_source_ready = jsearch_configured()
        sources = st.multiselect(
            "Sources",
            ["jsearch", "adzuna", "jooble"],
            default=["jsearch"] if full_jd_source_ready else ["adzuna", "jooble"],
            format_func=source_display_name,
        )
        if not full_jd_source_ready:
            st.info(
                "For automatic full job descriptions, add JSEARCH_API_KEY to `.env`. "
                "The existing Adzuna and Jooble keys can still discover jobs, "
                "but their official search responses contain snippets."
            )
        if "adzuna" in sources and not adzuna_supported:
            st.warning(
                "Adzuna is not available for this region. "
                "Jooble can still search this location."
            )
        fetch_limit_per_source = st.slider(
            "Jobs per source",
            min_value=5,
            max_value=MAX_FETCH_LIMIT_PER_SOURCE,
            value=DEFAULT_FETCH_LIMIT_PER_SOURCE,
            help="How many jobs to request from each source before filtering.",
        )
        submitted = st.form_submit_button("Find Jobs", type="primary", width="stretch")
    return submitted, sources, recommendation_limit, fetch_limit_per_source


def render_fetch_search_form(services: FetchPageServices) -> FetchSearchRequest:
    """Render search controls and return one immutable request."""
    query = initialize_fetch_query(services)
    (
        adzuna_country,
        adzuna_location,
        jooble_location,
        adzuna_supported,
    ) = render_fetch_region_fields(services)
    submitted, sources, recommendation_limit, fetch_limit_per_source = (
        render_fetch_options_form(
            services,
            adzuna_supported=adzuna_supported,
        )
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
    result_metrics[3].metric("Saved locally", len(outcome.saved_paths))
    result_metrics[4].metric("Issues", summary.issues)
    result_metrics[5].metric("Full JDs", summary.full_descriptions)


def render_fetch_job_results(
    outcome: FetchSearchOutcome,
    summary: FetchResultSummary,
    services: FetchPageServices,
) -> None:
    """Render new jobs or the no-new-results recovery actions."""
    if summary.new == 0 and outcome.runs:
        st.info(
            "No new jobs found.\n\n"
            "All returned jobs were already seen in previous searches.\n\n"
            "Try broadening the query, increasing jobs per source, changing region, "
            "or reviewing saved jobs."
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
                    "Errors": int(run.get("skipped_jobs_count", 0) or 0),
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
    services.render_page_header(
        "Find Jobs",
        "Search supported job sources and save roles for review.",
    )
    if services.demo_mode_enabled():
        st.caption("Live search is unavailable in the read-only Demo workspace.")
    st.caption(
        "JSearch returns full job descriptions for reliable scoring. "
        "Adzuna and Jooble remain optional discovery sources and may return summaries."
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
