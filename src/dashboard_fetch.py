"""Live job-discovery page for the local toolkit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import streamlit as st

from fetch_jobs import fetch_and_save_jobs, jsearch_configured
from dashboard_regions import source_display_name


DEFAULT_FETCH_LIMIT_PER_SOURCE = 20
MAX_FETCH_LIMIT_PER_SOURCE = 20
REGION_OPTIONS = ["Remote", "United States", "Canada", "Australia", "Custom"]
ADZUNA_SUPPORTED_COUNTRIES = {
    "sg", "gb", "us", "ca", "au", "nz", "de", "fr", "it", "nl", "pl", "br", "za", "in"
}
REGION_CONFIG = {
    "Remote": {"adzuna_country": "us", "adzuna_location": "Remote", "jooble_location": "Remote"},
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


def fetch_jobs_tab(services: FetchPageServices) -> None:
    """Render the fetch-jobs workflow."""
    services.render_page_header(
        "Find Jobs",
        "Search supported job sources and save roles for review.",
    )
    if services.demo_mode_enabled():
        st.info(
            "Demo workspace is active. Live job fetching is disabled; "
            "use Review Jobs to explore sample cards."
        )
    st.caption(
        "JSearch returns full job descriptions for reliable scoring. "
        "Adzuna and Jooble remain optional discovery sources and may return summaries."
    )
    backend_outputs: list[str] = []

    query = st.text_input("Target role / query", value="data analyst")
    region = st.selectbox("Region", REGION_OPTIONS, index=0, key="fetch_region")
    region_config = REGION_CONFIG[region]
    adzuna_country = region_config["adzuna_country"]
    adzuna_location = region_config["adzuna_location"]
    jooble_location = region_config["jooble_location"]

    if region == "Custom":
        location_text = st.text_input("Custom Location", key="fetch_custom_location")
        if services.show_debug_ui:
            adzuna_country = st.text_input("Developer: Adzuna country", value=adzuna_country)
        adzuna_location = location_text
        jooble_location = location_text

    with st.form("fetch_jobs_form"):
        recommendation_limit = st.slider(
            "Number of recommendations",
            min_value=services.min_recommendation_limit,
            max_value=services.max_recommendation_limit,
            value=services.default_recommendation_limit,
            help="How many ranked jobs to display after filtering and duplicate removal.",
        )
        full_jd_source_ready = jsearch_configured()
        default_sources = ["jsearch"] if full_jd_source_ready else ["adzuna", "jooble"]
        sources = st.multiselect(
            "Sources",
            ["jsearch", "adzuna", "jooble"],
            default=default_sources,
            format_func=source_display_name,
        )
        if not full_jd_source_ready:
            st.info(
                "For automatic full job descriptions, add JSEARCH_API_KEY to `.env`. "
                "The existing Adzuna and Jooble keys can still discover jobs, "
                "but their official search responses contain snippets."
            )
        adzuna_is_supported = adzuna_country.lower() in ADZUNA_SUPPORTED_COUNTRIES
        if "adzuna" in sources and not adzuna_is_supported:
            st.warning("Adzuna is not available for this region. Jooble can still search this location.")
        fetch_limit_per_source = st.slider(
            "Jobs per source",
            min_value=5,
            max_value=MAX_FETCH_LIMIT_PER_SOURCE,
            value=DEFAULT_FETCH_LIMIT_PER_SOURCE,
            help="How many jobs to request from each source before filtering.",
        )
        submitted = st.form_submit_button("Find Jobs")

    if submitted:
        if services.demo_mode_enabled():
            st.info(
                "Demo workspace does not call external job APIs. "
                "Select Personal and add API keys in `.env` for live fetch."
            )
            return
        if not sources:
            st.error("Select at least one source.")
            return

        all_saved_paths: list[Any] = []
        fetch_results: list[dict[str, Any]] = []
        fetch_errors: list[str] = []

        for source in sources:
            if source == "adzuna" and not adzuna_is_supported:
                backend_outputs.append(
                    f"[adzuna] Skipped because country `{adzuna_country}` is not supported."
                )
                continue

            source_location = adzuna_location if source in {"adzuna", "jsearch"} else jooble_location
            args = SimpleNamespace(
                source=source,
                country=adzuna_country,
                query=query,
                location=source_location,
                max_results=fetch_limit_per_source,
            )
            try:
                result, output = services.run_with_captured_output(fetch_and_save_jobs, args)
                saved_paths = list(result.get("saved_paths", []) if isinstance(result, dict) else [])
                saved_paths = services.relocate_fetched_jobs_to_workspace(saved_paths, source)
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
        total_full_descriptions = sum(
            int(run.get("full_descriptions_count", 0) or 0) for run in fetch_results
        )
        total_errors = len(fetch_errors) + total_skipped
        all_new_jobs = [job for run in fetch_results for job in (run.get("new_jobs", []) or [])]
        all_seen_jobs = [
            job for run in fetch_results for job in (run.get("previously_seen_jobs", []) or [])
        ]
        if fetch_results and fetch_errors:
            st.warning("Search completed for some sources. One or more sources could not be searched.")
        elif fetch_results:
            st.success("Search complete.")
        else:
            st.error("Search failed. Check API keys or use Demo workspace.")
        if any(
            ".env" in error or "API_KEY" in error or "APP_ID" in error or "APP_KEY" in error
            for error in fetch_errors
        ):
            st.info("Live job search requires API keys. You can use Demo workspace or add keys to `.env`.")
        result_metrics = st.columns(6)
        result_metrics[0].metric("Returned", total_returned)
        result_metrics[1].metric("New", total_new)
        result_metrics[2].metric("Already seen", total_seen)
        result_metrics[3].metric("Saved locally", len(all_saved_paths))
        result_metrics[4].metric("Issues", total_errors)
        result_metrics[5].metric("Full JDs", total_full_descriptions)
        st.session_state["recommendation_limit"] = recommendation_limit
        if total_new == 0 and fetch_results:
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
        elif all_new_jobs:
            for run in fetch_results:
                new_jobs = run.get("new_jobs", []) or []
                if not new_jobs:
                    continue
                st.markdown(f"**New jobs from {source_display_name(str(run.get('source', '')))}**")
                services.render_fetch_run_job_cards(new_jobs, "No new jobs in this search.")
            with st.expander("Compact table view", expanded=False):
                services.render_fetch_run_job_table(all_new_jobs, "No new jobs in this search.")

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
                            "Full JDs": int(run.get("full_descriptions_count", 0) or 0),
                            "Errors": int(run.get("skipped_jobs_count", 0) or 0),
                        }
                        for run in fetch_results
                    ],
                    width="stretch",
                    hide_index=True,
                )
                if all_seen_jobs:
                    st.markdown("**Already seen jobs**")
                    services.render_fetch_run_job_table(
                        all_seen_jobs,
                        "No already seen jobs in this search.",
                    )
                if services.show_debug_ui and backend_outputs:
                    st.markdown("**Developer fetch output**")
                    st.text("\n\n".join(backend_outputs))

    if services.show_debug_ui:
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
            services.render_fetch_history_section()
