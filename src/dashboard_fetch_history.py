"""Fetch-history presentation shared by dashboard discovery surfaces."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import streamlit as st

from dashboard_regions import normalize_location, source_display_name
from dashboard_titles import get_job_display_title
from fetch_history import load_fetch_runs


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TextReader = Callable[[Path], str]


def read_text_file(path: Path) -> str:
    """Read a saved Markdown preview without raising for missing files."""
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return ""


def fetch_run_label(run: dict[str, Any]) -> str:
    """Build a readable fetch-run selector label."""
    return (
        f"{run.get('created_at', '')} | "
        f"{source_display_name(str(run.get('source', '')))} | "
        f"{run.get('region', '-') or '-'} | {run.get('query', '-') or '-'} | "
        f"{run.get('new_jobs_count', 0)} new"
    )


def fetch_run_job_rows(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map stored job summaries to compact history-table rows."""
    return [
        {
            "Company": job.get("company", ""),
            "Role": get_job_display_title(job),
            "Location": normalize_location(str(job.get("location", ""))),
            "Source": source_display_name(str(job.get("source", ""))),
            "Saved": "Yes" if job.get("path") else "No",
        }
        for job in jobs
    ]


def fetch_history_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Map fetch-run records to the history summary table."""
    return [
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
    ]


def render_fetch_run_job_table(
    jobs: list[dict[str, Any]],
    empty_message: str,
) -> None:
    """Render compact job summaries stored on a fetch run."""
    if not jobs:
        st.info(empty_message)
        return
    st.dataframe(
        fetch_run_job_rows(jobs),
        width="stretch",
        hide_index=True,
    )


def render_fetch_run_job_cards(
    jobs: list[dict[str, Any]],
    empty_message: str,
    *,
    read_text: TextReader = read_text_file,
) -> None:
    """Render fetched jobs as simple cards without changing fetch behavior."""
    if not jobs:
        st.info(empty_message)
        return

    for job in jobs:
        with st.container(border=True):
            company = str(job.get("company", "") or "Unknown company")
            role = get_job_display_title(job, fallback="Unknown role")
            location = normalize_location(str(job.get("location", ""))) or "-"
            source = source_display_name(str(job.get("source", "")))
            st.markdown(f"**{company}**")
            st.write(role)
            st.caption(f"{location} | {source}")
            st.caption(
                "Saved locally · Fit and evidence quality are calculated in Review Jobs"
            )
            if job.get("job_url"):
                st.link_button("Open original posting", str(job["job_url"]))
            if job.get("path"):
                with st.expander("View Details", expanded=False):
                    path = PROJECT_ROOT / str(job.get("path", ""))
                    if path.exists():
                        st.markdown(read_text(path)[:1200])
                    else:
                        st.write(f"Saved path: `{job.get('path', '')}`")


def render_fetch_run_details(run: dict[str, Any]) -> None:
    """Show new jobs first, with repeated jobs collapsed by default."""
    if not run:
        return
    st.write(
        f"{source_display_name(str(run.get('source', '')))} / "
        f"{run.get('region', '-') or '-'} / {run.get('query', '-') or '-'}"
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
    render_fetch_run_job_cards(
        run.get("new_jobs", []) or [],
        "No new jobs were discovered in this search.",
    )
    with st.expander("Compact table view", expanded=False):
        render_fetch_run_job_table(
            run.get("new_jobs", []) or [],
            "No new jobs were discovered in this search.",
        )
    with st.expander("Advanced: already seen jobs from this search", expanded=False):
        render_fetch_run_job_cards(
            run.get("previously_seen_jobs", []) or [],
            "No already seen jobs were returned.",
        )


def render_fetch_history_section() -> None:
    """Render recent fetch-run history and a past-run review selector."""
    runs = load_fetch_runs(limit=20)
    st.markdown("**Fetch History**")
    if not runs:
        st.info("No fetch history yet.")
        return
    st.dataframe(
        fetch_history_rows(runs),
        width="stretch",
        hide_index=True,
    )
    labels = [fetch_run_label(run) for run in runs]
    selected_label = st.selectbox(
        "Review search results",
        labels,
        key="fetch_history_selected",
    )
    render_fetch_run_details(runs[labels.index(selected_label)])
