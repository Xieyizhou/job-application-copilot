"""Compact dropdown selection for the Review Jobs workflow."""

from __future__ import annotations

from typing import Any, Callable, Protocol

import streamlit as st

from dashboard_fit import build_fit_presentation
from dashboard_regions import source_display_name
from dashboard_titles import get_job_display_title


class ReviewSelectorServices(Protocol):
    package_status_for_job: Callable[..., str]
    tracker_status_for_job: Callable[..., str]


def job_picker_label(job: dict[str, Any]) -> str:
    """Return a compact identity and decision label for one dropdown option."""
    company = str(job.get("company", "Unknown company"))
    title = get_job_display_title(job)
    decision = str(job.get("recommendation", "Manual Review"))
    return f"{company} · {title} — {decision}"


def picked_review_job(jobs: list[dict[str, Any]], selected_path: str) -> dict[str, Any]:
    """Resolve a dropdown path without relying on potentially duplicate labels."""
    return next((job for job in jobs if str(job["path"]) == selected_path), jobs[0])


def render_review_job_picker(
    jobs: list[dict[str, Any]],
    tracker_rows: list[dict[str, Any]],
    current_job: dict[str, Any],
    summary_parts: list[str],
    services: ReviewSelectorServices,
    on_select: Callable[[dict[str, Any], str], None],
) -> dict[str, Any]:
    """Render one dropdown and return the job whose detail should be shown."""
    st.caption(" · ".join(summary_parts))
    paths = [str(job["path"]) for job in jobs]
    labels = {str(job["path"]): job_picker_label(job) for job in jobs}
    current_path = str(current_job["path"])
    picker_key = "review_job_picker_path"
    if st.session_state.get(picker_key) not in paths:
        st.session_state[picker_key] = current_path
    selected_path = st.selectbox(
        "Choose a job",
        paths,
        key=picker_key,
        format_func=lambda path: labels[path],
    )
    selected_job = picked_review_job(jobs, selected_path)
    if selected_path != st.session_state.get("selected_review_job_path"):
        on_select(selected_job, str(st.session_state.get("selected_review_tab", "Overview")))

    with st.expander(f"Compare {len(jobs)} jobs", expanded=False):
        st.dataframe(
            [
                {
                    "Company": job["company"],
                    "Role": get_job_display_title(job),
                    "Location": job["normalized_location"],
                    "Source": source_display_name(str(job["source"])),
                    "Role Fit": build_fit_presentation(job)["role_fit"],
                    "Recommendation": job["recommendation"],
                    "Tracker": job.get("tracker_status")
                    or services.tracker_status_for_job(job, tracker_rows),
                    "Cover Letter": job.get("package_status")
                    or services.package_status_for_job(job, tracker_rows),
                }
                for job in jobs
            ],
            width="stretch",
            hide_index=True,
        )
    return selected_job
