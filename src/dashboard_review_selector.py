"""Selectable comparison table for the Review Jobs workflow."""

from __future__ import annotations

import html
from typing import Callable

import streamlit as st

from dashboard_fit import confidence_level, eligibility_status
from dashboard_review_components import jd_quality_label, visible_role_fit
from dashboard_titles import get_job_display_title
from scoring_types import DashboardJob


def review_table_row(job: DashboardJob) -> dict[str, str]:
    """Return the five user-facing fields shown for one selectable job."""
    company = str(job.get("company", "Unknown company"))
    title = get_job_display_title(job)
    return {
        "Job": f"{company} · {title}",
        "Role Fit": visible_role_fit(job),
        "Eligibility": eligibility_status(job).replace("_", " ").title(),
        "Confidence": confidence_level(job.get("confidence")).title(),
        "JD Quality": jd_quality_label(job),
    }


def compact_review_table_row(job: DashboardJob) -> dict[str, str]:
    """Return the desktop master-list row without merging signal semantics."""
    row = review_table_row(job)
    eligibility = {
        "Passed": "Pass",
        "Failed": "Fail",
        "Manual Review": "Review",
    }.get(row["Eligibility"], row["Eligibility"])
    role_fit = row["Role Fit"].removesuffix("/100")
    jd_quality = row["JD Quality"].removesuffix(" JD")
    return {
        "Job": row["Job"],
        "Signals": " · ".join(
            [
                role_fit,
                eligibility,
                row["Confidence"],
                jd_quality,
            ]
        ),
    }


def selected_review_table_job(
    jobs: list[DashboardJob],
    selected_rows: list[int],
    fallback_path: str,
) -> DashboardJob:
    """Resolve a selected row, then a stable path, then the first visible job."""
    if selected_rows and 0 <= selected_rows[0] < len(jobs):
        return jobs[selected_rows[0]]
    return next(
        (job for job in jobs if str(job["path"]) == fallback_path),
        jobs[0],
    )


def render_review_job_table(
    jobs: list[DashboardJob],
    current_job: DashboardJob,
    *,
    demo: bool,
    on_select: Callable[[DashboardJob, str], None],
) -> DashboardJob:
    """Render a compact selectable job list and return its active job."""
    current_path = str(current_job["path"])
    with st.container(
        height=285 if demo else 500,
        border=False,
        key="review_job_list",
    ):
        for index, job in enumerate(jobs):
            row = compact_review_table_row(job)
            selected = str(job["path"]) == current_path
            with st.container(key=f"review_job_row_{index}"):
                if st.button(
                    row["Job"],
                    key=f"review_job_select_{index}",
                    type="secondary" if selected else "tertiary",
                    width="stretch",
                ):
                    on_select(job, str(st.session_state.get("selected_review_tab", "Overview")))
                    st.rerun()
                st.markdown(
                    '<div class="review-job-row-copy">'
                    f'<div class="review-job-row-title">{html.escape(row["Job"])}</div>'
                    f'<div class="review-job-row-signals">{html.escape(row["Signals"])}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
    return current_job
