"""Selectable comparison table for the Review Jobs workflow."""

from __future__ import annotations

import html
from typing import Callable

import streamlit as st

from dashboard_fit import confidence_level, eligibility_status
from dashboard_review_chrome import saved_job_context
from dashboard_review_components import jd_quality_label, visible_role_fit
from dashboard_titles import get_job_display_title
from output_paths import safe_slug
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


def mobile_job_picker_key(current_path: str) -> str:
    """Give each selected job an isolated responsive widget state."""
    return f"review_mobile_job_choice_{safe_slug(current_path)}"


def mobile_job_picker_options(jobs: list[DashboardJob]) -> tuple[list[str], dict[str, str]]:
    """Return unique option values with readable, location-aware labels."""
    paths = [str(job["path"]) for job in jobs]
    labels = {
        str(job["path"]): " · ".join(
            [
                get_job_display_title(job),
                str(job.get("company", "Unknown company")),
                saved_job_context(job),
            ]
        )
        for job in jobs
    }
    return paths, labels


def render_review_job_table(
    jobs: list[DashboardJob],
    current_job: DashboardJob,
    *,
    demo: bool,
    on_select: Callable[[DashboardJob, str], None],
) -> DashboardJob:
    """Render a compact selectable job list and return its active job."""
    current_path = str(current_job["path"])
    paths, labels = mobile_job_picker_options(jobs)
    current_index = paths.index(current_path) if current_path in paths else 0
    with st.container(key="review_mobile_job_picker"):
        mobile_choice = st.selectbox(
            "Saved job",
            paths,
            index=current_index,
            key=mobile_job_picker_key(current_path),
            format_func=labels.__getitem__,
            label_visibility="collapsed",
        )
    mobile_index = paths.index(mobile_choice)
    if mobile_index != current_index:
        on_select(jobs[mobile_index], str(st.session_state.get("selected_review_tab", "Overview")))
        st.rerun()
    with st.container(border=False, key="review_job_list"):
        for index, job in enumerate(jobs):
            selected = str(job["path"]) == current_path
            title = get_job_display_title(job)
            company = str(job.get("company", "Unknown company"))
            context = saved_job_context(job)
            fit = visible_role_fit(job)
            score = fit.removesuffix("/100") if fit.endswith("/100") else "—"
            try:
                score_value = int(score)
            except ValueError:
                score_value = 0
            score_tone = (
                "strong" if score_value >= 80 else "review" if score_value >= 60 else "weak"
            )
            with st.container(key=f"review_job_row_{index}"):
                if st.button(
                    f"{company} · {title}",
                    key=f"review_job_select_{index}",
                    type="secondary" if selected else "tertiary",
                    width="stretch",
                ):
                    on_select(job, str(st.session_state.get("selected_review_tab", "Overview")))
                    st.rerun()
                st.markdown(
                    '<div class="review-job-row-copy">'
                    '<div class="review-job-row-heading">'
                    f'<div class="review-job-row-title">{html.escape(title)}</div>'
                    f'<div class="review-job-row-score review-job-row-score-{score_tone}">{score}</div>'
                    "</div>"
                    f'<div class="review-job-row-company">{html.escape(company)}</div>'
                    f'<div class="review-job-row-context">{html.escape(context)}</div>'
                    "</div>",
                    unsafe_allow_html=True,
                )
    return current_job
