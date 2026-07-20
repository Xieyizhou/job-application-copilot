"""Home dashboard page with portfolio and next-action summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_fit import build_fit_presentation, eligibility_status
from dashboard_regions import source_display_name
from dashboard_review import (
    job_evidence_label,
    job_needs_full_jd,
    sorted_review_jobs,
    tracker_follow_up_due,
)
from dashboard_titles import get_job_display_title


@dataclass(frozen=True)
class HomePageServices:
    """Shared dashboard operations required by the home page."""

    count_generated_packages: Callable[[], int]
    demo_mode_enabled: Callable[[], bool]
    load_screened_jobs: Callable[..., list[dict[str, Any]]]
    load_tracker_rows: Callable[..., list[dict[str, Any]]]
    render_page_header: Callable[[str, str | None], None]


def dashboard_tab(services: HomePageServices) -> None:
    """Render the default customer-facing home page."""
    services.render_page_header(
        "Dashboard",
        "Your local-first application workspace. Find and save jobs, generate drafts, then review before applying.",
    )
    if services.demo_mode_enabled():
        st.info("Demo workspace is using sanitized sample jobs and a sample cover letter. Live fetch and tracker writes are disabled.")

    jobs = services.load_screened_jobs()
    tracker_rows = (
        []
        if services.demo_mode_enabled()
        else services.load_tracker_rows(sort_by="created_at", descending=True)
    )
    analyzed_jobs = [job for job in jobs if job.get("analysis_available")]
    packages = services.count_generated_packages()
    cover_letter_count_label = "cover letter" if packages == 1 else "cover letters"
    applied_rows = [row for row in tracker_rows if str(row.get("status", "")).lower() in {"applied", "interview", "rejected"}]
    interview_rows = [row for row in tracker_rows if str(row.get("status", "")).lower() == "interview"]
    follow_up_rows = [row for row in tracker_rows if tracker_follow_up_due(row)]
    low_evidence_jobs = [job for job in jobs if job_needs_full_jd(job)]

    metric_cols = st.columns(5)
    metric_cols[0].metric("Saved opportunities", len(jobs))
    metric_cols[1].metric("Need full JD", len(low_evidence_jobs))
    metric_cols[2].metric("Cover letters", packages)
    metric_cols[3].metric("Applications sent", len(applied_rows))
    metric_cols[4].metric("Interviews", len(interview_rows))

    st.markdown("**Your next actions**")
    action_cols = st.columns(3)
    with action_cols[0]:
        with st.container(border=True):
            st.markdown(f"**{len(low_evidence_jobs)} jobs need better evidence**")
            st.caption("Use a full-JD source or add the original posting before trusting fit.")
    with action_cols[1]:
        with st.container(border=True):
            st.markdown(f"**{packages} {cover_letter_count_label} ready to review**")
            st.caption("Check the evidence trace, claims, formatting, and employer details before applying manually.")
    with action_cols[2]:
        with st.container(border=True):
            st.markdown(f"**{len(follow_up_rows)} follow-ups due**")
            st.caption("Applied roles with no stage update for at least seven days appear here.")

    candidate_jobs = [job for job in analyzed_jobs if eligibility_status(job) != "failed"]
    priority_jobs = sorted_review_jobs(candidate_jobs, "Role Fit high to low")[:3]
    st.markdown("**Top opportunities to review**")
    if not priority_jobs:
        st.info("Find or add jobs to build a ranked shortlist.")
    for job in priority_jobs:
        with st.container(border=True):
            presentation = build_fit_presentation(job)
            top_left, top_right = st.columns([0.72, 0.28])
            with top_left:
                st.markdown(f"**{job['company']} · {get_job_display_title(job)}**")
                st.caption(f"{job['normalized_location']} · {source_display_name(str(job['source']))} · {job_evidence_label(job)}")
            with top_right:
                st.write(presentation["role_fit"])
                st.caption(job["recommendation"])
            st.caption(f"Why: {dict(job.get('analysis_result', {}) or {}).get('main_reason', presentation['card_status'])}")
