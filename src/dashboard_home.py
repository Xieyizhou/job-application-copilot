"""Home dashboard page with concise next-action summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_fit import confidence_level, eligibility_status
from dashboard_review import (
    job_needs_full_jd,
    review_job_next_action,
    sorted_review_jobs,
    tracker_follow_up_due,
)
from dashboard_titles import get_job_display_title
from scoring_types import DashboardJob


@dataclass(frozen=True)
class HomePageServices:
    """Shared dashboard operations required by the home page."""

    count_generated_packages: Callable[[], int]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    load_screened_jobs: Callable[..., list[DashboardJob]]
    load_tracker_rows: Callable[..., list[dict[str, Any]]]
    render_page_header: Callable[[str, str | None], None]


def home_summary(
    jobs: list[DashboardJob],
    tracker_rows: list[dict[str, Any]],
) -> dict[str, int]:
    """Return the three decision counts shown on Dashboard."""
    active_jobs = [job for job in jobs if eligibility_status(job) != "failed"]
    return {
        "active": len(active_jobs),
        "ready": sum(1 for row in tracker_rows if str(row.get("status", "")).lower() == "ready"),
        "follow_ups": sum(1 for row in tracker_rows if tracker_follow_up_due(row)),
    }


def dashboard_tab(services: HomePageServices) -> None:
    """Render next actions first, followed by three counters and a shortlist."""
    services.render_page_header(
        "Dashboard",
        "See what needs attention and move the strongest opportunities forward.",
    )
    if services.demo_mode_enabled():
        st.info("Demo workspace uses sanitized sample jobs and does not write to your tracker.")

    jobs = services.load_screened_jobs()
    tracker_rows = [] if services.demo_mode_enabled() else services.load_tracker_rows(
        sort_by="created_at", descending=True
    )
    packages = services.count_generated_packages()
    _render_next_actions(jobs, tracker_rows, packages, services)
    _render_home_metrics(home_summary(jobs, tracker_rows))
    _render_shortlist(jobs, services)


def _render_next_actions(
    jobs: list[DashboardJob],
    tracker_rows: list[dict[str, Any]],
    packages: int,
    services: HomePageServices,
) -> None:
    low_evidence = sum(1 for job in jobs if job_needs_full_jd(job))
    follow_ups = sum(1 for row in tracker_rows if tracker_follow_up_due(row))
    actions: list[tuple[str, str, str]] = []
    if follow_ups:
        actions.append((f"Follow up on {follow_ups} application(s)", "Record a response or send a follow-up.", "Tracker"))
    if packages:
        actions.append((f"Review {packages} cover letter(s)", "Verify claims and employer details before applying.", "Cover Letter"))
    if low_evidence:
        actions.append((f"Complete {low_evidence} job description(s)", "Add the original posting before trusting fit.", "Review Jobs"))
    if not actions:
        actions.append(("Add your first target job", "Paste a complete posting to begin a reliable fit review.", "Add Target Job"))

    st.markdown("**Next actions**")
    for index, (title, description, page) in enumerate(actions[:3]):
        with st.container(border=True):
            content, action = st.columns([0.78, 0.22])
            with content:
                st.markdown(f"**{title}**")
                st.caption(description)
            with action:
                if st.button("Open", key=f"home_action_{index}_{page}", width="stretch"):
                    services.go_to_page(page)


def _render_home_metrics(summary: dict[str, int]) -> None:
    columns = st.columns(3)
    columns[0].metric("Active opportunities", summary["active"])
    columns[1].metric("Ready to apply", summary["ready"])
    columns[2].metric("Follow-ups due", summary["follow_ups"])


def _render_shortlist(jobs: list[DashboardJob], services: HomePageServices) -> None:
    candidates = [job for job in jobs if job.get("analysis_available") and eligibility_status(job) != "failed"]
    priority_jobs = sorted_review_jobs(candidates, "Role Fit high to low")[:3]
    st.markdown("**Priority opportunities**")
    if not priority_jobs:
        st.info("Find or add jobs to build a reviewed shortlist.")
        return
    for index, job in enumerate(priority_jobs):
        with st.container(border=True):
            content, action = st.columns([0.8, 0.2])
            with content:
                confidence = confidence_level(job.get("confidence")).title()
                st.markdown(f"**{job['company']} · {get_job_display_title(dict(job))}**")
                st.caption(f"{job.get('recommendation', 'Manual Review')} · {confidence} confidence")
                st.write(review_job_next_action(job))
            with action:
                if st.button("Review", key=f"home_review_{index}", width="stretch"):
                    selected_path = str(job.get("path", "") or "")
                    if selected_path:
                        st.session_state["selected_review_job_path"] = selected_path
                    services.go_to_page("Review Jobs")
