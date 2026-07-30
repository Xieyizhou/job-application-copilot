"""Home dashboard page with concise next-action summaries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import streamlit as st

from dashboard_fit import eligibility_status
from dashboard_review import (
    job_needs_full_jd,
    tracker_follow_up_due,
)
from scoring_types import DashboardJob, TrackerRow


@dataclass(frozen=True)
class HomePageServices:
    """Shared dashboard operations required by the home page."""

    count_generated_packages: Callable[[], int]
    demo_mode_enabled: Callable[[], bool]
    go_to_page: Callable[[str], None]
    load_screened_jobs: Callable[..., list[DashboardJob]]
    load_tracker_rows: Callable[..., list[TrackerRow]]
    render_page_header: Callable[[str, str | None], None]


def home_summary(
    jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
) -> dict[str, int]:
    """Return the three decision counts shown on Dashboard."""
    active_jobs = [job for job in jobs if eligibility_status(job) != "failed"]
    return {
        "active": len(active_jobs),
        "ready": sum(1 for row in tracker_rows if str(row.get("status", "")).lower() == "ready"),
        "follow_ups": sum(1 for row in tracker_rows if tracker_follow_up_due(row)),
    }


def dashboard_tab(services: HomePageServices) -> None:
    """Render one clear next step with secondary status kept collapsed."""
    services.render_page_header(
        "Dashboard",
        "Continue from the single most useful next step.",
    )
    jobs = services.load_screened_jobs()
    tracker_rows = [] if services.demo_mode_enabled() else services.load_tracker_rows(
        sort_by="created_at", descending=True
    )
    packages = services.count_generated_packages()
    _render_primary_action(jobs, tracker_rows, packages, services)
    with st.expander("Workspace summary", expanded=False):
        _render_home_metrics(home_summary(jobs, tracker_rows))


def primary_home_action(
    jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
    packages: int,
) -> tuple[str, str, str]:
    """Select one deterministic next action for the Dashboard."""
    low_evidence = sum(1 for job in jobs if job_needs_full_jd(job))
    follow_ups = sum(1 for row in tracker_rows if tracker_follow_up_due(row))
    if follow_ups:
        return (
            f"Follow up on {follow_ups} application(s)",
            "Record a response or send a follow-up.",
            "Tracker",
        )
    if packages:
        return (
            f"Review {packages} cover letter(s)",
            "Verify evidence, gaps, and employer details before applying.",
            "Cover Letter",
        )
    if low_evidence:
        return (
            f"Complete {low_evidence} job description(s)",
            "Add the original posting before trusting fit.",
            "Review Jobs",
        )
    if jobs:
        return (
            "Review your strongest opportunities",
            "Compare fit, eligibility, confidence, and JD quality.",
            "Review Jobs",
        )
    return (
        "Add your first target job",
        "Paste a complete posting to begin a reliable fit review.",
        "Add Target Job",
    )


def _render_primary_action(
    jobs: list[DashboardJob],
    tracker_rows: list[TrackerRow],
    packages: int,
    services: HomePageServices,
) -> None:
    title, description, page = primary_home_action(jobs, tracker_rows, packages)
    st.markdown("**Your next step**")
    with st.container(border=True):
        content, action = st.columns([0.75, 0.25])
        with content:
            st.markdown(f"### {title}")
            st.write(description)
        with action:
            if st.button(
                "Continue",
                key=f"home_primary_action_{page}",
                type="primary",
                width="stretch",
            ):
                services.go_to_page(page)


def _render_home_metrics(summary: dict[str, int]) -> None:
    columns = st.columns(3)
    columns[0].metric("Active opportunities", summary["active"])
    columns[1].metric("Ready to apply", summary["ready"])
    columns[2].metric("Follow-ups due", summary["follow_ups"])
