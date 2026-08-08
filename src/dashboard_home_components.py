"""Presentation components for the Dashboard home page."""

from __future__ import annotations

from typing import Callable, Protocol

import streamlit as st

from dashboard_fit import eligibility_status
from dashboard_review import review_fit_band
from dashboard_review_chrome import saved_job_context
from dashboard_review_components import jd_quality_label, visible_role_fit
from dashboard_titles import get_job_display_title
from scoring_types import DashboardJob


class HomeNavigator(Protocol):
    """Navigation surface required by Dashboard components."""

    def go_to_page(self, page: str) -> None:
        """Navigate to another application page."""


def render_primary_action(
    jobs: list[DashboardJob],
    packages: int,
    services: HomeNavigator,
    select_action: Callable[[list[DashboardJob], int], tuple[str, str, str]],
) -> None:
    """Render the single most useful workflow action."""
    title, description, page = select_action(jobs, packages)
    action_label = {
        "Add Target Job": "Add target job",
        "Cover Letter": "Review cover letter",
        "Review Jobs": "Review jobs",
    }.get(page, "Continue")
    with st.container(border=True, key="home_next_action"):
        content, action = st.columns([0.75, 0.25])
        with content:
            st.caption("Next best action")
            st.markdown(f"### {title}")
            st.write(description)
        with action:
            if st.button(
                action_label,
                key=f"home_primary_action_{page}",
                type="primary",
                width="stretch",
            ):
                services.go_to_page(page)


def render_home_metrics(summary: dict[str, int]) -> None:
    """Render compact workspace counts."""
    st.markdown("**Workspace summary**")
    with st.container(border=True, key="home_summary"):
        columns = st.columns(4)
        columns[0].metric("Saved jobs", summary["saved"])
        columns[1].metric("Strong matches", summary["strong"])
        columns[2].metric("Needs full JD", summary["needs_full_jd"])
        columns[3].metric("Cover letters", summary["cover_letters"])


def render_recent_opportunities(
    jobs: list[DashboardJob],
    services: HomeNavigator,
) -> None:
    """Render three real saved opportunities with direct workflow actions."""
    heading, search_action = st.columns([0.78, 0.22], vertical_alignment="center")
    with heading:
        st.markdown("**Recent opportunities**")
    with search_action:
        if st.button(
            "Search fresh roles",
            key="home_find_jobs",
            icon=":material/search:",
            type="secondary",
            width="stretch",
        ):
            services.go_to_page("Find Jobs")
    ranked = sorted(
        jobs,
        key=lambda job: (
            eligibility_status(job) != "failed",
            {"Strong": 2, "Review": 1, "Weak": 0}.get(review_fit_band(job), 0),
            int(job.get("score") or -1),
        ),
        reverse=True,
    )[:3]
    if not ranked:
        st.caption("No saved jobs yet. Search fresh roles or add a target job to begin.")
        return
    with st.container(border=True, key="home_recent_jobs"):
        for index, job in enumerate(ranked):
            title, company, fit, quality, action = st.columns(
                [2.3, 1.4, 0.9, 1.2, 0.72],
                vertical_alignment="center",
            )
            with title:
                st.markdown(f"**{get_job_display_title(job)}**")
                st.caption(saved_job_context(job))
            with company:
                st.caption("Company")
                st.write(str(job.get("company", "Unknown company")))
            with fit:
                st.caption("Fit")
                st.write(visible_role_fit(job))
            with quality:
                st.caption("JD quality")
                st.write(jd_quality_label(job))
            with action:
                if st.button(
                    "Review",
                    key=f"home_review_job_{index}",
                    type="secondary",
                    width="stretch",
                ):
                    st.session_state["selected_review_job_path"] = str(job["path"])
                    st.session_state["selected_review_job_label"] = job["label"]
                    st.session_state["selected_review_tab"] = "Overview"
                    st.session_state["review_detail_tabs"] = "Decision"
                    services.go_to_page("Review Jobs")
            if index < len(ranked) - 1:
                st.divider()


def render_home_styles() -> None:
    """Keep home-specific layout CSS scoped to keyed containers."""
    st.markdown(
        """
        <style>
        .st-key-home_next_action {margin:.45rem 0 1rem}
        .st-key-home_next_action [data-testid="stVerticalBlockBorderWrapper"] {padding:.2rem .25rem}
        .st-key-home_next_action [data-testid="stColumn"]:last-child {
            display:flex;align-items:center;
        }
        .st-key-home_summary {margin:.35rem 0 1.05rem}
        .st-key-home_summary [data-testid="stMetricValue"] {font-size:1.55rem}
        .st-key-home_recent_jobs [data-testid="stMarkdownContainer"] p {overflow-wrap:anywhere}
        .st-key-home_recent_jobs hr {margin:.15rem 0}
        </style>
        """,
        unsafe_allow_html=True,
    )
