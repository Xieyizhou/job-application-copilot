"""Reference-aligned chrome for the Review Jobs saved-job workspace."""

from __future__ import annotations

import html
from datetime import datetime
from typing import Any, Callable

import streamlit as st

from dashboard_titles import get_job_display_title
from scoring_types import DashboardJob


def saved_job_context(job: DashboardJob | dict[str, Any]) -> str:
    """Return enough truthful context to distinguish visually similar jobs."""
    location = str(job.get("normalized_location") or job.get("location") or "").strip()
    saved_date = ""
    for key in ("last_seen_at", "first_seen_at"):
        value = str(job.get(key, "") or "").strip()
        if not value or value.lower() == "unknown":
            continue
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            saved_date = parsed.strftime("%b %-d, %Y")
        except ValueError:
            saved_date = value[:20]
        break
    return " · ".join(part for part in (location, saved_date) if part) or "Saved job"


def render_saved_jobs_header(go_to_page: Callable[[str], None]) -> None:
    """Render the Saved jobs rail title and its real add-job action."""
    title, action = st.columns([0.82, 0.18], vertical_alignment="center")
    with title:
        st.markdown(
            '<div class="saved-jobs-title">'
            '<span class="saved-jobs-desktop-title">Saved jobs</span>'
            '<span class="saved-jobs-mobile-title">Review Jobs</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with action:
        if st.button(
            "Add target job",
            key="review_add_target_job",
            icon=":material/add:",
            type="tertiary",
            help="Add a target job",
            width="stretch",
        ):
            go_to_page("Add Target Job")


def render_selected_job_context(job: DashboardJob) -> None:
    """Render role, employer, context, and the original-posting action."""
    identity, action = st.columns([0.78, 0.22], vertical_alignment="center")
    role = html.escape(get_job_display_title(job))
    company = html.escape(str(job.get("company", "Unknown company")))
    context = html.escape(saved_job_context(job))
    with identity:
        st.markdown(
            '<div class="selected-job-context">'
            '<span class="selected-job-context-icon">work</span>'
            f'<span class="selected-job-context-role">{role}</span>'
            '<span class="selected-job-context-separator">•</span>'
            f'<span class="selected-job-context-meta">{company}</span>'
            '<span class="selected-job-context-separator">•</span>'
            f'<span class="selected-job-context-meta">{context}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
    with action:
        if job.get("job_url"):
            st.link_button(
                "Open job",
                str(job["job_url"]),
                icon=":material/open_in_new:",
                type="secondary",
                width="stretch",
            )
