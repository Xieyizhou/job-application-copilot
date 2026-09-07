"""Review Jobs action for one-click public ATS description completion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

from ats_jd_completion import ats_completion_progress
from scoring_types import DashboardJob


def render_ats_completion_action(
    all_jobs: list[DashboardJob],
    services: Any,
) -> None:
    """Offer one safe batch action for incomplete jobs on supported public ATSs."""
    paths = [Path(job["path"]) for job in all_jobs]
    candidates = services.public_ats_completion_candidates(
        paths, include_resolvable=True
    )
    progress = ats_completion_progress(paths)
    previous = st.session_state.pop("ats_completion_summary", None)
    retry_available = bool(progress["failed"])
    if isinstance(previous, dict):
        updated = int(previous.get("updated", 0))
        failed = int(previous.get("failed", 0))
        eligible = int(previous.get("eligible", 0))
        attempted = int(previous.get("attempted", 0))
        deferred = int(previous.get("deferred", 0))
        retry_available = retry_available or bool(previous.get("failures"))
        if updated:
            st.success(
                f"Completed {updated} job description{'s' if updated != 1 else ''}."
            )
        elif failed:
            st.warning(
                "Employer postings were found, but none passed every identity and quality check."
            )
        elif not eligible:
            st.info(
                "No safe full-posting matches were found for the incomplete saved jobs."
            )
        if failed:
            with st.expander(f"Review {failed} job{'s' if failed != 1 else ''} not updated"):
                for failure in previous.get("failures", []):
                    if isinstance(failure, dict):
                        st.caption(str(failure.get("message", "The JD was not updated.")))
        if attempted:
            st.caption(
                f"Checked {attempted} job{'s' if attempted != 1 else ''}. "
                f"{deferred} remain for a later batch."
            )
    # This used to be a wide popover. Inside the narrow Saved Jobs column the
    # floating layer could cover the sidebar and most of the review surface.
    # An inline expander keeps the action in context and never obscures the job.
    with st.expander(
        "Recover full job descriptions",
        icon=":material/auto_fix_high:",
        expanded=False,
    ):
        st.caption(
            "Check supported employer-hosted links. Preview-only jobs remain unchanged "
            "unless an exact public posting passes identity and completeness checks."
        )
        if candidates:
            st.caption(
                f"{progress['incomplete']} incomplete · {progress['pending']} not yet checked · "
                f"{progress['failed']} retryable. Each batch processes at most 10."
            )
        run_column, retry_column = st.columns(2, gap="small")
        with run_column:
            run_clicked = st.button(
                "Run safe batch",
                key="review_complete_public_ats_jds",
                type="primary",
                width="stretch",
                disabled=not progress["pending"] or services.demo_mode_enabled(),
            )
        with retry_column:
            retry_clicked = st.button(
                "Retry failed",
                key="review_retry_public_ats_jds",
                width="stretch",
                disabled=not retry_available or services.demo_mode_enabled(),
            )
        if run_clicked or retry_clicked:
            with st.spinner("Checking original employer postings…"):
                summary = services.complete_public_ats_jobs(
                    paths,
                    resolve=services.enrich_saved_job_description,
                    max_jobs=10,
                    retry_failed=retry_clicked,
                )
            st.session_state["ats_completion_summary"] = summary
            st.rerun()
