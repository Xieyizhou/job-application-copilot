"""Confirmed, recoverable saved-job management for Review Jobs."""

from __future__ import annotations

from pathlib import Path
from datetime import date, timedelta
from typing import Any

import streamlit as st

from dashboard_titles import get_job_display_title
from scoring_types import DashboardJob
from saved_job_deletion import old_saved_jobs, archive_job_paths


def _clear_review_selection() -> None:
    for key in ("selected_review_job_path", "selected_review_job_label"):
        st.session_state.pop(key, None)


def render_saved_job_delete_notice() -> None:
    """Render and consume the success message from a completed archive action."""
    count = st.session_state.pop("saved_job_delete_notice", None)
    if isinstance(count, int) and count > 0:
        st.success(f"Moved {count} saved job{'s' if count != 1 else ''} to local Trash.")
    failures = st.session_state.pop("saved_job_delete_failures", {})
    if failures:
        st.error(f"Could not move {len(failures)} jobs. They remain saved; refresh and retry.")


def render_saved_job_management(
    all_jobs: list[DashboardJob],
    selected_job: DashboardJob,
    services: Any,
) -> None:
    """Offer delete-one and delete-all actions with an explicit confirmation step."""
    mode = str(st.session_state.get("saved_job_delete_mode", ""))
    old_jobs, unknown_dates = old_saved_jobs(all_jobs)
    cutoff = date.today() - timedelta(days=30)
    with st.popover(
        "Manage saved jobs",
        icon=":material/delete_sweep:",
        width="stretch",
        disabled=services.demo_mode_enabled(),
    ):
        st.caption("Deleted jobs move to local Trash and can be recovered from disk.")
        if st.button(
            f"Delete jobs saved over 30 days ago ({len(old_jobs)})",
            key="request_delete_old_jobs",
            disabled=not old_jobs,
            width="stretch",
        ):
            st.session_state["saved_job_delete_mode"] = "old"
            st.session_state["saved_job_old_paths"] = [str(job["path"]) for job in old_jobs]
            st.rerun()
        if unknown_dates:
            st.caption(f"{unknown_dates} jobs have no valid first-save date and are excluded.")
        if st.button(
            "Delete selected job",
            key="request_delete_selected_job",
            icon=":material/delete:",
            width="stretch",
        ):
            st.session_state["saved_job_delete_mode"] = "selected"
            st.rerun()
        if st.button(
            f"Delete all {len(all_jobs)} jobs",
            key="request_delete_all_jobs",
            icon=":material/delete_forever:",
            width="stretch",
        ):
            st.session_state["saved_job_delete_mode"] = "all"
            st.rerun()

    if mode not in {"selected", "all", "old"}:
        return
    if services.demo_mode_enabled():
        return
    old_paths = set(st.session_state.get("saved_job_old_paths", []))
    targets = [job for job in old_jobs if str(job["path"]) in old_paths]
    if mode == "old":
        prompt = f"Delete {len(targets)} jobs first saved before {cutoff.isoformat()}?"
        detail = "Applies to all saved jobs, including those outside the current filters. Application history and cover letters are retained."
    elif mode == "selected":
        prompt = f"Delete {get_job_display_title(selected_job)}?"
        detail = "Only this saved job will move to local Trash."
    else:
        prompt = f"Delete all {len(all_jobs)} saved jobs?"
        detail = "Every saved job in this Personal workspace will move to one local Trash folder."
    with st.container(border=True, key="saved_job_delete_confirmation"):
        st.warning(prompt)
        st.caption(detail)
        if mode == "old":
            with st.expander("Jobs to move to Trash"):
                for job in targets:
                    st.write(f"{get_job_display_title(job)} · {job.get('first_seen_at') or job.get('created_at')}")
        confirm, cancel = st.columns(2)
        with confirm:
            if st.button(
                "Confirm delete",
                key="confirm_saved_job_delete",
                type="primary",
                width="stretch",
                disabled=mode == "old" and not targets,
            ):
                workspace = services.current_workspace()
                trash_dir = workspace.root / "trash" / "jobs"
                if mode == "old":
                    moved, failures = archive_job_paths(
                        [Path(job["path"]) for job in targets],
                        jobs_dir=workspace.jobs_dir,
                        trash_dir=trash_dir,
                    )
                    count = len(moved)
                    st.session_state["saved_job_delete_failures"] = failures
                elif mode == "selected":
                    services.archive_saved_job(
                        Path(selected_job["path"]),
                        jobs_dir=workspace.jobs_dir,
                        trash_dir=trash_dir,
                    )
                    count = 1
                else:
                    count = len(
                        services.archive_all_saved_jobs(
                            jobs_dir=workspace.jobs_dir,
                            trash_dir=trash_dir,
                        )
                    )
                st.session_state.pop("saved_job_delete_mode", None)
                st.session_state.pop("saved_job_old_paths", None)
                _clear_review_selection()
                st.session_state["saved_job_delete_notice"] = count
                st.rerun()
        with cancel:
            if st.button("Cancel", key="cancel_saved_job_delete", width="stretch"):
                st.session_state.pop("saved_job_delete_mode", None)
                st.session_state.pop("saved_job_old_paths", None)
                st.rerun()
