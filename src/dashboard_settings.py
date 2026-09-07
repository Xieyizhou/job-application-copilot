"""Settings and Personal workspace setup pages for the dashboard."""

from __future__ import annotations

from dashboard_ui import render_page_header

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_candidate_profile import render_candidate_profile_editor
from dashboard_demo_resume import render_demo_resume
from dashboard_settings_sections import render_settings_sections
from scoring_types import TrackerRow
from workspace import (
    SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS,
    SUPPORTED_EXPERIENCE_BANK_EXTENSIONS,
    SUPPORTED_RESUME_EXTENSIONS,
    Workspace,
    WorkspaceError,
    initialize_personal_workspace,
)


@dataclass(frozen=True)
class SettingsPageServices:
    """Shared dashboard operations required by Settings and workspace setup."""

    current_workspace: Callable[[], Workspace]
    demo_mode_enabled: Callable[[], bool]
    list_job_description_files: Callable[..., list[Any]]
    load_tracker_rows: Callable[..., list[TrackerRow]]


def safety_notes_tab(services: SettingsPageServices) -> None:
    """Render compact Settings health with topic-specific details."""
    render_page_header(
        "Settings",
        "Check workspace health, job sources, scoring definitions, and privacy boundaries.",
    )
    workspace = services.current_workspace()
    jobs_count = len(services.list_job_description_files()) if workspace.ready else 0
    tracker_count = (
        0
        if services.demo_mode_enabled()
        else len(services.load_tracker_rows(sort_by="created_at", descending=True))
    )
    render_settings_sections(
        st,
        workspace=workspace,
        jobs_count=jobs_count,
        tracker_count=tracker_count,
        demo_mode=services.demo_mode_enabled(),
    )


def render_candidate_workspace_setup(workspace: Workspace) -> None:
    """Collect candidate files before enabling Personal workflows."""
    if workspace.mode == "demo":
        render_page_header("Resume", "Review the example resume setup used for local fit analysis and cover letters.")
        render_demo_resume(workspace)
        return
    render_page_header(
        "Resume",
        (
            "Review or replace the resume used for local fit analysis and cover letters."
            if workspace.ready
            else "Upload a resume before searching and scoring jobs."
        ),
    )
    if workspace.ready:
        st.success("Resume ready. Upload another file only when you want to replace it.")
        render_candidate_profile_editor(st, workspace)
        st.divider()
    else:
        st.info("Your resume is the factual source for every fit result and cover letter.")
    resume_upload = st.file_uploader(
        "Resume file",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_RESUME_EXTENSIONS)],
        help="Files are parsed locally and stored as canonical Markdown. Text-based PDFs only; no OCR.",
        key="workspace_resume_upload",
    )
    with st.expander("Optional supporting files", expanded=False):
        experience_upload = st.file_uploader(
            "Experience bank",
            type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXPERIENCE_BANK_EXTENSIONS)],
            key="workspace_experience_upload",
        )
        template_upload = st.file_uploader(
            "Cover-letter template",
            type=[extension.lstrip(".") for extension in sorted(SUPPORTED_COVER_LETTER_TEMPLATE_EXTENSIONS)],
            key="workspace_template_upload",
        )
    if workspace.ready:
        st.caption("Saving replaces the current resume and any optional file selected here.")
    if not st.button("Save resume", type="primary", disabled=resume_upload is None):
        return

    try:
        assert resume_upload is not None
        updated = initialize_personal_workspace(
            resume_filename=resume_upload.name,
            resume_content=resume_upload.getvalue(),
            experience_bank=(experience_upload.name, experience_upload.getvalue()) if experience_upload else None,
            cover_letter_template=(template_upload.name, template_upload.getvalue()) if template_upload else None,
        )
        if not updated.ready:
            raise WorkspaceError("The Personal workspace could not be validated after setup.")
        st.session_state["workspace_setup_open"] = False
        format_label = (updated.candidate_original_extension or "source").lstrip(".").upper()
        extraction_label = (updated.candidate_extraction_method or "local extraction").replace("_", " ")
        details = f"Accepted {format_label}; extracted locally with {extraction_label}."
        if updated.candidate_pdf_page_count is not None:
            details += f" PDF pages: {updated.candidate_pdf_page_count}."
        st.success("Personal workspace configured. Candidate files remain local and ignored by Git.")
        st.caption(details)
        st.rerun()
    except WorkspaceError as error:
        st.error(str(error))
