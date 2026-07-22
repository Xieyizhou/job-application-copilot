"""Settings and Personal workspace setup pages for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_settings_sections import render_settings_sections
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
    load_tracker_rows: Callable[..., list[dict[str, Any]]]
    render_page_header: Callable[[str, str | None], None]


def safety_notes_tab(services: SettingsPageServices) -> None:
    """Render compact Settings health with topic-specific details."""
    services.render_page_header(
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


def render_candidate_workspace_setup(workspace: Workspace, services: SettingsPageServices) -> None:
    """Collect candidate files before enabling Personal workflows."""
    services.render_page_header(
        "Candidate Workspace Setup",
        "Add your candidate source to initialize the private local workspace.",
    )
    st.info("Your resume is the required factual source for fit analysis and cover letters.")
    resume_upload = st.file_uploader(
        "Candidate source",
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
        st.caption("Submitting replaces the candidate source and any optional file selected here.")
    if not st.button("Save Personal workspace", type="primary", disabled=resume_upload is None):
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
