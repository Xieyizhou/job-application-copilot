"""Settings and Personal workspace setup pages for the dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

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
    """Render workspace status, scoring guidance, and privacy boundaries."""
    services.render_page_header(
        "Settings",
        "Workspace health, scoring interpretation, privacy boundaries, and local data controls.",
    )
    workspace = services.current_workspace()
    jobs_count = len(services.list_job_description_files()) if workspace.ready else 0
    tracker_count = (
        0
        if services.demo_mode_enabled()
        else len(services.load_tracker_rows(sort_by="created_at", descending=True))
    )
    settings_metrics = st.columns(4)
    settings_metrics[0].metric("Workspace", workspace.mode.title())
    settings_metrics[1].metric("Candidate source", "Ready" if workspace.resume_source_path else "Missing")
    settings_metrics[2].metric("Saved jobs", jobs_count)
    settings_metrics[3].metric("Tracker records", tracker_count)

    with st.expander("Scoring & evidence", expanded=True):
        st.markdown(
            """
            - **Role Fit** is evidence-calibrated and used for ranking; it is not an interview probability.
            - **Observed coverage** measures only the requirements the parser recognized.
            - **Confidence** tells you whether the JD and candidate evidence are complete enough to trust the score.
            - **Eligibility** is a separate hard-constraint review and can override a high fit score.
            - JSearch is the preferred source for full job descriptions; Adzuna and Jooble search results remain provisional when they contain snippets.
            """
        )
    with st.expander("Privacy & Safety", expanded=True):
        st.markdown(
            """
            - Local-first workflow: saved jobs, tracker records, and generated cover-letter bundles stay on this machine.
            - No automatic submissions: the app does not submit applications or answer external forms for you.
            - Human-in-the-loop: manually review the uploaded resume, every cover letter, and every application answer before use.
            - Safe generation: cover-letter drafts may rephrase real experience from the uploaded resume but must not invent facts.
            """
        )
    with st.expander("Developer / Advanced notes", expanded=False):
        st.markdown(
            """
            - Manually confirm visa and work authorization questions before applying.
            - `.env` should never be committed to Git.
            - API keys should never be shared.
            - Internal debug UI is disabled by default for the public app.
            """
        )


def render_candidate_workspace_setup(workspace: Workspace, services: SettingsPageServices) -> None:
    """Collect candidate files before enabling Personal workflows."""
    services.render_page_header(
        "Candidate Workspace Setup",
        "Add your candidate source to initialize the private local workspace.",
    )
    setup_cols = st.columns(3)
    setup_cols[0].info("Required · Resume/CV used as the factual source for fit and cover letters")
    setup_cols[1].info("Optional · Experience bank adds verified examples not visible in the resume")
    setup_cols[2].info("Optional · Cover-letter template controls structure, not Role Fit")
    st.write("Upload a PDF, DOCX, Markdown, or TXT candidate source. Files are parsed and stored locally.")
    resume_upload = st.file_uploader(
        "Candidate source",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_RESUME_EXTENSIONS)],
        help="Files are parsed locally and stored as canonical Markdown. Text-based PDFs only; no OCR.",
        key="workspace_resume_upload",
    )
    experience_upload = st.file_uploader(
        "Experience bank (optional)",
        type=[extension.lstrip(".") for extension in sorted(SUPPORTED_EXPERIENCE_BANK_EXTENSIONS)],
        key="workspace_experience_upload",
    )
    template_upload = st.file_uploader(
        "Cover-letter template (optional)",
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
