"""Decision-focused Cover Letter page components."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard_cover_letter_evidence import render_evidence_and_gaps
from dashboard_cover_letter_materials import (
    cover_letter_editor_mode,
    render_secondary_materials,
)
from dashboard_packages import existing_package_files
from dashboard_titles import display_title_from_value
from output_paths import safe_slug
from tracker import update_status
from workspace import WorkspaceError


DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"


@dataclass(frozen=True)
class CoverLetterArtifacts:
    """Known files for one generated cover-letter bundle."""

    package_dir: Path
    markdown: Path
    docx: Path
    analysis: Path
    internal_notes: list[Path]


def cover_letter_artifacts(package_dir: Path) -> CoverLetterArtifacts:
    """Collect allowlisted cover-letter files without opening their contents."""
    return CoverLetterArtifacts(
        package_dir=package_dir,
        markdown=package_dir / "cover_letter.md",
        docx=package_dir / "cover_letter.docx",
        analysis=package_dir / "analysis.md",
        internal_notes=existing_package_files(package_dir, ["cover_letter_notes.md"]),
    )


def render_cover_letter_workspace(
    package_dir: Path,
    tracker_row: dict[str, Any] | None,
    services: Any,
) -> None:
    """Compatibility wrapper for the cover-letter context and document."""
    artifacts = cover_letter_artifacts(package_dir)
    package_key = safe_slug(services.relative_path(package_dir)) or "selected_package"
    render_cover_letter_context(artifacts, tracker_row, services)
    render_cover_letter_document(artifacts, tracker_row, package_key, services)


def render_cover_letter_context(
    artifacts: CoverLetterArtifacts,
    tracker_row: dict[str, Any] | None,
    services: Any,
) -> None:
    """Render the evidence context in the left side of the desktop workspace."""
    _render_identity(tracker_row)
    render_evidence_and_gaps(artifacts, services)
    _render_readiness_statement(artifacts)


def render_cover_letter_document(
    artifacts: CoverLetterArtifacts,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    """Render the editable draft and actions in the right-side document pane."""
    _render_draft(artifacts, tracker_row, package_key, services)
    render_secondary_materials(artifacts, tracker_row, package_key, services)


def _render_identity(tracker_row: dict[str, Any] | None) -> None:
    if not tracker_row:
        return
    st.markdown(f"**{tracker_row.get('company', '-')}**")
    st.write(display_title_from_value(tracker_row.get("role"), fallback="Sample Job"))
    st.caption(f"Application stage: {str(tracker_row.get('status', 'draft')).title()}")


def _render_readiness_statement(artifacts: CoverLetterArtifacts) -> None:
    evidence = "evidence trace available" if artifacts.analysis.exists() else "evidence trace missing"
    gaps = "review unresolved gaps" if artifacts.internal_notes else "no separate gap notes available"
    st.info(
        f"Readiness: {evidence}; {gaps}; verify employer, role, dates, contact details, and every claim before sending."
    )


def _render_draft(
    artifacts: CoverLetterArtifacts,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    st.markdown("**Cover letter draft**")
    if not artifacts.markdown.exists():
        st.warning("The cover letter draft is missing. Generate it from Review Jobs first.")
        return

    draft = services.read_text_file(artifacts.markdown)
    editor_mode = cover_letter_editor_mode(package_key)
    st.markdown(
        f'<div class="cover-letter-editor-state cover-letter-editor-{editor_mode}"></div>',
        unsafe_allow_html=True,
    )
    if services.demo_mode_enabled():
        with st.container(border=True, key="cover_letter_demo_draft"):
            st.markdown(draft)
    else:
        edited_draft = st.text_area(
            "Review and edit draft",
            value=draft,
            height=520,
            key=f"cover_letter_editor_{package_key}",
            help="Edits are saved only when you use Save Draft.",
        )
        _render_primary_actions(edited_draft, artifacts, tracker_row, package_key, services)

    if artifacts.docx.exists():
        st.download_button(
            "Download Cover Letter DOCX",
            data=artifacts.docx.read_bytes(),
            file_name=artifacts.docx.name,
            mime=DOCX_MIME,
            key=f"download_cover_letter_docx_{package_key}",
            type="primary",
            width="stretch",
        )
    elif not services.demo_mode_enabled():
        if st.button("Create Cover Letter DOCX", key=f"create_docx_{package_key}", type="primary", width="stretch"):
            generated_path, warnings = services.generate_cover_letter_docx_for_package(artifacts.package_dir)
            if generated_path:
                st.success("Cover Letter DOCX created.")
                st.rerun()
            for warning in warnings:
                st.warning(str(warning))


def _render_primary_actions(
    edited_draft: str,
    artifacts: CoverLetterArtifacts,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    save_column, applied_column = st.columns(2, gap="small")
    with save_column:
        save_clicked = st.button(
            "Save Draft",
            key=f"save_cover_letter_{package_key}",
            help="Saving refreshes the employer-facing DOCX from this draft.",
            width="stretch",
        )
    with applied_column:
        already_applied = str((tracker_row or {}).get("status", "")).lower() == "applied"
        applied_clicked = st.button(
            "Mark as Applied",
            key=f"mark_cover_letter_applied_{package_key}",
            disabled=tracker_row is None or already_applied,
            width="stretch",
        )
    if save_clicked:
        artifacts.markdown.write_text(edited_draft.rstrip() + "\n", encoding="utf-8")
        generated_path, warnings = services.generate_cover_letter_docx_for_package(artifacts.package_dir)
        if generated_path:
            st.success("Draft saved and DOCX refreshed.")
        else:
            st.warning("Draft saved, but the DOCX could not be refreshed.")
        if warnings:
            with st.expander("DOCX warnings", expanded=False):
                for warning in warnings:
                    st.write(f"- {warning}")
    if applied_clicked and tracker_row:
        try:
            database_path = services.current_workspace().tracker_database_path
            if database_path is None:
                raise WorkspaceError("Tracker is unavailable in Demo workspace.")
            services.run_with_captured_output(update_status, int(tracker_row["id"]), "applied", database_path)
            st.success("Marked as applied.")
            st.rerun()
        except Exception as error:  # noqa: BLE001
            st.error(str(error))
