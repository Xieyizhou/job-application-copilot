"""Decision-focused Cover Letter page components."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

from dashboard_packages import build_application_package_zip, existing_package_files, package_zip_filename
from dashboard_review import tracker_follow_up_due, tracker_next_action
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
    """Render the selected letter with its primary action first."""
    artifacts = cover_letter_artifacts(package_dir)
    package_key = safe_slug(services.relative_path(package_dir)) or "selected_package"
    _render_identity(tracker_row)
    _render_readiness_statement(artifacts)
    _render_draft(artifacts, package_key, services)
    _render_application_status(tracker_row, services)
    _render_secondary_materials(artifacts, tracker_row, package_key, services)


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


def _render_draft(artifacts: CoverLetterArtifacts, package_key: str, services: Any) -> None:
    st.markdown("**Cover letter draft**")
    if not artifacts.markdown.exists():
        st.warning("The cover letter draft is missing. Generate it from Review Jobs first.")
        return

    draft = services.read_text_file(artifacts.markdown)
    if services.demo_mode_enabled():
        with st.container(border=True):
            st.markdown(draft)
    else:
        edited_draft = st.text_area(
            "Review and edit draft",
            value=draft,
            height=520,
            key=f"cover_letter_editor_{package_key}",
            help="Edits are saved only when you use Save Draft.",
        )
        edit_left, edit_right = st.columns([0.35, 0.65])
        with edit_left:
            if st.button("Save Draft", key=f"save_cover_letter_{package_key}", width="stretch"):
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
        with edit_right:
            st.caption("Saving refreshes the employer-facing DOCX from this draft.")

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


def _render_application_status(tracker_row: dict[str, Any] | None, services: Any) -> None:
    if services.demo_mode_enabled():
        return
    action_left, action_right = st.columns(2)
    with action_left:
        if tracker_row and st.button("Mark as Applied", width="stretch"):
            try:
                database_path = services.current_workspace().tracker_database_path
                if database_path is None:
                    raise WorkspaceError("Tracker is unavailable in Demo workspace.")
                services.run_with_captured_output(update_status, int(tracker_row["id"]), "applied", database_path)
                st.success("Marked as applied.")
                st.rerun()
            except Exception as error:  # noqa: BLE001
                st.error(str(error))
    with action_right:
        if st.button("Open Tracker", key="package_open_tracker", width="stretch"):
            services.go_to_page("Tracker")


def _render_secondary_materials(
    artifacts: CoverLetterArtifacts,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    with st.expander("Supporting materials and details", expanded=False):
        if tracker_row:
            services.render_action_callout(
                tracker_next_action(tracker_row),
                caution=tracker_follow_up_due(tracker_row),
            )
            notes = str(tracker_row.get("notes", "") or "").strip()
            if notes:
                st.caption(f"Tracker notes: {notes}")

        services.render_readiness_checklist(
            artifacts.markdown,
            artifacts.docx,
            artifacts.analysis,
            artifacts.internal_notes,
        )
        _render_secondary_downloads(artifacts, package_key)
        services.render_markdown_file(artifacts.analysis, "Stored Match Report")
        internal_text = "\n\n".join(services.read_text_file(path) for path in artifacts.internal_notes)
        if internal_text:
            with st.expander("Internal Notes", expanded=False):
                st.markdown(internal_text)
        st.caption(f"Bundle folder: {services.relative_path(artifacts.package_dir)}")


def _render_secondary_downloads(artifacts: CoverLetterArtifacts, package_key: str) -> None:
    left, middle, right = st.columns(3)
    with left:
        if artifacts.analysis.exists():
            st.download_button(
                "Match Report",
                data=artifacts.analysis.read_bytes(),
                file_name=artifacts.analysis.name,
                mime="text/markdown",
                key=f"download_match_report_{package_key}",
                width="stretch",
            )
    with middle:
        _render_internal_notes_download(artifacts.internal_notes, package_key)
    with right:
        zip_bytes, zip_paths = build_application_package_zip(artifacts.package_dir)
        if zip_paths:
            st.download_button(
                "Bundle ZIP",
                data=zip_bytes,
                file_name=package_zip_filename(artifacts.package_dir),
                mime="application/zip",
                key=f"download_full_package_zip_{package_key}",
                width="stretch",
            )


def _render_internal_notes_download(paths: list[Path], package_key: str) -> None:
    if not paths:
        return
    if len(paths) == 1:
        data, name, mime = paths[0].read_bytes(), paths[0].name, "text/markdown"
    else:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in paths:
                archive.write(path, arcname=path.name)
        data, name, mime = buffer.getvalue(), "internal_notes.zip", "application/zip"
    st.download_button(
        "Internal Notes",
        data=data,
        file_name=name,
        mime=mime,
        key=f"download_internal_notes_{package_key}",
        width="stretch",
    )
