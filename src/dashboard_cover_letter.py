"""Cover Letter page for reviewing and exporting generated bundles."""

from __future__ import annotations

import io
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from dashboard_packages import (
    build_application_package_zip,
    existing_package_files,
    package_zip_filename,
)
from dashboard_review import tracker_follow_up_due, tracker_next_action
from dashboard_titles import display_title_from_value
from output_paths import safe_slug
from tracker import update_status
from workspace import WorkspaceError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACKAGE_DIR = PROJECT_ROOT / "data" / "demo" / "sample_package"
INTERNAL_NOTES_FILE_ORDER = ["cover_letter_notes.md"]


@dataclass(frozen=True)
class CoverLetterPageServices:
    """Shared dashboard operations required by the Cover Letter page."""

    current_workspace: Callable[[], Any]
    demo_mode_enabled: Callable[[], bool]
    generate_cover_letter_docx_for_package: Callable[..., Any]
    go_to_page: Callable[[str], None]
    latest_package_for_company_role: Callable[..., Any]
    load_package_notes: Callable[[Path], str]
    load_tracker_rows: Callable[..., list[dict[str, Any]]]
    read_text_file: Callable[[Path], str]
    relative_path: Callable[[Path], str]
    render_action_callout: Callable[..., None]
    render_markdown_file: Callable[[Path, str], None]
    render_page_header: Callable[[str, str | None], None]
    render_readiness_checklist: Callable[..., None]
    resolve_package_dir_from_tracker: Callable[[dict[str, Any]], Path | None]
    run_with_captured_output: Callable[..., tuple[Any, str]]


def package_viewer_tab(services: CoverLetterPageServices) -> None:
    """Render a viewer for tracker-linked cover-letter bundles."""
    services.render_page_header(
        "Cover Letter",
        "Review a resume-grounded draft, its exact evidence trace, and its honest gaps before using it.",
    )
    if services.demo_mode_enabled():
        st.info("This sanitized sample demonstrates the cover letter and review files produced in Personal workspace.")
        package_dir = DEMO_PACKAGE_DIR if DEMO_PACKAGE_DIR.exists() else None
        tracker_row = None
        if package_dir is None:
            st.info("Demo cover-letter sample is unavailable.")
            return
    else:
        package_dir = None
        tracker_row = None

    all_records = services.load_tracker_rows(sort_by="created_at", descending=True)

    if not services.demo_mode_enabled():
        view_mode = st.radio(
            "View source",
            ["Tracker record", "Bundle folder path"],
            horizontal=True,
        )
    else:
        view_mode = "Demo cover letter"

    if view_mode == "Tracker record":
        if not all_records:
            latest_dir = st.session_state.get("latest_generated_package_dir", "")
            if latest_dir:
                candidate = Path(latest_dir)
                if candidate.exists() and candidate.is_dir():
                    package_dir = candidate
                    st.info("Showing the latest generated cover letter. Save jobs to tracker to browse by tracker record.")
                else:
                    st.info("No cover letters generated yet. Review a job and generate a cover letter.")
                    return
            else:
                st.info("No cover letters generated yet. Review a job and generate a cover letter.")
                return

        if all_records:
            tracker_id = st.selectbox(
                "Select tracker id",
                [row["id"] for row in all_records],
                key="package_viewer_tracker_id",
            )
            tracker_row = next((row for row in all_records if row["id"] == tracker_id), None)
            if tracker_row:
                package_dir = services.resolve_package_dir_from_tracker(tracker_row)
    elif view_mode == "Bundle folder path":
        folder_input = st.text_input(
            "Bundle folder path",
            value=str(services.current_workspace().generated_dir),
        )
        candidate = Path(folder_input).expanduser()
        if not candidate.is_absolute():
            candidate = PROJECT_ROOT / candidate
        candidate = candidate.resolve()
        generated_root = services.current_workspace().generated_dir.resolve()
        if candidate.exists() and candidate.is_dir() and candidate.is_relative_to(generated_root):
            package_dir = candidate
        else:
            st.warning("Enter a bundle folder inside the Personal workspace generated directory.")

    if tracker_row:
        package_dir = package_dir or services.latest_package_for_company_role(
            tracker_row["company"],
            tracker_row["role"],
        )
        summary_left, summary_right = st.columns([0.64, 0.36], gap="large")
        with summary_left:
            st.markdown(f"**{tracker_row['company']}**")
            st.write(display_title_from_value(tracker_row["role"], fallback="Sample Job"))
            st.caption(
                f"Status: {tracker_row['status']} | Stored generation-time Role Fit: "
                f"{tracker_row['match_score'] if tracker_row['match_score'] is not None else '-'}"
            )
        with summary_right:
            if tracker_row["job_url"]:
                st.link_button("Open Job URL", tracker_row["job_url"], width="stretch")
            notes = str(tracker_row["notes"] or "").strip()
            if notes:
                with st.expander("Tracker notes", expanded=False):
                    st.write(notes)
    elif package_dir is not None:
        notes = services.load_package_notes(package_dir)
        if notes:
            with st.expander("Cover letter notes", expanded=False):
                st.markdown(notes)

    if package_dir is None:
        st.info("No cover letters generated yet. Review a job and generate a cover letter.")
        return

    analysis_path = package_dir / "analysis.md"
    cover_letter_md_path = package_dir / "cover_letter.md"
    cover_letter_docx_path = package_dir / "cover_letter.docx"
    internal_notes_paths = existing_package_files(package_dir, INTERNAL_NOTES_FILE_ORDER)
    internal_notes = "\n\n".join(services.read_text_file(path) for path in internal_notes_paths)

    material_checks = {
        "Cover letter source": cover_letter_md_path.exists(),
        "Cover letter DOCX": cover_letter_docx_path.exists(),
        "Match report": analysis_path.exists(),
        "Internal notes": bool(internal_notes_paths),
    }
    ready_materials = sum(material_checks.values())
    package_metrics = st.columns(4)
    package_metrics[0].metric(
        "Stored Role Fit",
        f"{tracker_row['match_score']}/100" if tracker_row and tracker_row.get("match_score") is not None else "Snapshot",
    )
    package_metrics[1].metric("Materials ready", f"{ready_materials}/{len(material_checks)}")
    package_metrics[2].metric("Application stage", str(tracker_row.get("status", "Draft")).title() if tracker_row else "Draft")
    package_metrics[3].metric("Employer files", int(material_checks["Cover letter DOCX"]))
    if tracker_row:
        services.render_action_callout(
            tracker_next_action(tracker_row),
            caution=tracker_follow_up_due(tracker_row),
        )
    else:
        services.render_action_callout("Review every claim and its resume evidence before sharing this cover letter.")

    package_key = safe_slug(services.relative_path(package_dir)) or "selected_package"
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    flash_key = f"package_flash_{package_key}"
    flash_message = st.session_state.pop(flash_key, "")
    if flash_message:
        st.success(flash_message)

    st.markdown("**Cover letter readiness and actions**")
    package_left, package_right = st.columns([0.48, 0.52], gap="large")
    with package_left:
        services.render_readiness_checklist(
            cover_letter_md_path,
            cover_letter_docx_path,
            analysis_path,
            internal_notes_paths,
        )
        st.caption("Review generated files before sharing them with employers.")

        if not cover_letter_docx_path.exists() and cover_letter_md_path.exists():
            if services.demo_mode_enabled():
                st.caption("Cover Letter DOCX unavailable in this sample.")
            elif st.button("Generate Cover Letter DOCX", key=f"generate_cover_letter_docx_{package_key}"):
                try:
                    generated_path, warnings = services.generate_cover_letter_docx_for_package(package_dir)
                    if generated_path:
                        st.session_state[flash_key] = "Cover Letter DOCX generated."
                        if warnings:
                            with st.expander("Advanced: cover letter DOCX warnings", expanded=False):
                                for warning in warnings:
                                    st.write(f"- {warning}")
                        st.rerun()
                    else:
                        st.info("Cover Letter DOCX could not be generated because the cover letter source is missing.")
                except Exception as error:  # noqa: BLE001
                    st.error(f"Could not generate Cover Letter DOCX: {error}")
        elif not cover_letter_docx_path.exists():
            st.info("Cover Letter DOCX needs a cover letter source before it can be generated.")

    with package_right:
        st.warning("Before sending: verify company/title, truthful claims, dates, contact details, formatting, and file names.")
        download_left, download_right = st.columns(2)
        with download_left:
            if analysis_path.exists():
                st.download_button(
                    "Download Match Report",
                    data=analysis_path.read_bytes(),
                    file_name=analysis_path.name,
                    mime="text/markdown",
                    key=f"download_match_report_{package_key}",
                    width="stretch",
                )
            else:
                st.info("Match report not available yet")

        with download_right:
            if cover_letter_docx_path.exists():
                st.download_button(
                    "Download Cover Letter DOCX",
                    data=cover_letter_docx_path.read_bytes(),
                    file_name=cover_letter_docx_path.name,
                    mime=docx_mime,
                    key=f"download_cover_letter_docx_{package_key}",
                    width="stretch",
                )
            elif not services.demo_mode_enabled():
                st.info("Cover Letter DOCX not generated yet")

            if internal_notes_paths:
                if len(internal_notes_paths) == 1:
                    internal_notes_data = internal_notes_paths[0].read_bytes()
                    internal_notes_file_name = internal_notes_paths[0].name
                    internal_notes_mime = "text/markdown"
                else:
                    internal_notes_buffer = io.BytesIO()
                    with zipfile.ZipFile(internal_notes_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                        for notes_path in internal_notes_paths:
                            zip_file.write(notes_path, arcname=notes_path.name)
                    internal_notes_data = internal_notes_buffer.getvalue()
                    internal_notes_file_name = "internal_notes.zip"
                    internal_notes_mime = "application/zip"
                st.download_button(
                    "Download Internal Notes",
                    data=internal_notes_data,
                    file_name=internal_notes_file_name,
                    mime=internal_notes_mime,
                    key=f"download_internal_notes_{package_key}",
                    width="stretch",
                )
            else:
                st.info("Internal notes not available yet")

        zip_bytes, zip_paths = build_application_package_zip(package_dir)
        if zip_paths:
            st.download_button(
                "Download Cover Letter Bundle ZIP",
                data=zip_bytes,
                file_name=package_zip_filename(package_dir),
                mime="application/zip",
                key=f"download_full_package_zip_{package_key}",
                width="stretch",
            )
            st.caption("ZIP includes the cover letter, match report, and evidence-trace notes. Your uploaded resume is not duplicated or rewritten.")
        else:
            st.info("The cover letter bundle ZIP is available after generated files exist.")

    st.markdown("**Preview**")
    services.render_markdown_file(cover_letter_md_path, "Preview Cover Letter")
    st.caption("These reports are a generation-time snapshot; current live analysis appears in Review Jobs.")
    services.render_markdown_file(analysis_path, "Preview Stored Match Report")
    if internal_notes:
        with st.expander("Preview Internal Notes", expanded=False):
            st.markdown(internal_notes)

    st.markdown("**Application status**")
    status_left, status_middle, status_right = st.columns(3)
    with status_left:
        if services.demo_mode_enabled():
            st.info("Mark as Applied is disabled in Demo workspace.")
        elif tracker_row:
            if st.button("Mark as Applied"):
                try:
                    database_path = services.current_workspace().tracker_database_path
                    if database_path is None:
                        raise WorkspaceError("Tracker is unavailable in Demo workspace.")
                    _, output = services.run_with_captured_output(
                        update_status, int(tracker_row["id"]), "applied", database_path
                    )
                    st.success("Marked as applied.")
                    if output:
                        with st.expander("Advanced: tracker output", expanded=False):
                            st.text(output)
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
        else:
            st.info("Select a tracker record to mark an application as applied.")
    with status_middle:
        if st.button("Open Tracker", key="package_open_tracker"):
            services.go_to_page("Tracker")
    with status_right:
        if not services.demo_mode_enabled():
            st.info("Your uploaded resume remains unchanged and outside this generated bundle.")

    with st.expander("Advanced: cover-letter files", expanded=False):
        st.write(f"Bundle folder: `{services.relative_path(package_dir)}`")
        for label, path in [
            ("cover_letter.docx", cover_letter_docx_path),
            ("cover_letter.md", cover_letter_md_path),
            ("analysis.md", analysis_path),
            ("cover_letter_notes.md", package_dir / "cover_letter_notes.md"),
        ]:
            if path.exists():
                st.write(f"{label}: `{services.relative_path(path)}`")
            else:
                st.write(f"{label}: not found")

    if cover_letter_docx_path.exists():
        st.caption(f"Cover letter DOCX: `{services.relative_path(cover_letter_docx_path)}`")
