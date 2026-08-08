"""Focused supporting-material interactions for the Cover Letter page."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Any, Protocol

import streamlit as st

from dashboard_packages import build_application_package_zip, package_zip_filename
from dashboard_review import tracker_follow_up_due, tracker_next_action


class CoverLetterArtifactsLike(Protocol):
    """Files needed by the supporting-material workspace."""

    @property
    def package_dir(self) -> Path: ...

    @property
    def markdown(self) -> Path: ...

    @property
    def docx(self) -> Path: ...

    @property
    def analysis(self) -> Path: ...

    @property
    def internal_notes(self) -> list[Path]: ...


def _materials_key(package_key: str) -> str:
    return f"cover_letter_materials_open_{package_key}"


def _detail_key(package_key: str) -> str:
    return f"cover_letter_detail_view_{package_key}"


def cover_letter_editor_mode(package_key: str) -> str:
    """Return the CSS mode that gives the active lower panel room."""
    detail = st.session_state.get(_detail_key(package_key))
    if detail:
        return "detail"
    if st.session_state.get(_materials_key(package_key), False):
        return "supporting"
    return "default"


def _toggle_materials(package_key: str) -> None:
    key = _materials_key(package_key)
    st.session_state[key] = not bool(st.session_state.get(key, False))
    st.session_state[_detail_key(package_key)] = None


def _open_detail(package_key: str, detail: str) -> None:
    st.session_state[_materials_key(package_key)] = False
    st.session_state[_detail_key(package_key)] = detail


def _back_to_materials(package_key: str) -> None:
    """Return from a focused artifact to the supporting-material workspace."""
    st.session_state[_detail_key(package_key)] = None
    st.session_state[_materials_key(package_key)] = True


def render_secondary_materials(
    artifacts: CoverLetterArtifactsLike,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    """Render one secondary workspace at a time."""
    materials_open = bool(st.session_state.get(_materials_key(package_key), False))
    detail = st.session_state.get(_detail_key(package_key))
    toggle_label = "Hide supporting materials" if materials_open else "Supporting materials and details"
    toggle_icon = ":material/expand_less:" if materials_open else ":material/expand_more:"
    with st.container(key="cover_letter_supporting_toggle"):
        st.button(
            toggle_label,
            key=f"toggle_cover_letter_materials_{package_key}",
            icon=toggle_icon,
            on_click=_toggle_materials,
            args=(package_key,),
            width="stretch",
        )

    if materials_open:
        _render_materials_panel(artifacts, tracker_row, package_key, services)
    elif detail in {"report", "notes"}:
        _render_focused_detail(artifacts, package_key, str(detail), services)


def _render_materials_panel(
    artifacts: CoverLetterArtifactsLike,
    tracker_row: dict[str, Any] | None,
    package_key: str,
    services: Any,
) -> None:
    with st.container(border=True, key="cover_letter_supporting_panel"):
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
        _render_detail_actions(artifacts, package_key)
        st.caption(f"Bundle folder: {services.relative_path(artifacts.package_dir)}")


def _render_detail_actions(artifacts: CoverLetterArtifactsLike, package_key: str) -> None:
    report_column, notes_column = st.columns(2, gap="small")
    with report_column:
        st.button(
            "View Stored Match Report",
            key=f"view_match_report_{package_key}",
            icon=":material/description:",
            disabled=not artifacts.analysis.exists(),
            on_click=_open_detail,
            args=(package_key, "report"),
            width="stretch",
        )
    with notes_column:
        st.button(
            "View Internal Notes",
            key=f"view_internal_notes_{package_key}",
            icon=":material/sticky_note_2:",
            disabled=not artifacts.internal_notes,
            on_click=_open_detail,
            args=(package_key, "notes"),
            width="stretch",
        )


def _render_focused_detail(
    artifacts: CoverLetterArtifactsLike,
    package_key: str,
    detail: str,
    services: Any,
) -> None:
    title = "Stored Match Report" if detail == "report" else "Internal Notes"
    if detail == "report":
        content = services.read_text_file(artifacts.analysis) if artifacts.analysis.exists() else ""
    else:
        content = "\n\n".join(services.read_text_file(path) for path in artifacts.internal_notes)
    _, center, _ = st.columns([0.02, 0.96, 0.02])
    with center, st.container(border=True, key="cover_letter_focused_detail"):
        title_column, back_column = st.columns([0.78, 0.22], vertical_alignment="center")
        with title_column:
            st.markdown(f"**{title}**")
        with back_column:
            st.button(
                "Back",
                key=f"back_from_cover_letter_detail_{package_key}",
                icon=":material/arrow_back:",
                on_click=_back_to_materials,
                args=(package_key,),
                width="stretch",
            )
        with st.container(height=520, key="cover_letter_focused_detail_body"):
            st.markdown(content or f"{title} not found.")


def _render_secondary_downloads(artifacts: CoverLetterArtifactsLike, package_key: str) -> None:
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
