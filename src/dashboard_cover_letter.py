"""Cover Letter page for reviewing and exporting generated bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from dashboard_cover_letter_components import (
    cover_letter_artifacts,
    render_cover_letter_context,
    render_cover_letter_document,
)
from dashboard_cover_letter_selection import select_cover_letter_package
from dashboard_desktop_styles import render_cover_letter_workspace_styles
from output_paths import safe_slug
from scoring_types import TrackerRow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_PACKAGE_DIR = PROJECT_ROOT / "data" / "demo" / "sample_package"


@dataclass(frozen=True)
class CoverLetterPageServices:
    """Shared dashboard operations required by the Cover Letter page."""

    current_workspace: Callable[[], Any]
    demo_mode_enabled: Callable[[], bool]
    generate_cover_letter_docx_for_package: Callable[..., Any]
    go_to_page: Callable[[str], None]
    latest_package_for_company_role: Callable[..., Any]
    load_package_notes: Callable[[Path], str]
    load_tracker_rows: Callable[..., list[TrackerRow]]
    read_text_file: Callable[[Path], str]
    relative_path: Callable[[Path], str]
    render_action_callout: Callable[..., None]
    render_markdown_file: Callable[[Path, str], None]
    render_page_header: Callable[[str, str | None], None]
    render_readiness_checklist: Callable[..., None]
    resolve_package_dir_from_tracker: Callable[[dict[str, Any]], Path | None]
    run_with_captured_output: Callable[..., tuple[Any, str]]


def package_viewer_tab(services: CoverLetterPageServices) -> None:
    """Render the selected cover-letter draft and its primary download."""
    st.markdown(
        '<div class="desktop-workspace-marker"></div>',
        unsafe_allow_html=True,
    )
    render_cover_letter_workspace_styles()
    _render_cover_letter_panels(services)


def _render_cover_letter_panels(services: CoverLetterPageServices) -> None:
    """Render aligned context and document panes."""
    context_panel, document_panel = st.columns(
        [0.4, 0.6],
        gap="large",
        vertical_alignment="top",
    )
    with context_panel, st.container(key="cover_letter_context_panel"):
        services.render_page_header(
            "Cover Letter",
            "Review the draft against your resume, edit it, and download the employer-facing DOCX.",
        )
        package_dir, tracker_row = select_cover_letter_package(
            services,
            demo_package_dir=DEMO_PACKAGE_DIR,
            project_root=PROJECT_ROOT,
        )
        if package_dir is None:
            st.info(
                "No cover letters generated yet. Review a job and generate a cover letter."
            )
            return
        artifacts = cover_letter_artifacts(package_dir)
        package_key = safe_slug(services.relative_path(package_dir)) or "selected_package"
        render_cover_letter_context(artifacts, tracker_row, services)
    with document_panel, st.container(key="cover_letter_document_panel"):
        render_cover_letter_document(
            artifacts,
            tracker_row,
            package_key,
            services,
        )
