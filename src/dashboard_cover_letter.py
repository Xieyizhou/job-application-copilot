"""Cover Letter page for reviewing and exporting generated bundles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import streamlit as st

from dashboard_cover_letter_components import render_cover_letter_workspace
from dashboard_cover_letter_selection import select_cover_letter_package


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
    """Render the selected cover-letter draft and its primary download."""
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
        st.info("No cover letters generated yet. Review a job and generate a cover letter.")
        return
    render_cover_letter_workspace(package_dir, tracker_row, services)
