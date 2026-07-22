"""Cover-letter bundle selection for the dashboard."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st


def select_cover_letter_package(
    services: Any,
    *,
    demo_package_dir: Path,
    project_root: Path,
) -> tuple[Path | None, dict[str, Any] | None]:
    """Return the selected bundle and optional tracker record."""
    if services.demo_mode_enabled():
        st.info("This sanitized sample shows the files produced in Personal workspace.")
        if not demo_package_dir.exists():
            st.info("Demo cover-letter sample is unavailable.")
            return None, None
        return demo_package_dir, None

    records = services.load_tracker_rows(sort_by="created_at", descending=True)
    source = st.radio(
        "Choose cover letter",
        ["Tracker record", "Bundle folder"],
        horizontal=True,
    )
    if source == "Tracker record":
        return _select_tracker_package(records, services), _selected_tracker_row(records)
    return _select_bundle_folder(services, project_root), None


def _selected_tracker_row(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not records:
        return None
    tracker_id = st.session_state.get("package_viewer_tracker_id")
    return next((row for row in records if row["id"] == tracker_id), None)


def _select_tracker_package(records: list[dict[str, Any]], services: Any) -> Path | None:
    if not records:
        latest_dir = st.session_state.get("latest_generated_package_dir", "")
        candidate = Path(latest_dir) if latest_dir else None
        if candidate and candidate.exists() and candidate.is_dir():
            st.info("Showing the latest generated cover letter.")
            return candidate
        return None

    tracker_id = st.selectbox(
        "Saved application",
        [row["id"] for row in records],
        key="package_viewer_tracker_id",
        format_func=lambda value: _tracker_label(records, value),
    )
    tracker_row = next((row for row in records if row["id"] == tracker_id), None)
    if tracker_row is None:
        return None
    return services.resolve_package_dir_from_tracker(tracker_row) or services.latest_package_for_company_role(
        tracker_row["company"], tracker_row["role"]
    )


def _tracker_label(records: list[dict[str, Any]], tracker_id: Any) -> str:
    row = next((item for item in records if item["id"] == tracker_id), None)
    if row is None:
        return str(tracker_id)
    return f"{row.get('company', '-')} · {row.get('role', '-')}"


def _select_bundle_folder(services: Any, project_root: Path) -> Path | None:
    folder_input = st.text_input(
        "Bundle folder",
        value=str(services.current_workspace().generated_dir),
    )
    candidate = Path(folder_input).expanduser()
    if not candidate.is_absolute():
        candidate = project_root / candidate
    candidate = candidate.resolve()
    generated_root = services.current_workspace().generated_dir.resolve()
    if candidate.exists() and candidate.is_dir() and candidate.is_relative_to(generated_root):
        return candidate
    st.warning("Choose a bundle folder inside the Personal workspace generated directory.")
    return None
