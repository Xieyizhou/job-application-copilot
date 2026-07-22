"""Application tracker page for the local toolkit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_tracker_components import (
    render_selected_tracker_row,
    render_tracker_filters,
    render_tracker_summary,
    render_tracker_table,
    select_tracker_row,
)
from scoring_types import TrackerRow
from tracker import (
    VALID_STATUSES,
    delete_application as delete_application,
    update_status as update_status,
)


@dataclass(frozen=True)
class TrackerPageServices:
    """Shared dashboard operations required by the tracker page."""

    current_workspace: Callable[[], Any]
    demo_mode_enabled: Callable[[], bool]
    load_tracker_rows: Callable[..., list[TrackerRow]]
    render_action_callout: Callable[..., None]
    render_page_header: Callable[[str, str | None], None]
    run_with_captured_output: Callable[..., tuple[Any, str]]


def tracker_tab(services: TrackerPageServices) -> None:
    """Render the application list before the selected stage workflow."""
    services.render_page_header(
        "Tracker",
        "Keep application stages current and act on the next follow-up.",
    )
    if services.demo_mode_enabled():
        st.info("Demo workspace does not read or update the Personal tracker database.")
        return

    status_options = [
        status
        for status in ["saved", "ready", "applied", "interview", "rejected", "archived"]
        if status in VALID_STATUSES
    ]
    all_records = services.load_tracker_rows(sort_by="created_at", descending=True)
    render_tracker_summary(st, all_records)
    filters = render_tracker_filters(st, status_options)
    records = services.load_tracker_rows(
        statuses=filters["statuses"],
        minimum_score=filters["minimum_score"],
        company_search=filters["company_search"],
        sort_by=filters["sort_by"],
        descending=filters["descending"],
    )
    render_tracker_table(st, records)
    selected_row = select_tracker_row(st, records)
    if selected_row is not None:
        render_selected_tracker_row(st, selected_row, status_options, services)
