"""Application tracker page for the local toolkit dashboard."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from dashboard_review import tracker_age_days, tracker_follow_up_due, tracker_next_action
from dashboard_titles import display_title_from_value
from scoring_types import TrackerRow
from tracker import VALID_STATUSES, delete_application, update_status
from workspace import WorkspaceError


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
    """Render the local SQLite tracker table and record actions."""
    services.render_page_header(
        "Tracker",
        "Your application pipeline, follow-ups, documents, and outcomes in one local source of truth.",
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
    active_records = [
        row
        for row in all_records
        if str(row.get("status", "")).lower() not in {"rejected", "archived"}
    ]
    sent_records = [
        row
        for row in all_records
        if str(row.get("status", "")).lower() in {"applied", "interview", "rejected"}
    ]
    interview_records = [
        row for row in all_records if str(row.get("status", "")).lower() == "interview"
    ]
    follow_up_records = [row for row in all_records if tracker_follow_up_due(row)]

    pipeline_metrics = st.columns(5)
    pipeline_metrics[0].metric("Active pipeline", len(active_records))
    pipeline_metrics[1].metric(
        "Ready to apply", sum(1 for row in all_records if row.get("status") == "ready")
    )
    pipeline_metrics[2].metric("Applications sent", len(sent_records))
    pipeline_metrics[3].metric("Interviews", len(interview_records))
    pipeline_metrics[4].metric("Follow-ups due", len(follow_up_records))

    if follow_up_records:
        st.warning(
            f"{len(follow_up_records)} applied role(s) have had no stage update for at least seven days. "
            "Open a record below to review the next action."
        )

    with st.expander("Filter and sort", expanded=False):
        filter_left, filter_middle, filter_right = st.columns(3)
        with filter_left:
            selected_statuses = st.multiselect("Status", status_options, default=status_options)
        with filter_middle:
            minimum_score = st.slider("Minimum Role Fit", min_value=0, max_value=100, value=0)
        with filter_right:
            company_search = st.text_input("Company or keyword", value="")
        sort_left, sort_right = st.columns(2)
        with sort_left:
            sort_by = st.selectbox("Sort by", ["created_at", "match_score", "status"], index=0)
        with sort_right:
            descending = st.checkbox("Newest / highest first", value=True)

    records = services.load_tracker_rows(
        statuses=selected_statuses,
        minimum_score=minimum_score,
        company_search=company_search,
        sort_by=sort_by,
        descending=descending,
    )

    if records:
        st.dataframe(
            [
                {
                    "ID": row["id"],
                    "Status": str(row["status"]).title(),
                    "Role Fit": row["match_score"],
                    "Company": row["company"],
                    "Role": display_title_from_value(row["role"], fallback="Sample Job"),
                    "Location": row["location"],
                    "Age": f"{tracker_age_days(row)}d" if tracker_age_days(row) is not None else "—",
                    "Next action": tracker_next_action(row),
                }
                for row in records
            ],
            width="stretch",
            hide_index=True,
        )
    else:
        st.info("No tracker records match these filters. Save a reviewed job to start the pipeline.")

    if not records:
        return

    selected_id = st.selectbox(
        "Select application id",
        [row["id"] for row in records],
        key="tracker_selected_id",
    )
    selected_row = next((row for row in records if row["id"] == selected_id), None)
    if selected_row is None:
        return

    st.markdown("**Selected application**")
    selected_left, selected_right = st.columns([0.7, 0.3], gap="large")
    with selected_left:
        st.markdown(
            f"**{selected_row['company']} · "
            f"{display_title_from_value(selected_row['role'], fallback='Sample Job')}**"
        )
        st.caption(
            f"{selected_row.get('location') or 'Location not recorded'} · "
            f"Tracked {tracker_age_days(selected_row) if tracker_age_days(selected_row) is not None else '—'} days"
        )
        st.write(f"Stored recommendation: {selected_row['recommendation'] or 'Not recorded'}")
    with selected_right:
        st.metric(
            "Stored Role Fit",
            f"{selected_row['match_score']}/100" if selected_row["match_score"] is not None else "—",
        )
        st.caption(f"Current stage: {str(selected_row['status']).title()}")
    services.render_action_callout(
        tracker_next_action(selected_row),
        caution=tracker_follow_up_due(selected_row),
    )

    detail_left, detail_right = st.columns(2)
    with detail_left:
        if selected_row.get("job_url"):
            st.link_button("Open original job", str(selected_row["job_url"]), width="stretch")
        st.caption(f"Applied: {selected_row.get('applied_date') or 'Not yet'}")
    with detail_right:
        material_count = sum(
            bool(str(selected_row.get(key, "") or "").strip())
            for key in ["resume_file", "cover_letter_file"]
        )
        st.write(f"Application documents recorded: {material_count}/2")
        if str(selected_row.get("notes", "") or "").strip():
            with st.expander("Notes", expanded=False):
                st.write(selected_row["notes"])

    status_left, status_right = st.columns([0.68, 0.32])
    with status_left:
        new_status = st.selectbox(
            "Move to stage",
            status_options,
            index=status_options.index(selected_row["status"]),
        )
    with status_right:
        st.caption("Stage updates preserve the original applied date.")
    if st.button("Update Stage", key=f"update_{selected_id}", type="primary"):
        try:
            database_path = services.current_workspace().tracker_database_path
            if database_path is None:
                raise WorkspaceError("Tracker is unavailable in Demo workspace.")
            _, output = services.run_with_captured_output(
                update_status,
                selected_id,
                new_status,
                database_path,
            )
            st.success(f"Updated application #{selected_id} to {new_status}.")
            if output:
                st.text(output)
            st.rerun()
        except Exception as error:  # noqa: BLE001
            st.error(str(error))

    with st.expander("Archive / delete options", expanded=False):
        st.caption(
            "Prefer Archived for history and funnel analysis. Delete only removes the local tracker row."
        )
        confirm_delete = st.checkbox(
            "I understand this only deletes the tracker record.",
            key=f"confirm_delete_{selected_id}",
        )
        if st.button("Delete Record", key=f"delete_{selected_id}"):
            if not confirm_delete:
                st.warning("Check the confirmation box before deleting this tracker record.")
            else:
                try:
                    database_path = services.current_workspace().tracker_database_path
                    if database_path is None:
                        raise WorkspaceError("Tracker is unavailable in Demo workspace.")
                    _, output = services.run_with_captured_output(
                        delete_application,
                        selected_id,
                        database_path,
                    )
                    st.success(f"Deleted tracker record #{selected_id}.")
                    if output:
                        st.text(output)
                    st.rerun()
                except Exception as error:  # noqa: BLE001
                    st.error(str(error))
