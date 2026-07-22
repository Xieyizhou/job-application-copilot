"""Compact Tracker table, stage update, and secondary record details."""

from __future__ import annotations

from typing import Any

from dashboard_review import tracker_age_days, tracker_follow_up_due, tracker_next_action
from dashboard_titles import display_title_from_value
from scoring_types import TrackerRow
from tracker import delete_application, update_status
from workspace import WorkspaceError


def tracker_summary(rows: list[TrackerRow]) -> dict[str, int]:
    """Return the three pipeline counts used on the default Tracker view."""
    active = [row for row in rows if str(row.get("status", "")).lower() not in {"rejected", "archived"}]
    return {
        "active": len(active),
        "follow_ups": sum(1 for row in rows if tracker_follow_up_due(row)),
        "interviews": sum(1 for row in rows if str(row.get("status", "")).lower() == "interview"),
    }


def render_tracker_summary(ui: Any, rows: list[TrackerRow]) -> None:
    """Render only actionable pipeline counters."""
    summary = tracker_summary(rows)
    columns = ui.columns(3)
    columns[0].metric("Active", summary["active"])
    columns[1].metric("Follow-ups due", summary["follow_ups"])
    columns[2].metric("Interviews", summary["interviews"])
    if summary["follow_ups"]:
        ui.warning(f"{summary['follow_ups']} application(s) need a follow-up or outcome update.")


def render_tracker_filters(ui: Any, status_options: list[str]) -> dict[str, Any]:
    """Keep nonessential filtering controls collapsed."""
    with ui.expander("Filter and sort", expanded=False):
        left, middle, right = ui.columns(3)
        with left:
            statuses = ui.multiselect("Status", status_options, default=status_options)
        with middle:
            minimum_score = ui.slider("Minimum stored Role Fit", min_value=0, max_value=100, value=0)
        with right:
            company_search = ui.text_input("Company or keyword", value="")
        sort_left, sort_right = ui.columns(2)
        with sort_left:
            sort_by = ui.selectbox("Sort by", ["created_at", "match_score", "status"], index=0)
        with sort_right:
            descending = ui.checkbox("Newest / highest first", value=True)
    return {
        "statuses": statuses,
        "minimum_score": minimum_score,
        "company_search": company_search,
        "sort_by": sort_by,
        "descending": descending,
    }


def render_tracker_table(ui: Any, rows: list[TrackerRow]) -> None:
    """Render the application list before any selected-record details."""
    if not rows:
        ui.info("No tracker records match these filters. Save a reviewed job to start the pipeline.")
        return
    ui.dataframe(
        [
            {
                "Stage": str(row["status"]).title(),
                "Company": row["company"],
                "Role": display_title_from_value(row["role"], fallback="Sample Job"),
                "Location": row.get("location") or "—",
                "Age": f"{tracker_age_days(row)}d" if tracker_age_days(row) is not None else "—",
                "Next action": tracker_next_action(row),
            }
            for row in rows
        ],
        width="stretch",
        hide_index=True,
    )


def select_tracker_row(ui: Any, rows: list[TrackerRow]) -> TrackerRow | None:
    """Select a readable application label while retaining the numeric record id."""
    if not rows:
        return None
    selected_id = ui.selectbox(
        "Application",
        [row["id"] for row in rows],
        key="tracker_selected_id",
        format_func=lambda value: _tracker_label(rows, value),
    )
    return next((row for row in rows if row["id"] == selected_id), None)


def _tracker_label(rows: list[TrackerRow], record_id: Any) -> str:
    row = next((item for item in rows if item["id"] == record_id), None)
    if row is None:
        return str(record_id)
    return f"{row.get('company', '—')} · {display_title_from_value(row.get('role'), fallback='Sample Job')}"


def render_selected_tracker_row(
    ui: Any,
    row: TrackerRow,
    status_options: list[str],
    services: Any,
) -> None:
    """Make stage movement primary and keep historical metadata secondary."""
    record_id = int(row["id"])
    ui.markdown("**Update application**")
    ui.markdown(
        f"**{row['company']} · {display_title_from_value(row['role'], fallback='Sample Job')}**"
    )
    ui.caption(f"{row.get('location') or 'Location not recorded'} · Current stage: {str(row['status']).title()}")
    services.render_action_callout(tracker_next_action(row), caution=tracker_follow_up_due(row))

    current_status = str(row.get("status", "saved")).lower()
    stage_index = status_options.index(current_status) if current_status in status_options else 0
    stage_left, stage_right = ui.columns([0.65, 0.35])
    with stage_left:
        new_status = ui.selectbox("Move to stage", status_options, index=stage_index)
    with stage_right:
        ui.caption("The original applied date is preserved.")
    if ui.button("Update Stage", key=f"update_{record_id}", type="primary", width="stretch"):
        _update_stage(ui, row, new_status, services)

    with ui.expander("Application details", expanded=False):
        if row.get("job_url"):
            ui.link_button("Open original job", str(row["job_url"]), width="stretch")
        ui.write(f"Stored recommendation: {row.get('recommendation') or 'Not recorded'}")
        role_fit = f"{row['match_score']}/100" if row.get("match_score") is not None else "Not recorded"
        ui.write(f"Stored Role Fit: {role_fit}")
        ui.write(f"Applied: {row.get('applied_date') or 'Not yet'}")
        material_count = sum(
            bool(str(row.get(key, "") or "").strip())
            for key in ["resume_file", "cover_letter_file"]
        )
        ui.write(f"Application documents recorded: {material_count}/2")
        if str(row.get("notes", "") or "").strip():
            ui.write(f"Notes: {row['notes']}")

    _render_delete_controls(ui, row, services)


def _update_stage(ui: Any, row: TrackerRow, new_status: str, services: Any) -> None:
    try:
        database_path = services.current_workspace().tracker_database_path
        if database_path is None:
            raise WorkspaceError("Tracker is unavailable in Demo workspace.")
        _, output = services.run_with_captured_output(update_status, int(row["id"]), new_status, database_path)
        ui.success(f"Updated application #{row['id']} to {new_status}.")
        if output:
            with ui.expander("Update details", expanded=False):
                ui.text(output)
        ui.rerun()
    except Exception as error:  # noqa: BLE001
        ui.error(str(error))


def _render_delete_controls(ui: Any, row: TrackerRow, services: Any) -> None:
    record_id = int(row["id"])
    with ui.expander("Archive / delete", expanded=False):
        ui.caption("Prefer Archived for history. Delete removes only the local tracker row.")
        confirm_delete = ui.checkbox(
            "I understand this only deletes the tracker record.",
            key=f"confirm_delete_{record_id}",
        )
        if not ui.button("Delete Record", key=f"delete_{record_id}"):
            return
        if not confirm_delete:
            ui.warning("Check the confirmation box before deleting this tracker record.")
            return
        try:
            database_path = services.current_workspace().tracker_database_path
            if database_path is None:
                raise WorkspaceError("Tracker is unavailable in Demo workspace.")
            services.run_with_captured_output(delete_application, record_id, database_path)
            ui.success(f"Deleted tracker record #{record_id}.")
            ui.rerun()
        except Exception as error:  # noqa: BLE001
            ui.error(str(error))
