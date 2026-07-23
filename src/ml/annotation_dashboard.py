"""Standalone local ML interface for requirement/evidence dataset labeling."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from ml.annotation import (
        DEFAULT_EVENTS_PATH,
        DEFAULT_QUEUE_PATH,
        annotation_summary,
        append_event,
        latest_task_states,
        load_jsonl,
        load_queue,
        repeat_conflict_task_ids,
    )
except ModuleNotFoundError:
    from annotation import (  # type: ignore[no-redef]
        DEFAULT_EVENTS_PATH,
        DEFAULT_QUEUE_PATH,
        annotation_summary,
        append_event,
        latest_task_states,
        load_jsonl,
        load_queue,
        repeat_conflict_task_ids,
    )


NONE_CANDIDATE = "__none__"
QUEUE_PATH = Path(os.getenv("JOB_COPILOT_ANNOTATION_QUEUE", str(DEFAULT_QUEUE_PATH)))
EVENTS_PATH = Path(os.getenv("JOB_COPILOT_ANNOTATION_EVENTS", str(DEFAULT_EVENTS_PATH)))


def task_matches_view(task: dict[str, Any], state: dict[str, Any] | None, view: str) -> bool:
    """Return whether one task belongs in the requested review queue."""
    if view == "Unlabeled":
        return state is None
    if view == "Uncertain":
        return bool(state and state.get("support_label") == "Uncertain")
    if view == "Skipped":
        return bool(state and state.get("action") == "skip")
    if view == "Completed":
        return bool(state and state.get("action") == "label")
    return True


def visible_tasks(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
    *,
    view: str,
    role_family: str,
) -> list[dict[str, Any]]:
    """Filter tasks without exposing weak labels or retrieval scores."""
    conflict_ids = repeat_conflict_task_ids(tasks, states) if view == "Conflicts" else set()
    return [
        task
        for task in tasks
        if (role_family == "All" or task.get("role_family") == role_family)
        and (
            task["task_id"] in conflict_ids
            if view == "Conflicts"
            else task_matches_view(task, states.get(task["task_id"]), view)
        )
    ]


def candidate_label(task: dict[str, Any], candidate_id: str) -> str:
    """Render one readable evidence choice without its model score."""
    if candidate_id == NONE_CANDIDATE:
        return "None — no candidate supports this requirement"
    candidates = list(task["candidates"])
    index, candidate = next(
        (index, item)
        for index, item in enumerate(candidates)
        if item["candidate_id"] == candidate_id
    )
    return f"{chr(65 + index)} — {candidate['evidence']}"


def save_label(
    task: dict[str, Any],
    selected_candidate_id: str,
    support_label: str,
    cover_letter_safe: bool,
    note: str,
) -> str | None:
    """Validate and append one local label event; return an error message if invalid."""
    selected = None if selected_candidate_id == NONE_CANDIDATE else selected_candidate_id
    if support_label in {"Direct", "Partial"} and selected is None:
        return "Choose the best supporting evidence before using Direct or Partial."
    if support_label == "No Support":
        selected = None
        cover_letter_safe = False
    append_event(
        task["task_id"],
        "label",
        events_path=EVENTS_PATH,
        selected_candidate_id=selected,
        support_label=support_label,
        cover_letter_safe=cover_letter_safe if support_label in {"Direct", "Partial"} else None,
        note=note,
    )
    return None


def render_progress(summary: dict[str, Any]) -> None:
    """Render queue progress and consistency without model diagnostics."""
    total = int(summary["total"])
    completed = int(summary["completed"])
    st.progress(completed / total if total else 0.0)
    left, middle, right = st.columns(3)
    left.metric("Completed", completed)
    middle.metric("Remaining", summary["remaining"])
    agreement = summary["repeat_agreement"]
    right.metric("Repeat agreement", "—" if agreement is None else f"{agreement:.0%}")


def render_task(task: dict[str, Any], state: dict[str, Any] | None) -> None:
    """Render a single compact annotation decision."""
    task_id = str(task["task_id"])
    st.caption(f"Role family: {task['role_family']}")
    st.markdown("### Requirement")
    st.info(str(task["requirement"]))
    st.markdown("### Best resume evidence")
    options = [str(item["candidate_id"]) for item in task["candidates"]] + [NONE_CANDIDATE]
    default_candidate = str(state.get("selected_candidate_id") or NONE_CANDIDATE) if state else NONE_CANDIDATE
    if default_candidate not in options:
        default_candidate = NONE_CANDIDATE
    selected = st.radio(
        "Select one candidate",
        options,
        index=options.index(default_candidate),
        format_func=lambda candidate_id: candidate_label(task, candidate_id),
        key=f"candidate_{task_id}",
        label_visibility="collapsed",
    )
    safe_default = bool(state and state.get("cover_letter_safe"))
    cover_letter_safe = st.checkbox(
        "Safe to quote or paraphrase in a cover letter",
        value=safe_default,
        key=f"safe_{task_id}",
    )
    with st.expander("Optional note", expanded=False):
        note = st.text_area(
            "Reason or ambiguity",
            value=str(state.get("note", "")) if state else "",
            key=f"note_{task_id}",
        )

    st.markdown("### Support decision")
    direct, partial, unsupported, uncertain = st.columns(4)
    decisions = [
        (direct, "Direct", "Direct support"),
        (partial, "Partial", "Partial support"),
        (unsupported, "No Support", "No support"),
        (uncertain, "Uncertain", "Uncertain"),
    ]
    for column, label, button_text in decisions:
        with column:
            if st.button(button_text, key=f"label_{label}_{task_id}", width="stretch"):
                error = save_label(task, selected, label, cover_letter_safe, note)
                if error:
                    st.error(error)
                else:
                    st.rerun()

    footer_left, footer_middle, footer_right = st.columns(3)
    with footer_left:
        if st.button("Skip", key=f"skip_{task_id}", width="stretch"):
            append_event(task_id, "skip", events_path=EVENTS_PATH)
            st.rerun()
    with footer_middle:
        if state and st.button("Clear label", key=f"clear_{task_id}", width="stretch"):
            append_event(task_id, "clear", events_path=EVENTS_PATH)
            st.rerun()
    with footer_right:
        st.caption("Candidate order is randomized; retrieval rank and similarity are not stored.")


def main() -> None:
    """Run the local evidence annotation workspace."""
    st.set_page_config(page_title="Evidence Dataset Labeling", layout="wide")
    st.title("Evidence Dataset Labeling")
    st.caption("Review one requirement at a time. Labels are saved locally and never submitted.")
    if not QUEUE_PATH.is_file():
        st.warning("No local annotation queue was found.")
        st.code("python scripts/ml/build_annotation_queue.py")
        return
    tasks = load_queue(QUEUE_PATH)
    states = latest_task_states(load_jsonl(EVENTS_PATH))
    summary = annotation_summary(tasks, states)
    render_progress(summary)

    with st.sidebar:
        st.markdown("### Queue")
        view = st.radio(
            "View",
            ["Unlabeled", "Conflicts", "Uncertain", "Skipped", "Completed", "All"],
        )
        families = ["All", *sorted({str(task["role_family"]) for task in tasks})]
        role_family = st.selectbox("Role family", families)
        st.caption(f"Local queue: {QUEUE_PATH.name}")
        st.caption(f"Local labels: {EVENTS_PATH.name}")

    current_tasks = visible_tasks(tasks, states, view=view, role_family=role_family)
    if not current_tasks:
        message = (
            "All repeat conflicts are resolved."
            if view == "Conflicts"
            else "No tasks remain in this view."
        )
        st.success(message)
        return
    if view == "Conflicts":
        st.warning(
            "These decisions disagree with a hidden repeat. "
            "Review the requirement and evidence again using the same criteria."
        )
    cursor_key = f"annotation_cursor_{view}_{role_family}"
    cursor = min(int(st.session_state.get(cursor_key, 0)), len(current_tasks) - 1)
    st.session_state[cursor_key] = cursor
    previous, position, next_column = st.columns([0.2, 0.6, 0.2])
    with previous:
        if st.button("Previous", disabled=cursor == 0, width="stretch"):
            st.session_state[cursor_key] = max(0, cursor - 1)
            st.rerun()
    with position:
        st.caption(f"Task {cursor + 1} of {len(current_tasks)} in this view")
    with next_column:
        if st.button("Next", disabled=cursor >= len(current_tasks) - 1, width="stretch"):
            st.session_state[cursor_key] = min(len(current_tasks) - 1, cursor + 1)
            st.rerun()
    render_task(current_tasks[cursor], states.get(current_tasks[cursor]["task_id"]))


if __name__ == "__main__":
    main()
