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
        latest_task_states,
        load_jsonl,
        load_queue,
        repeat_conflict_task_ids,
    )
    from ml.annotation_translation import (
        AnnotationTranslationError,
        index_translation_rows,
        verified_task_translation,
    )
    from ml.candidate_judgments import (
        has_complete_candidate_labels,
    )
    from ml import annotation_task_ui as task_ui
except ModuleNotFoundError:
    from annotation import (  # type: ignore[no-redef]
        DEFAULT_EVENTS_PATH,
        DEFAULT_QUEUE_PATH,
        annotation_summary,
        latest_task_states,
        load_jsonl,
        load_queue,
        repeat_conflict_task_ids,
    )
    from annotation_translation import (  # type: ignore[no-redef]
        AnnotationTranslationError,
        index_translation_rows,
        verified_task_translation,
    )
    from candidate_judgments import (  # type: ignore[no-redef]
        has_complete_candidate_labels,
    )
    import annotation_task_ui as task_ui  # type: ignore[no-redef]


NONE_CANDIDATE = task_ui.NONE_CANDIDATE
candidate_label = task_ui.candidate_label
QUEUE_PATH = Path(os.getenv("JOB_COPILOT_ANNOTATION_QUEUE", str(DEFAULT_QUEUE_PATH)))
EVENTS_PATH = Path(os.getenv("JOB_COPILOT_ANNOTATION_EVENTS", str(DEFAULT_EVENTS_PATH)))
TRANSLATIONS_VALUE = os.getenv("JOB_COPILOT_ANNOTATION_TRANSLATIONS", "").strip()
TRANSLATIONS_PATH = Path(TRANSLATIONS_VALUE) if TRANSLATIONS_VALUE else None


def task_matches_view(task: dict[str, Any], state: dict[str, Any] | None, view: str) -> bool:
    """Return whether one task belongs in the requested review queue."""
    if view == "Unlabeled":
        return state is None
    if view == "Uncertain":
        return bool(state and state.get("support_label") == "Uncertain")
    if view == "Skipped":
        return bool(state and state.get("action") == "skip")
    if view == "Completed":
        return has_complete_candidate_labels(task, state)
    if view == "Candidate labels needed":
        return not has_complete_candidate_labels(task, state)
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


def save_label(
    task: dict[str, Any],
    selected_candidate_id: str,
    support_label: str,
    cover_letter_safe: bool,
    note: str,
) -> str | None:
    """Preserve the legacy save function for callers and regression tests."""
    return task_ui.save_legacy_label(
        task,
        selected_candidate_id,
        support_label,
        cover_letter_safe,
        note,
        events_path=EVENTS_PATH,
    )


def save_complete_label(
    task: dict[str, Any],
    candidate_labels: dict[str, str],
    selected_candidate_id: str,
    cover_letter_safe: bool,
    note: str,
) -> str | None:
    """Save through the focused task-control module."""
    return task_ui.save_complete_label(
        task,
        candidate_labels,
        selected_candidate_id,
        cover_letter_safe,
        note,
        events_path=EVENTS_PATH,
    )


def render_progress(summary: dict[str, Any]) -> None:
    """Render queue progress and consistency without model diagnostics."""
    total = int(summary["total"])
    completed = int(summary.get("candidate_labels_completed", summary["completed"]))
    remaining = int(summary.get("candidate_labels_remaining", summary["remaining"]))
    st.progress(completed / total if total else 0.0)
    left, middle, right = st.columns(3)
    left.metric("Candidate-complete", completed)
    middle.metric("Remaining", remaining)
    agreement = summary["repeat_agreement"]
    right.metric("Repeat agreement", "—" if agreement is None else f"{agreement:.0%}")


def render_task(
    task: dict[str, Any],
    state: dict[str, Any] | None,
    *,
    translation: dict[str, Any] | None = None,
    show_translation: bool = False,
) -> None:
    """Render one task while keeping page orchestration compact."""
    task_ui.render_task(
        task,
        state,
        events_path=EVENTS_PATH,
        translation=translation,
        show_translation=show_translation,
        ui=st,
    )


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
    translations: dict[str, dict[str, Any]] = {}
    translation_error = ""
    if TRANSLATIONS_PATH is not None:
        try:
            translations = index_translation_rows(load_jsonl(TRANSLATIONS_PATH))
        except AnnotationTranslationError as error:
            translation_error = str(error)
    summary = annotation_summary(tasks, states)
    render_progress(summary)

    with st.sidebar:
        st.markdown("### Queue")
        views = [
            "Candidate labels needed",
            "Unlabeled",
            "Conflicts",
            "Uncertain",
            "Skipped",
            "Completed",
            "All",
        ]
        view = st.radio(
            "View",
            views,
        )
        families = ["All", *sorted({str(task["role_family"]) for task in tasks})]
        role_family = st.selectbox("Role family", families)
        show_translation = st.toggle(
            "显示中文辅助翻译",
            value=bool(translations),
            disabled=not translations,
        )
        st.caption(f"Local queue: {QUEUE_PATH.name}")
        st.caption(f"Local labels: {EVENTS_PATH.name}")
        if TRANSLATIONS_PATH is not None:
            st.caption(f"Local translations: {TRANSLATIONS_PATH.name}")
        if translation_error:
            st.warning(f"Translation sidecar ignored: {translation_error}")

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
    current_task = current_tasks[cursor]
    translation = verified_task_translation(current_task, translations)
    if show_translation and translations and translation is None:
        st.warning(
            "This task's translation does not match the current English source and was hidden."
        )
    render_task(
        current_task,
        states.get(current_task["task_id"]),
        translation=translation,
        show_translation=show_translation,
    )


if __name__ == "__main__":
    main()
