"""Single-task Streamlit controls for complete evidence judgments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import streamlit as st

try:
    from ml.annotation import append_event
    from ml.annotation_translation import (
        candidate_translation,
        requirement_translation,
    )
    from ml.candidate_judgments import (
        CANDIDATE_SUPPORT_LABELS,
        CandidateJudgmentError,
        POSITIVE_SUPPORT_LABELS,
        UNLABELED_CANDIDATE_LABEL,
        legacy_candidate_labels,
        validate_complete_candidate_labels,
    )
except ModuleNotFoundError:
    from annotation import append_event  # type: ignore[no-redef]
    from annotation_translation import (  # type: ignore[no-redef]
        candidate_translation,
        requirement_translation,
    )
    from candidate_judgments import (  # type: ignore[no-redef]
        CANDIDATE_SUPPORT_LABELS,
        CandidateJudgmentError,
        POSITIVE_SUPPORT_LABELS,
        UNLABELED_CANDIDATE_LABEL,
        legacy_candidate_labels,
        validate_complete_candidate_labels,
    )


NONE_CANDIDATE = "__none__"


def candidate_label(
    task: dict[str, Any],
    candidate_id: str,
    translation: dict[str, Any] | None = None,
    *,
    show_translation: bool = False,
) -> str:
    """Render one readable evidence choice without its model score."""
    if candidate_id == NONE_CANDIDATE:
        label = "None — no candidate supports this requirement"
        return (
            f"{label}\n\n中文辅助：以上候选均不支持该要求"
            if show_translation
            else label
        )
    candidates = list(task["candidates"])
    index, candidate = next(
        (index, item)
        for index, item in enumerate(candidates)
        if item["candidate_id"] == candidate_id
    )
    label = f"{chr(65 + index)} — {candidate['evidence']}"
    translated = candidate_translation(translation, candidate_id)
    if show_translation and translated:
        return f"{label}\n\n中文辅助：{translated}"
    return label


def save_legacy_label(
    task: dict[str, Any],
    selected_candidate_id: str,
    support_label: str,
    cover_letter_safe: bool,
    note: str,
    *,
    events_path: Path,
) -> str | None:
    """Keep the old task-level save contract available for compatibility."""
    selected = (
        None if selected_candidate_id == NONE_CANDIDATE else selected_candidate_id
    )
    if support_label in POSITIVE_SUPPORT_LABELS and selected is None:
        return "Choose the best supporting evidence before using Direct or Partial."
    if support_label == "No Support":
        selected = None
        cover_letter_safe = False
    append_event(
        str(task["task_id"]),
        "label",
        events_path=events_path,
        selected_candidate_id=selected,
        support_label=support_label,
        cover_letter_safe=(
            cover_letter_safe if support_label in POSITIVE_SUPPORT_LABELS else None
        ),
        note=note,
    )
    return None


def save_complete_label(
    task: dict[str, Any],
    candidate_labels: dict[str, str],
    selected_candidate_id: str,
    cover_letter_safe: bool,
    note: str,
    *,
    events_path: Path,
) -> str | None:
    """Save complete candidate judgments and their derived task decision."""
    selected = (
        None if selected_candidate_id == NONE_CANDIDATE else selected_candidate_id
    )
    try:
        support_label, normalized = validate_complete_candidate_labels(
            task,
            candidate_labels,
            selected,
        )
    except CandidateJudgmentError as error:
        return str(error)
    append_event(
        str(task["task_id"]),
        "label",
        events_path=events_path,
        selected_candidate_id=selected,
        support_label=support_label,
        candidate_labels=normalized,
        cover_letter_safe=(
            cover_letter_safe if support_label in POSITIVE_SUPPORT_LABELS else None
        ),
        note=note,
    )
    return None


def _candidate_controls(
    task: dict[str, Any],
    state: dict[str, Any] | None,
    translation: dict[str, Any] | None,
    *,
    show_translation: bool,
    ui: Any,
) -> dict[str, str]:
    defaults = legacy_candidate_labels(task, state)
    options = [UNLABELED_CANDIDATE_LABEL, *CANDIDATE_SUPPORT_LABELS]
    labels: dict[str, str] = {}
    for index, candidate in enumerate(task["candidates"]):
        candidate_id = str(candidate["candidate_id"])
        ui.markdown(f"**{chr(65 + index)} — {candidate['evidence']}**")
        translated = candidate_translation(translation, candidate_id)
        if show_translation and translated:
            ui.caption(f"中文辅助：{translated}")
        default = defaults.get(candidate_id, UNLABELED_CANDIDATE_LABEL)
        labels[candidate_id] = ui.radio(
            f"Support for candidate {chr(65 + index)}",
            options,
            index=options.index(default),
            horizontal=True,
            key=f"candidate_support_{task['task_id']}_{candidate_id}",
        )
    return labels


def render_task(
    task: dict[str, Any],
    state: dict[str, Any] | None,
    *,
    events_path: Path,
    translation: dict[str, Any] | None = None,
    show_translation: bool = False,
    ui: Any = st,
) -> None:
    """Render complete candidate judgments and one strongest-evidence choice."""
    task_id = str(task["task_id"])
    ui.caption(f"Role family: {task['role_family']}")
    ui.markdown("### Requirement")
    ui.info(str(task["requirement"]))
    translated_requirement = requirement_translation(translation)
    if show_translation and translated_requirement:
        ui.caption(f"中文辅助翻译：{translated_requirement}")
        ui.caption("请以英文原文作为最终裁决依据。")
    ui.markdown("### Label every candidate")
    ui.caption(
        "Direct = clearly proves the requirement; Partial = relevant but incomplete; "
        "No Support = does not prove it."
    )
    candidate_labels = _candidate_controls(
        task,
        state,
        translation,
        show_translation=show_translation,
        ui=ui,
    )

    ui.markdown("### Best resume evidence")
    supported = [
        candidate_id
        for candidate_id, label in candidate_labels.items()
        if label in POSITIVE_SUPPORT_LABELS
    ]
    options = [*supported, NONE_CANDIDATE]
    default = (
        str(state.get("selected_candidate_id") or NONE_CANDIDATE)
        if state
        else NONE_CANDIDATE
    )
    if default not in options:
        default = NONE_CANDIDATE
    selected = ui.radio(
        "Select the strongest supported candidate",
        options,
        index=options.index(default),
        format_func=lambda candidate_id: candidate_label(
            task,
            candidate_id,
            translation,
            show_translation=show_translation,
        ),
        key=f"best_candidate_{task_id}",
        label_visibility="collapsed",
    )
    cover_letter_safe = ui.checkbox(
        "Safe to quote or paraphrase in a cover letter",
        value=bool(state and state.get("cover_letter_safe")),
        key=f"safe_{task_id}",
    )
    with ui.expander("Optional note", expanded=False):
        note = ui.text_area(
            "Reason or ambiguity",
            value=str(state.get("note", "")) if state else "",
            key=f"note_{task_id}",
        )

    save_column, uncertain_column = ui.columns(2)
    with save_column:
        if ui.button(
            "Save complete labels",
            key=f"save_complete_{task_id}",
            type="primary",
            width="stretch",
        ):
            error = save_complete_label(
                task,
                candidate_labels,
                selected,
                cover_letter_safe,
                note,
                events_path=events_path,
            )
            ui.error(error) if error else ui.rerun()
    with uncertain_column:
        if ui.button(
            "Mark uncertain",
            key=f"label_uncertain_{task_id}",
            width="stretch",
        ):
            append_event(
                task_id,
                "label",
                events_path=events_path,
                selected_candidate_id=None,
                support_label="Uncertain",
                note=note,
            )
            ui.rerun()

    footer_left, footer_middle, footer_right = ui.columns(3)
    with footer_left:
        if ui.button("Skip", key=f"skip_{task_id}", width="stretch"):
            append_event(task_id, "skip", events_path=events_path)
            ui.rerun()
    with footer_middle:
        if state and ui.button(
            "Clear label",
            key=f"clear_{task_id}",
            width="stretch",
        ):
            append_event(task_id, "clear", events_path=events_path)
            ui.rerun()
    with footer_right:
        ui.caption(
            "Candidate order is randomized; retrieval rank and similarity stay hidden."
        )
