"""Optional source-verified translations for the local annotation interface."""

from __future__ import annotations

from typing import Any, Iterable


class AnnotationTranslationError(ValueError):
    """Raised when an annotation translation sidecar is malformed."""


def index_translation_rows(
    rows: Iterable[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Validate and index translation rows without altering annotation tasks."""
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("task_id", "")).strip()
        if not task_id:
            raise AnnotationTranslationError("Translation rows require task_id.")
        if task_id in indexed:
            raise AnnotationTranslationError(
                f"Duplicate translation task id: {task_id}"
            )
        requirement = row.get("requirement")
        candidates = row.get("candidates")
        if not isinstance(requirement, dict):
            raise AnnotationTranslationError(
                f"Translation task {task_id} requires a requirement object."
            )
        if not isinstance(candidates, list):
            raise AnnotationTranslationError(
                f"Translation task {task_id} requires candidate translations."
            )
        candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
        if any(not candidate_id for candidate_id in candidate_ids):
            raise AnnotationTranslationError(
                f"Translation task {task_id} has a candidate without an id."
            )
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AnnotationTranslationError(
                f"Translation task {task_id} has duplicate candidate ids."
            )
        indexed[task_id] = row
    return indexed


def verified_task_translation(
    task: dict[str, Any],
    indexed: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    """Return one translation only when every source string still matches."""
    translation = indexed.get(str(task.get("task_id", "")))
    if translation is None:
        return None
    requirement = translation["requirement"]
    if str(requirement.get("source", "")) != str(task.get("requirement", "")):
        return None
    translated_candidates = {
        str(candidate["candidate_id"]): candidate
        for candidate in translation["candidates"]
    }
    task_candidates = list(task.get("candidates", []))
    if set(translated_candidates) != {
        str(candidate.get("candidate_id", "")) for candidate in task_candidates
    }:
        return None
    for candidate in task_candidates:
        translated = translated_candidates[str(candidate["candidate_id"])]
        if str(translated.get("source", "")) != str(candidate.get("evidence", "")):
            return None
    return translation


def requirement_translation(translation: dict[str, Any] | None) -> str:
    """Return the optional Chinese requirement translation."""
    if translation is None:
        return ""
    requirement = translation.get("requirement")
    return str(requirement.get("zh", "")).strip() if isinstance(requirement, dict) else ""


def candidate_translation(
    translation: dict[str, Any] | None,
    candidate_id: str,
) -> str:
    """Return one optional Chinese evidence translation."""
    if translation is None:
        return ""
    for candidate in translation.get("candidates", []):
        if str(candidate.get("candidate_id", "")) == candidate_id:
            return str(candidate.get("zh", "")).strip()
    return ""
