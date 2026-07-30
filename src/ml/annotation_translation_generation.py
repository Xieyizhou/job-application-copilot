"""Create source-verified annotation translation sidecar rows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from ml.annotation import validate_queue
from ml.annotation_translation import index_translation_rows, verified_task_translation


class TranslationGenerationError(ValueError):
    """Raised when an external translation response is not usable."""


def build_translation_sidecar(
    tasks: Iterable[dict[str, Any]],
    *,
    translate: Callable[[str], str],
) -> list[dict[str, Any]]:
    """Translate queue text without changing any annotation-source fields."""
    rows: list[dict[str, Any]] = []
    for task in validate_queue(tasks):
        requirement = str(task["requirement"])
        translated_requirement = _translated(requirement, translate)
        candidates = []
        for candidate in task["candidates"]:
            source = str(candidate["evidence"])
            candidates.append(
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "source": source,
                    "zh": _translated(source, translate),
                }
            )
        rows.append(
            {
                "task_id": str(task["task_id"]),
                "requirement": {
                    "source": requirement,
                    "zh": translated_requirement,
                },
                "candidates": candidates,
            }
        )
    indexed = index_translation_rows(rows)
    for task in validate_queue(tasks):
        if verified_task_translation(task, indexed) is None:
            raise TranslationGenerationError(
                f"Translation source verification failed for {task['task_id']}."
            )
    return rows


def _translated(source: str, translate: Callable[[str], str]) -> str:
    translation = str(translate(source)).strip()
    if not translation:
        raise TranslationGenerationError("Translation response is empty.")
    return translation
