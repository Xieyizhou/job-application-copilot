"""Queue and append-only event storage for local evidence annotation."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Iterable
from uuid import uuid4


SCHEMA_VERSION = 1
SUPPORT_LABELS = ("Direct", "Partial", "No Support", "Uncertain")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ANNOTATION_DIR = PROJECT_ROOT / "data" / "ml" / "annotations"
DEFAULT_QUEUE_PATH = DEFAULT_ANNOTATION_DIR / "pilot_queue_v3.jsonl"
DEFAULT_EVENTS_PATH = DEFAULT_ANNOTATION_DIR / "pilot_annotations_v3.jsonl"


class AnnotationDataError(ValueError):
    """Raised when local annotation data violates the queue contract."""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load JSON objects from a local JSONL file."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise AnnotationDataError(f"Invalid JSONL at line {line_number}: {error}") from error
        if not isinstance(row, dict):
            raise AnnotationDataError(f"Annotation row {line_number} must be an object.")
        rows.append(row)
    return rows


def validate_queue(tasks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Validate task identity, labels, and candidate uniqueness."""
    checked = list(tasks)
    task_ids = [str(task.get("task_id", "")) for task in checked]
    if any(not task_id for task_id in task_ids) or len(task_ids) != len(set(task_ids)):
        raise AnnotationDataError("Queue task_id values must be present and unique.")
    for task in checked:
        if int(task.get("schema_version", 0)) != SCHEMA_VERSION:
            raise AnnotationDataError(f"Unsupported queue schema for {task['task_id']}.")
        if not str(task.get("requirement", "")).strip():
            raise AnnotationDataError(f"Task {task['task_id']} has no requirement.")
        candidates = task.get("candidates")
        if not isinstance(candidates, list) or len(candidates) < 2:
            raise AnnotationDataError(f"Task {task['task_id']} needs at least two candidates.")
        candidate_ids = [str(candidate.get("candidate_id", "")) for candidate in candidates]
        if any(not candidate_id for candidate_id in candidate_ids):
            raise AnnotationDataError(f"Task {task['task_id']} has a candidate without an id.")
        if len(candidate_ids) != len(set(candidate_ids)):
            raise AnnotationDataError(f"Task {task['task_id']} has duplicate candidate ids.")
    return checked


def load_queue(path: Path = DEFAULT_QUEUE_PATH) -> list[dict[str, Any]]:
    """Load and validate a local annotation queue."""
    return validate_queue(load_jsonl(path))


def write_queue(tasks: Iterable[dict[str, Any]], path: Path = DEFAULT_QUEUE_PATH) -> None:
    """Write a deterministic local queue after validating it."""
    checked = validate_queue(tasks)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(json.dumps(task, sort_keys=True) + "\n" for task in checked)
    path.write_text(payload, encoding="utf-8")


def append_event(
    task_id: str,
    action: str,
    *,
    events_path: Path = DEFAULT_EVENTS_PATH,
    selected_candidate_id: str | None = None,
    support_label: str | None = None,
    cover_letter_safe: bool | None = None,
    note: str = "",
) -> dict[str, Any]:
    """Append one label, skip, or clear event and return it."""
    if action not in {"label", "skip", "clear"}:
        raise AnnotationDataError(f"Unsupported annotation action: {action}")
    if action == "label" and support_label not in SUPPORT_LABELS:
        raise AnnotationDataError(f"Unsupported support label: {support_label}")
    event = {
        "schema_version": SCHEMA_VERSION,
        "event_id": uuid4().hex,
        "task_id": task_id,
        "action": action,
        "selected_candidate_id": selected_candidate_id,
        "support_label": support_label,
        "cover_letter_safe": cover_letter_safe,
        "note": note.strip(),
        "annotated_at": datetime.now(timezone.utc).isoformat(),
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
    return event


def latest_task_states(events: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Replay append-only events into the latest state for each task."""
    states: dict[str, dict[str, Any]] = {}
    for event in events:
        task_id = str(event.get("task_id", ""))
        if not task_id:
            continue
        if event.get("action") == "clear":
            states.pop(task_id, None)
        elif event.get("action") in {"label", "skip"}:
            states[task_id] = dict(event)
    return states


def annotation_summary(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Return progress, label counts, and blind-repeat consistency."""
    completed = [task for task in tasks if task["task_id"] in states]
    labels = Counter(
        str(states[task["task_id"]].get("support_label"))
        for task in completed
        if states[task["task_id"]].get("action") == "label"
    )
    selected_positions: Counter[str] = Counter()
    for task in completed:
        state = states[task["task_id"]]
        if state.get("action") != "label":
            continue
        selected_id = state.get("selected_candidate_id")
        position = next(
            (
                chr(65 + index)
                for index, candidate in enumerate(task["candidates"])
                if candidate["candidate_id"] == selected_id
            ),
            "None",
        )
        selected_positions[position] += 1
    repeat_results: list[bool] = []
    for task in tasks:
        original_id = str(task.get("blind_duplicate_of", ""))
        if not original_id or task["task_id"] not in states or original_id not in states:
            continue
        current = states[task["task_id"]]
        original = states[original_id]
        repeat_results.append(
            current.get("support_label") == original.get("support_label")
            and current.get("selected_candidate_id") == original.get("selected_candidate_id")
        )
    return {
        "total": len(tasks),
        "completed": len(completed),
        "remaining": len(tasks) - len(completed),
        "label_counts": dict(labels),
        "selected_position_counts": dict(selected_positions),
        "repeat_pairs": len(repeat_results),
        "repeat_agreement": (
            sum(repeat_results) / len(repeat_results) if repeat_results else None
        ),
    }


def repeat_conflict_task_ids(
    tasks: list[dict[str, Any]],
    states: dict[str, dict[str, Any]],
) -> set[str]:
    """Return both task ids for completed blind repeats that disagree."""
    conflicts: set[str] = set()
    for task in tasks:
        task_id = str(task["task_id"])
        original_id = str(task.get("blind_duplicate_of", ""))
        if not original_id or task_id not in states or original_id not in states:
            continue
        current = states[task_id]
        original = states[original_id]
        agrees = (
            current.get("support_label") == original.get("support_label")
            and current.get("selected_candidate_id") == original.get("selected_candidate_id")
        )
        if not agrees:
            conflicts.update((original_id, task_id))
    return conflicts
