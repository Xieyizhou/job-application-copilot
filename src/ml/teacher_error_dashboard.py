"""Standalone local UI for diagnosing teacher ranking errors."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Any

import streamlit as st

try:
    from ml.annotation import load_jsonl
except ModuleNotFoundError:
    from annotation import load_jsonl  # type: ignore[no-redef]


DEFAULT_DIR = Path("data/ml/distillation/teacher_error_audit_v1")
QUEUE_PATH = Path(
    os.getenv("JOB_COPILOT_TEACHER_AUDIT_QUEUE", str(DEFAULT_DIR / "queue.jsonl"))
)
EVENTS_PATH = Path(
    os.getenv("JOB_COPILOT_TEACHER_AUDIT_EVENTS", str(DEFAULT_DIR / "decisions.jsonl"))
)
ERROR_TYPES = (
    "Teacher ranking failure",
    "Lexical or role-language misdirection",
    "Compound or partial-support ambiguity",
    "Human-label ambiguity",
    "Evidence extraction problem",
    "Other",
)


def latest_decisions(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the latest append-only decision for each audit item."""
    return {str(row["audit_id"]): row for row in rows if row.get("audit_id")}


def save_decision(
    task: dict[str, Any],
    *,
    error_type: str,
    note: str,
    events_path: Path = EVENTS_PATH,
) -> None:
    """Append one diagnostic decision without modifying human labels."""
    if error_type not in ERROR_TYPES:
        raise ValueError("Unknown teacher-error category.")
    event = {
        "schema_version": 1,
        "audit_id": str(task["audit_id"]),
        "task_id": str(task["task_id"]),
        "error_type": error_type,
        "note": note.strip(),
        "reviewed_at": datetime.now(timezone.utc).isoformat(),
        "diagnostic_only": True,
        "human_gold_changed": False,
        "model_selection_allowed": False,
    }
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, sort_keys=True) + "\n")
    events_path.chmod(0o600)


def main() -> None:
    """Render the local teacher-error audit queue."""
    st.set_page_config(page_title="Teacher Error Audit", layout="wide")
    st.title("Teacher Error Audit")
    st.caption(
        "Diagnostic review only. Decisions do not change human gold or authorize model use."
    )
    if not QUEUE_PATH.is_file():
        st.warning("No local teacher-error queue was found.")
        return
    tasks = load_jsonl(QUEUE_PATH)
    decisions = latest_decisions(load_jsonl(EVENTS_PATH))
    remaining = [task for task in tasks if str(task["audit_id"]) not in decisions]
    st.progress((len(tasks) - len(remaining)) / len(tasks) if tasks else 0.0)
    st.caption(f"{len(tasks) - len(remaining)} completed · {len(remaining)} remaining")
    view = st.radio("View", ["Remaining", "All"], horizontal=True)
    visible = remaining if view == "Remaining" else tasks
    if not visible:
        st.success("Teacher-error audit is complete.")
        return

    index = st.number_input(
        "Audit item",
        min_value=1,
        max_value=len(visible),
        value=1,
        step=1,
    )
    task = visible[int(index) - 1]
    st.subheader(f"{task['role_family']} · {', '.join(task['audit_reasons'])}")
    st.markdown("**Requirement**")
    st.write(task["requirement"])
    st.markdown("**Candidates**")
    for candidate in task["candidates"]:
        with st.container(border=True):
            st.write(candidate["evidence"])
            left, middle, right = st.columns(3)
            left.caption(f"Human: {candidate['human_label']}")
            middle.caption(f"Qwen: {candidate['reranker_score']:.4f}")
            right.caption(f"Embedding: {candidate['embedding_score']:.4f}")

    previous = decisions.get(str(task["audit_id"]))
    default_type = previous.get("error_type") if previous else ERROR_TYPES[0]
    error_type = st.radio(
        "Primary diagnosis",
        ERROR_TYPES,
        index=ERROR_TYPES.index(default_type),
    )
    note = st.text_area("Optional note", value=previous.get("note", "") if previous else "")
    if st.button("Save diagnosis", type="primary"):
        save_decision(task, error_type=error_type, note=note)
        st.rerun()


if __name__ == "__main__":
    main()
