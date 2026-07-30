"""Create a blinded queue for disagreements between two candidate reviewers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "candidate_completion_v1"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.candidate_judgments import (  # noqa: E402
    validate_complete_candidate_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--output-queue", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def latest_events(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Return the final label event for every task."""
    latest: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("action") == "label":
            latest[str(row["task_id"])] = row
    return latest


def decision_signature(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return all fields that affect candidate-completion gold."""
    return (
        event.get("support_label"),
        event.get("selected_candidate_id"),
        bool(event.get("cover_letter_safe")),
        tuple(sorted(dict(event.get("candidate_labels", {})).items())),
    )


def main() -> None:
    args = parse_args()
    queue = load_jsonl(args.queue)
    queue_by_id = {str(task["task_id"]): task for task in queue}
    reviewer_a = latest_events(load_jsonl(args.reviewer_a))
    reviewer_b = latest_events(load_jsonl(args.reviewer_b))
    expected = set(queue_by_id)
    if set(reviewer_a) != expected or set(reviewer_b) != expected:
        raise SystemExit("Both reviewers must label every queue task exactly once.")

    disagreement_tasks: list[dict[str, Any]] = []
    disagreement_fields: Counter[str] = Counter()
    candidate_disagreements = 0
    for task_id, task in queue_by_id.items():
        a_event = reviewer_a[task_id]
        b_event = reviewer_b[task_id]
        for event in (a_event, b_event):
            derived_label, _ = validate_complete_candidate_labels(
                task,
                dict(event.get("candidate_labels", {})),
                event.get("selected_candidate_id"),
            )
            if derived_label != event.get("support_label"):
                raise SystemExit(
                    f"Reviewer event has inconsistent task label for {task_id}."
                )
        if decision_signature(a_event) == decision_signature(b_event):
            continue
        disagreement_tasks.append(task)
        for field in (
            "support_label",
            "selected_candidate_id",
            "cover_letter_safe",
        ):
            if a_event.get(field) != b_event.get(field):
                disagreement_fields[field] += 1
        a_labels = dict(a_event["candidate_labels"])
        b_labels = dict(b_event["candidate_labels"])
        task_candidate_disagreements = sum(
            a_labels[candidate_id] != b_labels[candidate_id]
            for candidate_id in a_labels
        )
        candidate_disagreements += task_candidate_disagreements
        if task_candidate_disagreements:
            disagreement_fields["candidate_labels"] += 1

    write_queue(disagreement_tasks, args.output_queue)
    report = {
        "schema_version": 1,
        "tasks": len(queue),
        "exact_agreements": len(queue) - len(disagreement_tasks),
        "exact_agreement_rate": (
            (len(queue) - len(disagreement_tasks)) / len(queue) if queue else 1.0
        ),
        "adjudication_tasks": len(disagreement_tasks),
        "candidate_judgment_disagreements": candidate_disagreements,
        "disagreement_field_counts": dict(disagreement_fields),
        "reviewer_a": str(args.reviewer_a),
        "reviewer_b": str(args.reviewer_b),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exact agreements: {report['exact_agreements']}/{report['tasks']}")
    print(f"Adjudication tasks: {report['adjudication_tasks']}")
    print(f"Candidate label disagreements: {candidate_disagreements}")
    print(f"Queue: {args.output_queue}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
