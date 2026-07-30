"""Merge reviewer consensus and blinded adjudications into final batch events."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import (  # noqa: E402
    append_event,
    latest_task_states,
    load_jsonl,
    load_queue,
)
from ml.candidate_judgments import (  # noqa: E402
    has_complete_candidate_labels,
    validate_complete_candidate_labels,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--reviewer-a", type=Path, required=True)
    parser.add_argument("--reviewer-b", type=Path, required=True)
    parser.add_argument("--adjudication-events", type=Path, required=True)
    parser.add_argument("--adjudication-queue", type=Path)
    parser.add_argument("--output-events", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def signature(event: dict[str, Any]) -> tuple[Any, ...]:
    """Return all fields that affect candidate-completion gold."""
    return (
        event.get("support_label"),
        event.get("selected_candidate_id"),
        bool(event.get("cover_letter_safe")),
        tuple(sorted(dict(event.get("candidate_labels", {})).items())),
    )


def checked_states(
    path: Path,
    tasks: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Load one complete, internally consistent decision per expected task."""
    states = latest_task_states(load_jsonl(path))
    if set(states) != set(tasks):
        raise SystemExit(f"{path.name} must cover exactly the expected tasks.")
    for task_id, event in states.items():
        derived, _ = validate_complete_candidate_labels(
            tasks[task_id],
            dict(event.get("candidate_labels", {})),
            event.get("selected_candidate_id"),
        )
        if derived != event.get("support_label"):
            raise SystemExit(f"Inconsistent task label in {path.name}: {task_id}")
    return states


def main() -> None:
    args = parse_args()
    queue = load_queue(args.queue)
    tasks = {str(task["task_id"]): task for task in queue}
    reviewer_a = checked_states(args.reviewer_a, tasks)
    reviewer_b = checked_states(args.reviewer_b, tasks)
    disagreements = {
        task_id
        for task_id in tasks
        if signature(reviewer_a[task_id]) != signature(reviewer_b[task_id])
    }
    all_adjudication_states = latest_task_states(
        load_jsonl(args.adjudication_events)
    )
    if args.adjudication_queue:
        allowed_adjudications = {
            str(task["task_id"]) for task in load_queue(args.adjudication_queue)
        }
        if set(all_adjudication_states) != allowed_adjudications:
            raise SystemExit(
                "Combined adjudication events must exactly cover the declared "
                "adjudication queue."
            )
        adjudication_states = {
            task_id: event
            for task_id, event in all_adjudication_states.items()
            if task_id in disagreements
        }
    else:
        adjudication_states = all_adjudication_states
    if set(adjudication_states) != disagreements:
        missing = sorted(disagreements - set(adjudication_states))
        extra = sorted(set(adjudication_states) - disagreements)
        raise SystemExit(
            f"Adjudications must exactly match disagreements; missing={missing}, "
            f"extra={extra}."
        )
    for task_id in disagreements:
        event = adjudication_states[task_id]
        derived, _ = validate_complete_candidate_labels(
            tasks[task_id],
            dict(event.get("candidate_labels", {})),
            event.get("selected_candidate_id"),
        )
        if derived != event.get("support_label"):
            raise SystemExit(f"Inconsistent adjudication task label: {task_id}")

    existing = latest_task_states(load_jsonl(args.output_events))
    sources: Counter[str] = Counter()
    final_labels: Counter[str] = Counter()
    for task_id in tasks:
        if task_id in disagreements:
            chosen = adjudication_states[task_id]
            source = "human_adjudication"
        elif has_complete_candidate_labels(tasks[task_id], existing.get(task_id)):
            chosen = existing[task_id]
            source = "existing_human_annotation"
        else:
            chosen = reviewer_a[task_id]
            source = "blind_reviewer_consensus"
        derived, _ = validate_complete_candidate_labels(
            tasks[task_id],
            dict(chosen.get("candidate_labels", {})),
            chosen.get("selected_candidate_id"),
        )
        if derived != chosen.get("support_label"):
            raise SystemExit(f"Inconsistent final task label: {task_id}")
        if source == "existing_human_annotation":
            sources[source] += 1
            final_labels[str(chosen["support_label"])] += 1
            continue
        append_event(
            task_id,
            "label",
            events_path=args.output_events,
            selected_candidate_id=chosen.get("selected_candidate_id"),
            support_label=str(chosen["support_label"]),
            candidate_labels=dict(chosen["candidate_labels"]),
            cover_letter_safe=bool(chosen.get("cover_letter_safe")),
            note=f"Candidate completion source: {source}.",
        )
        sources[source] += 1
        final_labels[str(chosen["support_label"])] += 1

    report = {
        "schema_version": 1,
        "tasks": len(tasks),
        "reviewer_consensus_tasks": len(tasks) - len(disagreements),
        "human_adjudication_tasks": len(disagreements),
        "source_counts": dict(sources),
        "task_label_counts": dict(final_labels),
        "output_events": str(args.output_events),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Final candidate-complete tasks: {len(tasks)}")
    print(f"Reviewer consensus: {len(tasks) - len(disagreements)}")
    print(f"Human adjudications: {len(disagreements)}")
    print(f"Task labels: {dict(final_labels)}")
    print(f"Events: {args.output_events}")
    print(f"Report: {args.report}")


if __name__ == "__main__":
    main()
