"""Build one balanced batch for completing candidate-level training labels."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "candidate_completion_v1"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import append_event, load_jsonl, write_queue  # noqa: E402
from ml.candidate_completion import (  # noqa: E402
    tasks_requiring_candidate_completion,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.batch_index < 1:
        raise SystemExit("Batch size and index must be positive.")
    tasks = load_jsonl(args.dataset_dir / "annotated_tasks.jsonl")
    pairs = load_jsonl(args.dataset_dir / "training_pairs.jsonl")
    ordered = tasks_requiring_candidate_completion(
        tasks,
        pairs,
        random_state=args.random_state,
    )
    start = (args.batch_index - 1) * args.batch_size
    selected = ordered[start : start + args.batch_size]
    if not selected:
        raise SystemExit("The requested completion batch is empty.")

    suffix = f"v1_batch{args.batch_index}"
    queue_path = args.output_dir / f"candidate_completion_queue_{suffix}.jsonl"
    events_path = args.output_dir / f"candidate_completion_events_{suffix}.jsonl"
    manifest_path = args.output_dir / f"candidate_completion_manifest_{suffix}.json"
    if queue_path.exists() or events_path.exists() or manifest_path.exists():
        raise SystemExit("Candidate-completion batch exists; refusing overwrite.")

    write_queue([row["queue_task"] for row in selected], queue_path)
    for row in selected:
        task = row["queue_task"]
        append_event(
            str(task["task_id"]),
            "label",
            events_path=events_path,
            selected_candidate_id=row["selected_candidate_id"],
            support_label=str(row["support_label"]),
            candidate_labels=None,
            cover_letter_safe=row["cover_letter_safe"],
            note="Existing task-level review; candidate judgments still required.",
        )
    manifest = {
        "schema_version": 1,
        "dataset_role": "training_candidate_label_completion",
        "batch_index": args.batch_index,
        "batch_size": len(selected),
        "eligible_tasks": len(ordered),
        "offset": start,
        "random_state": args.random_state,
        "task_label_counts": dict(
            Counter(str(row["support_label"]) for row in selected)
        ),
        "role_counts": dict(
            Counter(str(row["queue_task"]["role_family"]) for row in selected)
        ),
        "evaluation_use": "prohibited",
        "training_use": "allowed_after_complete_human_review",
        "labels_exposed_in_queue": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Candidate-completion tasks: {len(selected)}")
    print(f"Eligible incomplete tasks: {len(ordered)}")
    print(f"Queue: {queue_path}")
    print(f"Events: {events_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
