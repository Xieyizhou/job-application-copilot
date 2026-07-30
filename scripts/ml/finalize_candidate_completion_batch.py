"""Merge one completed candidate-label batch into a new training corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_BASE_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
DEFAULT_ANNOTATION_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "candidate_completion_v1"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v4_batch1"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, load_queue  # noqa: E402
from ml.annotation_dataset import (  # noqa: E402
    build_annotated_tasks,
    build_training_pairs,
    write_jsonl,
)
from ml.candidate_completion import merge_completed_candidate_labels  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=DEFAULT_BASE_DIR)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_ANNOTATION_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-index", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=20260728)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suffix = f"v1_batch{args.batch_index}"
    queue_path = args.annotation_dir / f"candidate_completion_queue_{suffix}.jsonl"
    events_path = args.annotation_dir / f"candidate_completion_events_{suffix}.jsonl"
    if args.output_dir.exists():
        raise SystemExit("Candidate-complete output directory exists; refusing overwrite.")
    queue = load_queue(queue_path)
    states = latest_task_states(load_jsonl(events_path))
    completed_tasks = build_annotated_tasks(
        queue,
        states,
        random_state=args.random_state,
        require_complete=True,
        require_candidate_labels=True,
    )
    completed_pairs = build_training_pairs(completed_tasks)
    tasks, pairs, manifest = merge_completed_candidate_labels(
        load_jsonl(args.base_dir / "annotated_tasks.jsonl"),
        load_jsonl(args.base_dir / "training_pairs.jsonl"),
        completed_tasks,
        completed_pairs,
    )
    write_jsonl(tasks, args.output_dir / "annotated_tasks.jsonl")
    write_jsonl(pairs, args.output_dir / "training_pairs.jsonl")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Merged candidate-complete tasks: {manifest['completed_task_count']}")
    print(f"Combined tasks: {manifest['task_count']}")
    print(f"Combined pairs: {manifest['pair_count']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
