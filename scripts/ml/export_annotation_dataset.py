"""Export reviewed local annotations into leakage-aware training tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "data" / "ml" / "annotations" / "pilot_queue_v2.jsonl"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "ml" / "annotations" / "pilot_annotations_v2.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_v2"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, load_queue  # noqa: E402
from ml.annotation_dataset import (  # noqa: E402
    build_annotated_tasks,
    build_training_pairs,
    dataset_manifest,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export only completed, resolved tasks without requiring the full queue.",
    )
    parser.add_argument("--dataset-name", default="reviewed_evidence_pilot")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_queue(args.queue_path)
    states = latest_task_states(load_jsonl(args.events_path))
    annotated = build_annotated_tasks(
        tasks,
        states,
        random_state=args.random_state,
        require_complete=not args.allow_partial,
    )
    pairs = build_training_pairs(annotated)
    manifest = dataset_manifest(
        annotated,
        pairs,
        dataset_name=args.dataset_name,
        source_queue_complete=not args.allow_partial,
    )
    write_jsonl(annotated, args.output_dir / "annotated_tasks.jsonl")
    write_jsonl(pairs, args.output_dir / "training_pairs.jsonl")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved {len(annotated)} unique annotated tasks.")
    print(f"Saved {len(pairs)} high-confidence training pairs.")
    print(f"Template groups: {manifest['template_groups']}")
    print(f"Task labels: {manifest['task_label_counts']}")
    print(f"Pair labels: {manifest['pair_label_counts']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
