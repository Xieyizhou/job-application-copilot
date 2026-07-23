"""Report aggregate bias checks for local requirement/evidence labels."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_QUEUE_PATH = PROJECT_ROOT / "data" / "ml" / "annotations" / "pilot_queue_v3.jsonl"
DEFAULT_EVENTS_PATH = PROJECT_ROOT / "data" / "ml" / "annotations" / "pilot_annotations_v3.jsonl"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, load_queue  # noqa: E402
from ml.annotation_audit import audit_annotations  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_queue(args.queue_path)
    states = latest_task_states(load_jsonl(args.events_path))
    report = audit_annotations(tasks, states)
    print("Annotation Bias Audit")
    print("=====================")
    print(f"Progress: {report['completed']}/{report['total']}")
    print(f"Unique tasks: {report['unique_tasks']}")
    agreement = report["repeat_agreement"]
    agreement_text = "not available" if agreement is None else f"{agreement:.1%}"
    print(f"Repeat agreement: {agreement_text}")
    print(f"Conflicting repeat pairs: {report['conflicting_repeat_pairs']}")
    print(f"Labels: {report['label_counts']}")
    print(f"Selected positions: {report['selected_position_counts']}")
    print(f"Largest selected-position share: {report['position_max_share']:.1%}")
    print(f"Largest label share: {report['label_max_share']:.1%}")
    print(f"Largest repeated requirement prefix: {report['repeated_requirement_prefix_share']:.1%}")
    print(f"Forbidden queue fields: {report['forbidden_queue_fields'] or 'none'}")
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"- {warning}")
    else:
        print("Warnings: none")


if __name__ == "__main__":
    main()
