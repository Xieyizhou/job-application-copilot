"""Validate completed human adjudications and export conservative gold records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
PIPELINE_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "pipeline_v1"
DEFAULT_QUEUE_PATH = PIPELINE_DIR / "adjudication" / "queue.jsonl"
DEFAULT_EVENTS_PATH = PIPELINE_DIR / "adjudication" / "human_decisions.jsonl"
DEFAULT_CASES_PATH = PIPELINE_DIR / "proposals" / "merged.jsonl"
DEFAULT_OUTPUT_PATH = PIPELINE_DIR / "gold" / "human_adjudication_gold.jsonl"
DEFAULT_REPORT_PATH = PIPELINE_DIR / "manifests" / "human_adjudication_report.json"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, load_queue  # noqa: E402
from ml.annotation_dataset import write_jsonl  # noqa: E402
from ml.human_adjudication import (  # noqa: E402
    HumanAdjudicationError,
    build_human_adjudication_gold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-path", type=Path, default=DEFAULT_QUEUE_PATH)
    parser.add_argument("--events-path", type=Path, default=DEFAULT_EVENTS_PATH)
    parser.add_argument("--cases-path", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Export only fully resolved tasks instead of requiring the whole queue.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue = load_queue(args.queue_path)
    events = load_jsonl(args.events_path)
    cases = load_jsonl(args.cases_path)
    if not events:
        raise SystemExit(f"No adjudication decisions found: {args.events_path}")
    if not cases:
        raise SystemExit(f"No source cases found: {args.cases_path}")
    try:
        gold_tasks, report = build_human_adjudication_gold(
            queue,
            events,
            cases,
            require_complete=not args.allow_partial,
        )
    except HumanAdjudicationError as error:
        raise SystemExit(f"Human adjudication export blocked: {error}") from error
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(gold_tasks, args.output_path)
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Accepted human adjudication gold tasks: {len(gold_tasks)}")
    print(f"Gold: {args.output_path}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
