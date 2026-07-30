"""Compare completed reviews and create a content-only adjudication queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import append_event, latest_task_states, load_jsonl, write_queue  # noqa: E402
from ml.real_review import (  # noqa: E402
    build_adjudication_queue,
    build_adjudication_queue_with_audit,
    reviewer_packet_manifest,
)


SPLIT_CONFIG = {
    "validation": ("real_validation", "v1"),
    "reserve": ("real_reserve", "v1"),
    "reserve-v2": ("real_reserve", "v2"),
    "reserve-v4": ("real_reserve", "v4"),
    "development": ("real_development", "v2"),
    "development-v3": ("real_development", "v3"),
    "acceptance": ("acceptance_supplement", "v1"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument(
        "--split",
        choices=tuple(SPLIT_CONFIG),
        required=True,
    )
    parser.add_argument(
        "--agreement-audit-fraction",
        type=float,
        default=0.0,
        help="Deterministically add this fraction of exact agreements for audit.",
    )
    parser.add_argument(
        "--prefer-reviewer-a",
        action="store_true",
        help=(
            "Use Reviewer A's completed decision as the final human adjudication "
            "for every A/B disagreement."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = args.annotation_dir
    prefix, version = SPLIT_CONFIG[args.split]
    a_queue = load_jsonl(base / f"{prefix}_reviewer_a_queue_{version}.jsonl")
    b_queue = load_jsonl(base / f"{prefix}_reviewer_b_queue_{version}.jsonl")
    reviewer_packet_manifest(a_queue, {"reviewer_b": b_queue})
    queue_path = base / f"{prefix}_adjudication_queue_{version}.jsonl"
    events_path = base / f"{prefix}_adjudication_decisions_{version}.jsonl"
    report_path = base / f"{prefix}_agreement_report_{version}.json"
    if queue_path.exists() or report_path.exists() or (
        args.prefer_reviewer_a and events_path.exists()
    ):
        raise SystemExit(
            f"{args.split.title()} adjudication packet exists; refusing overwrite."
        )
    reviewer_a_events = load_jsonl(
        base / f"{prefix}_reviewer_a_decisions_{version}.jsonl"
    )
    reviewer_b_events = load_jsonl(
        base / f"{prefix}_reviewer_b_decisions_{version}.jsonl"
    )
    if args.agreement_audit_fraction:
        disagreements, report = build_adjudication_queue_with_audit(
            a_queue,
            reviewer_a_events,
            reviewer_b_events,
            agreement_audit_fraction=args.agreement_audit_fraction,
            random_state=20260729,
        )
    else:
        disagreements, report = build_adjudication_queue(
            a_queue,
            reviewer_a_events,
            reviewer_b_events,
        )
    report["split"] = args.split
    report["adjudication_policy"] = (
        "reviewer_a_is_final_human_authority"
        if args.prefer_reviewer_a
        else "independent_human_adjudication"
    )
    if args.prefer_reviewer_a:
        report["status"] = "ready_for_gold_reviewer_a_authority"
    write_queue(disagreements, queue_path)
    if args.prefer_reviewer_a:
        reviewer_a_states = latest_task_states(reviewer_a_events)
        for task in disagreements:
            state = reviewer_a_states[str(task["task_id"])]
            append_event(
                str(task["task_id"]),
                "label",
                events_path=events_path,
                selected_candidate_id=state.get("selected_candidate_id"),
                support_label=str(state["support_label"]),
                cover_letter_safe=state.get("cover_letter_safe"),
                note=(
                    "Reviewer A was designated as the final human authority "
                    "for A/B disagreements."
                ),
            )
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Exact agreements: {report['exact_agreements']}/{report['tasks']}")
    print(f"Label agreements: {report['label_agreements']}/{report['tasks']}")
    print(f"Reviewer disagreements: {report['adjudication_tasks']}")
    print(f"Human review tasks: {report.get('human_review_tasks', report['adjudication_tasks'])}")
    print(f"Queue: {queue_path}")
    if args.prefer_reviewer_a:
        print(f"Reviewer A adjudications: {events_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
