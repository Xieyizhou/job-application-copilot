"""Create a blinded human-adjudication queue for model-only reserve disagreements.

This is a diagnostic workflow only.  It never promotes model consensus to gold
data; a human decision is required for every task in the resulting queue.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reserve_v3"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.real_review import build_adjudication_queue, reviewer_packet_manifest  # noqa: E402


def main() -> None:
    source_path = DEFAULT_DIR / "real_reserve_queue_v3.jsonl"
    reviewer_a_queue_path = DEFAULT_DIR / "real_reserve_reviewer_a_queue_v3.jsonl"
    reviewer_b_queue_path = DEFAULT_DIR / "real_reserve_reviewer_b_queue_v3.jsonl"
    reviewer_a_events_path = (
        DEFAULT_DIR / "real_reserve_model_reviewer_a_decisions_v3.jsonl"
    )
    reviewer_b_events_path = (
        DEFAULT_DIR / "real_reserve_model_reviewer_b_decisions_v3.jsonl"
    )
    queue_path = (
        DEFAULT_DIR / "real_reserve_model_diagnostic_adjudication_queue_v3.jsonl"
    )
    report_path = (
        DEFAULT_DIR / "real_reserve_model_diagnostic_adjudication_report_v3.json"
    )
    if queue_path.exists() or report_path.exists():
        raise SystemExit("Model-only diagnostic adjudication packet exists; refusing overwrite.")

    source = load_jsonl(source_path)
    reviewer_packet_manifest(
        source,
        {
            "reviewer_a": load_jsonl(reviewer_a_queue_path),
            "reviewer_b": load_jsonl(reviewer_b_queue_path),
        },
    )
    disagreements, report = build_adjudication_queue(
        source,
        load_jsonl(reviewer_a_events_path),
        load_jsonl(reviewer_b_events_path),
    )
    report.update(
        {
            "split": "reserve-v3-model-diagnostic",
            "reviewer_mode": "two_independent_model_only_diagnostics",
            "adjudication_policy": "independent_human_adjudication",
            "human_labels_required": True,
            "training_use": "prohibited_until_human_adjudication",
            "promotion_eligible": False,
            "labels_or_predictions_exposed_in_queue": False,
        }
    )
    write_queue(disagreements, queue_path)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Human adjudication tasks: {report['adjudication_tasks']}")
    print(f"Queue: {queue_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
