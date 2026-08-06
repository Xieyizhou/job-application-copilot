"""Compare two completed successor reviews and create a blind adjudication queue."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_ANNOTATION_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "successor_development_v8"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import latest_task_states, load_jsonl, write_queue  # noqa: E402
from ml.operational_development_v3 import (  # noqa: E402
    build_blind_reviewer_queue,
    cohen_kappa,
)
from ml.operational_review import operational_review_agreement  # noqa: E402
from ml.real_holdout import sha256_bytes  # noqa: E402
from ml.real_review import build_adjudication_queue  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument(
        "--reviewer-a-decisions",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--reviewer-b-decisions",
        type=Path,
        required=True,
    )
    parser.add_argument("--random-state", type=int, default=20260807)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    queue_path = args.annotation_dir / "adjudication_queue.jsonl"
    report_path = args.annotation_dir / "agreement_report.json"
    if queue_path.exists() or report_path.exists():
        raise SystemExit("Adjudication packet exists; refusing overwrite.")
    source = load_jsonl(args.annotation_dir / "queue.jsonl")
    reviewer_a = load_jsonl(args.reviewer_a_decisions)
    reviewer_b = load_jsonl(args.reviewer_b_decisions)
    disagreements, report = build_adjudication_queue(
        source,
        reviewer_a,
        reviewer_b,
    )
    detailed = operational_review_agreement(
        source,
        latest_task_states(reviewer_a),
        latest_task_states(reviewer_b),
    )
    a_states = latest_task_states(reviewer_a)
    b_states = latest_task_states(reviewer_b)
    task_ids = sorted(a_states)
    candidate_left: list[str] = []
    candidate_right: list[str] = []
    for task_id in task_ids:
        candidate_ids = sorted(
            str(candidate["candidate_id"])
            for candidate in next(
                task for task in source if str(task["task_id"]) == task_id
            )["candidates"]
        )
        candidate_left.extend(
            str(a_states[task_id]["candidate_labels"][candidate_id])
            for candidate_id in candidate_ids
        )
        candidate_right.extend(
            str(b_states[task_id]["candidate_labels"][candidate_id])
            for candidate_id in candidate_ids
        )
    blind_queue = build_blind_reviewer_queue(
        disagreements,
        reviewer_id="adjudicator",
        random_state=args.random_state,
    )
    write_queue(blind_queue, queue_path)
    payload = {
        **report,
        **detailed,
        "reviewer_b_role": "independent_human_review",
        "candidate_cohen_kappa": cohen_kappa(
            candidate_left,
            candidate_right,
        ),
        "task_label_cohen_kappa": cohen_kappa(
            [str(a_states[task_id]["support_label"]) for task_id in task_ids],
            [str(b_states[task_id]["support_label"]) for task_id in task_ids],
        ),
        "candidate_kappa_gate": 0.70,
        "candidate_kappa_gate_passed": (
            cohen_kappa(candidate_left, candidate_right) >= 0.70
        ),
        "adjudication_policy": "independent_human_adjudication",
        "adjudication_queue_blind": True,
        "labels_in_adjudication_queue": False,
        "construction_metadata_in_adjudication_queue": False,
        "reviewer_a_decisions_sha256": sha256_bytes(
            args.reviewer_a_decisions.read_bytes()
        ),
        "reviewer_b_decisions_sha256": sha256_bytes(
            args.reviewer_b_decisions.read_bytes()
        ),
        "status": (
            "human_adjudication_required"
            if blind_queue
            else "ready_for_gold"
        ),
    }
    report_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    queue_path.chmod(0o600)
    report_path.chmod(0o600)
    print(
        "Task-label agreement: "
        f"{detailed['task_label_agreements']}/{detailed['tasks']}"
    )
    print(
        "Candidate-label agreement: "
        f"{detailed['candidate_label_agreements']}/"
        f"{detailed['candidate_judgments']}"
    )
    print(f"Exact agreements: {detailed['task_exact_agreements']}")
    print(f"Adjudication tasks: {len(blind_queue)}")
    print("Labels in adjudication queue: False")
    print(f"Queue: {queue_path}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
