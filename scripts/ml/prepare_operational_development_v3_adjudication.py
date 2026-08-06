"""Freeze v3 repeat agreement and prepare content-only self-adjudication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DIR = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "annotations"
    / "operational_development_v3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import (  # noqa: E402
    latest_task_states,
    load_jsonl,
    load_queue,
    write_queue,
)
from ml.candidate_judgments import has_complete_candidate_labels  # noqa: E402
from ml.operational_development_v3 import (  # noqa: E402
    build_self_adjudication_queue,
    repeat_agreement_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--random-state", type=int, default=20260731)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_paths = {
        "agreement": args.annotation_dir / "repeat_agreement.json",
        "queue": args.annotation_dir / "self_adjudication_queue.jsonl",
    }
    if any(path.exists() for path in output_paths.values()):
        raise SystemExit("Repeat agreement or adjudication queue already exists.")
    source = load_queue(args.annotation_dir / "queue.jsonl")
    reviewer = load_queue(args.annotation_dir / "reviewer_queue.jsonl")
    repeat_map = json.loads(
        (args.annotation_dir / "repeat_map.json").read_text(encoding="utf-8")
    )
    states = latest_task_states(
        load_jsonl(args.annotation_dir / "reviewer_decisions.jsonl")
    )
    expected_presentations = {str(task["task_id"]) for task in reviewer}
    if set(states) != expected_presentations:
        missing = len(expected_presentations - set(states))
        raise SystemExit(
            f"Complete all reviewer presentations first; {missing} remain."
        )
    incomplete = [
        str(task["task_id"])
        for task in reviewer
        if not has_complete_candidate_labels(
            task,
            states.get(str(task["task_id"])),
        )
    ]
    if incomplete:
        raise SystemExit(
            f"{len(incomplete)} presentations lack complete candidate labels."
        )
    report = repeat_agreement_report(source, repeat_map, states)
    queue = build_self_adjudication_queue(
        source,
        report["disagreement_task_ids"],
        random_state=args.random_state,
    )
    output_paths["agreement"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_queue(queue, output_paths["queue"])
    print(
        "Candidate Cohen kappa: "
        f"{report['candidate_cohen_kappa']:.3f}"
    )
    print(
        "Task exact agreement: "
        f"{report['task_exact_agreement_rate']:.3f}"
    )
    print(f"Self-adjudication tasks: {len(queue)}")
    print(f"Agreement gate passed: {report['agreement_gate_passed']}")
    print(f"Output: {args.annotation_dir}")


if __name__ == "__main__":
    main()
