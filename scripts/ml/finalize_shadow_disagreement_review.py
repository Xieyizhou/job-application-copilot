"""Finalize a completed blind shadow review into diagnostic metrics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_REVIEW_DIR = (
    PROJECT_ROOT / "data" / "ml" / "shadow"
    / "real_batch_v2_candidate4" / "human_review"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_shadow_review import evaluate_shadow_review  # noqa: E402
from ml.real_holdout import _validated_states  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--events", type=Path)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = parse_args()
    events_path = args.events or args.review_dir / "decisions.jsonl"
    output_path = args.review_dir / "diagnostic_report.json"
    if output_path.exists():
        raise SystemExit("Shadow diagnostic report exists; refusing overwrite.")
    queue_path = args.review_dir / "queue.jsonl"
    construction_path = args.review_dir / "construction.json"
    tasks = load_jsonl(queue_path)
    events = load_jsonl(events_path)
    states = _validated_states(
        tasks,
        events,
        reviewer="Shadow human reviewer",
    )
    construction = json.loads(
        construction_path.read_text(encoding="utf-8")
    )
    report = evaluate_shadow_review(tasks, states, construction)
    report["input_sha256"] = {
        "queue": _sha256(queue_path),
        "events": _sha256(events_path),
        "construction": _sha256(construction_path),
    }
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Tasks: {report['tasks']}")
    for method, metrics in report["methods"].items():
        print(
            f"{method}: accuracy={metrics['diagnostic_accuracy']:.3f}, "
            f"errors={metrics['error_counts']}"
        )
    print(f"Report: {output_path}")


if __name__ == "__main__":
    main()
