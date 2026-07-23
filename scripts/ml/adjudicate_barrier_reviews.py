"""Unblind local reviewer decisions and promote unanimous consensus into gold tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PIPELINE_DIR = PROJECT_ROOT / "data" / "ml" / "annotations" / "pipeline_v1"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_dataset import write_jsonl  # noqa: E402
from ml.barrier_contracts import BarrierContractError  # noqa: E402
from ml.review_consensus import (  # noqa: E402
    adjudicate_reviews,
    human_review_cases,
    promote_consensus_gold,
    unblind_review,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument("--minimum-reviewers", type=int, default=3)
    return parser.parse_args()


def _read_all(directory: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.jsonl")):
        rows.extend(load_jsonl(path))
    return rows


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir
    cases = load_jsonl(pipeline_dir / "proposals" / "merged.jsonl")
    case_by_id = {str(case["case_id"]): case for case in cases}
    mappings = _read_all(pipeline_dir / "blind_maps")
    mapping_by_task = {
        str(mapping["review_task_id"]): mapping
        for mapping in mappings
    }
    blind_reviews = _read_all(pipeline_dir / "reviews")
    canonical_reviews: list[dict[str, Any]] = []
    unblind_errors: list[str] = []
    for blind_review in blind_reviews:
        task_id = str(blind_review.get("review_task_id", ""))
        mapping = mapping_by_task.get(task_id)
        if mapping is None:
            unblind_errors.append(f"Review references unknown task: {task_id or 'missing'}")
            continue
        case = case_by_id.get(str(mapping["case_id"]))
        if case is None:
            unblind_errors.append(f"Mapping references unknown case: {mapping['case_id']}")
            continue
        try:
            canonical_reviews.append(unblind_review(blind_review, mapping, case))
        except BarrierContractError as error:
            unblind_errors.append(f"{task_id}: {error}")

    decisions, report = adjudicate_reviews(
        cases,
        canonical_reviews,
        minimum_reviewers=args.minimum_reviewers,
    )
    report["unblind_errors"] = unblind_errors
    gold_tasks = promote_consensus_gold(cases, decisions, canonical_reviews)
    review_queue = human_review_cases(cases, decisions)

    write_jsonl(canonical_reviews, pipeline_dir / "consensus" / "canonical_reviews.jsonl")
    write_jsonl(decisions, pipeline_dir / "consensus" / "decisions.jsonl")
    write_jsonl(review_queue, pipeline_dir / "adjudication" / "queue.jsonl")
    write_jsonl(gold_tasks, pipeline_dir / "gold" / "gold_tasks.jsonl")
    report.update(
        {
            "blind_review_records": len(blind_reviews),
            "canonical_review_records": len(canonical_reviews),
            "gold_tasks": len(gold_tasks),
            "human_review_tasks": len(review_queue),
            "gold_decision_sources": sorted(
                {str(task["decision_source"]) for task in gold_tasks}
            ),
        }
    )
    report_path = pipeline_dir / "manifests" / "consensus_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Canonical reviews: {len(canonical_reviews)}")
    print(f"Gold tasks: {len(gold_tasks)}")
    print(f"Human review queue: {len(review_queue)}")
    print(f"Unblind errors: {len(unblind_errors)}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
