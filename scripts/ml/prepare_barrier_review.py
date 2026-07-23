"""Validate local synthetic proposals and build strict blind-review packets."""

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
from ml.barrier_audit import (  # noqa: E402
    audit_proposal_pool,
    blind_packet_leaks,
)
from ml.review_consensus import build_blind_queue, merge_generated_cases  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-dir", type=Path, default=DEFAULT_PIPELINE_DIR)
    parser.add_argument(
        "--reviewers",
        nargs="+",
        default=["reviewer_01", "reviewer_02", "reviewer_03"],
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _load_proposals(proposal_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    source_files: list[str] = []
    for path in sorted(proposal_dir.glob("*.jsonl")):
        if path.name == "merged.jsonl":
            continue
        records.extend(load_jsonl(path))
        source_files.append(path.name)
    return records, source_files


def main() -> None:
    args = parse_args()
    pipeline_dir = args.pipeline_dir
    proposal_dir = pipeline_dir / "proposals" / "incoming"
    records, source_files = _load_proposals(proposal_dir)
    cases, rejected = merge_generated_cases(records)
    if not cases:
        raise SystemExit(f"No valid proposals found in {proposal_dir}.")

    merged_path = pipeline_dir / "proposals" / "merged.jsonl"
    write_jsonl(cases, merged_path)
    leak_failures: list[str] = []
    for reviewer_id in args.reviewers:
        queue, mappings = build_blind_queue(
            cases,
            reviewer_id=reviewer_id,
            random_state=args.random_state,
        )
        leaks = blind_packet_leaks(queue)
        if leaks:
            leak_failures.extend(f"{reviewer_id}: {leak}" for leak in leaks)
            continue
        write_jsonl(queue, pipeline_dir / "blind_packets" / f"{reviewer_id}.jsonl")
        write_jsonl(mappings, pipeline_dir / "blind_maps" / f"{reviewer_id}.jsonl")
    if leak_failures:
        raise SystemExit("Blind packet validation failed:\n" + "\n".join(leak_failures))

    report = audit_proposal_pool(cases, rejected_records=rejected)
    report.update(
        {
            "source_files": source_files,
            "input_record_count": len(records),
            "reviewers": args.reviewers,
            "random_state": args.random_state,
            "blind_packet_leaks": [],
        }
    )
    report_path = pipeline_dir / "manifests" / "prepare_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Accepted proposals: {len(cases)}")
    print(f"Rejected proposals: {len(rejected)}")
    print(f"Blind reviewer packets: {len(args.reviewers)}")
    print(f"Quality warnings: {len(report['warnings'])}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
