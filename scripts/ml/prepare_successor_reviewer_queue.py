"""Create one blind reviewer queue for a successor development batch."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_ANNOTATION_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "successor_development_v8"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.operational_development_v3 import (  # noqa: E402
    build_blind_reviewer_queue,
)
from ml.real_holdout import jsonl_bytes, sha256_bytes  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=DEFAULT_ANNOTATION_DIR,
    )
    parser.add_argument("--reviewer-id", required=True)
    parser.add_argument("--random-state", type=int, default=20260806)
    return parser.parse_args()


def _content_signature(tasks: list[dict[str, Any]]) -> str:
    canonical: list[dict[str, Any]] = []
    for task in tasks:
        candidates = sorted(
            (
                {
                    "candidate_id": str(candidate["candidate_id"]),
                    "evidence": str(candidate["evidence"]),
                }
                for candidate in task["candidates"]
            ),
            key=lambda candidate: candidate["candidate_id"],
        )
        canonical.append(
            {
                "task_id": str(task["task_id"]),
                "role_family": str(task["role_family"]),
                "requirement": str(task["requirement"]),
                "candidates": candidates,
            }
        )
    canonical.sort(key=lambda task: task["task_id"])
    return sha256_bytes(jsonl_bytes(canonical))


def main() -> None:
    args = parse_args()
    reviewer_id = str(args.reviewer_id).strip().lower()
    if not reviewer_id.replace("_", "").isalnum():
        raise SystemExit("Reviewer id may contain only letters, digits, or underscore.")
    source_path = args.annotation_dir / "queue.jsonl"
    output_path = args.annotation_dir / f"{reviewer_id}_queue.jsonl"
    manifest_path = args.annotation_dir / f"{reviewer_id}_manifest.json"
    if output_path.exists() or manifest_path.exists():
        raise SystemExit("Reviewer packet exists; refusing overwrite.")
    source = load_jsonl(source_path)
    reviewer = build_blind_reviewer_queue(
        source,
        reviewer_id=reviewer_id,
        random_state=args.random_state,
    )
    source_signature = _content_signature(source)
    reviewer_signature = _content_signature(reviewer)
    if source_signature != reviewer_signature:
        raise ValueError("Blind reviewer queue changed task content.")
    source_orders = {
        str(task["task_id"]): [
            str(candidate["candidate_id"]) for candidate in task["candidates"]
        ]
        for task in source
    }
    changed_orders = sum(
        [
            str(candidate["candidate_id"]) for candidate in task["candidates"]
        ]
        != source_orders[str(task["task_id"])]
        for task in reviewer
    )
    if changed_orders != len(source):
        raise ValueError("Every reviewer task must use a changed candidate order.")
    write_queue(reviewer, output_path)
    manifest = {
        "schema_version": 1,
        "reviewer_id": reviewer_id,
        "tasks": len(reviewer),
        "candidate_judgments": sum(
            len(task["candidates"]) for task in reviewer
        ),
        "source_queue_sha256": sha256_bytes(source_path.read_bytes()),
        "content_sha256": source_signature,
        "ordered_packet_sha256": sha256_bytes(jsonl_bytes(reviewer)),
        "candidate_orders_changed": changed_orders,
        "labels_included": False,
        "model_predictions_included": False,
        "construction_metadata_included": False,
        "other_reviewer_decisions_read": False,
        "random_state_commitment": hashlib.sha256(
            f"{args.random_state}:{reviewer_id}".encode()
        ).hexdigest(),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_path.chmod(0o600)
    manifest_path.chmod(0o600)
    print(f"Reviewer: {reviewer_id}")
    print(f"Tasks: {manifest['tasks']}")
    print(f"Candidate judgments: {manifest['candidate_judgments']}")
    print(f"Candidate orders changed: {manifest['candidate_orders_changed']}")
    print("Labels included: False")
    print("Other reviewer decisions read: False")
    print(f"Queue: {output_path}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
