"""Create independently shuffled reviewer packets for a real-data split."""

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

from ml.annotation import load_jsonl, write_queue  # noqa: E402
from ml.real_review import reviewer_packet_manifest, shuffled_reviewer_queue  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation-dir", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--split", choices=("validation", "reserve"), required=True)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefix = f"real_{args.split}"
    source_path = args.annotation_dir / f"{prefix}_queue_v1.jsonl"
    outputs = {
        "reviewer_a": args.annotation_dir / f"{prefix}_reviewer_a_queue_v1.jsonl",
        "reviewer_b": args.annotation_dir / f"{prefix}_reviewer_b_queue_v1.jsonl",
    }
    manifest_path = args.annotation_dir / f"{prefix}_reviewer_manifest_v1.json"
    if manifest_path.exists() or any(path.exists() for path in outputs.values()):
        raise SystemExit(f"{args.split.title()} reviewer packet exists; refusing overwrite.")
    source = load_jsonl(source_path)
    queues = {
        reviewer_id: shuffled_reviewer_queue(
            source,
            reviewer_id=f"{args.split}_{reviewer_id}",
            random_state=args.random_state,
        )
        for reviewer_id in outputs
    }
    for reviewer_id, path in outputs.items():
        write_queue(queues[reviewer_id], path)
    manifest = {
        **reviewer_packet_manifest(source, queues),
        "split": args.split,
        "source_queue": source_path.name,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"{args.split.title()} tasks per reviewer: {manifest['tasks']}")
    print("Content identity verified: yes")
    print(f"Reviewer A: {outputs['reviewer_a']}")
    print(f"Reviewer B: {outputs['reviewer_b']}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
