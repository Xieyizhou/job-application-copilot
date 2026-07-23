"""Build anonymous real-data train, validation, and frozen holdout splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_INPUT_PATH = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_reviewed_tasks.jsonl"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "ml" / "processed" / "real_validation_v1"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_dataset import write_jsonl  # noqa: E402
from ml.grouped_validation import (  # noqa: E402
    SPLIT_NAMES,
    grouped_split,
    grouped_split_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(args.input_path)
    if not rows:
        raise SystemExit(f"No reviewed real-data tasks found at {args.input_path}.")
    split_rows = grouped_split(rows, random_state=args.random_state)
    report = grouped_split_report(split_rows)
    for split in SPLIT_NAMES:
        write_jsonl(split_rows[split], args.output_dir / f"{split}.jsonl")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Split counts: {report['split_counts']}")
    print(f"Holdout checksum: {report['holdout_checksum_sha256']}")
    print(f"Warnings: {len(report['warnings'])}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
