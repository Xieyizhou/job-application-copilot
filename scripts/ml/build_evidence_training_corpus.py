"""Combine the reviewed v3 human seed with accepted consensus gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_HUMAN_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_v3_seed"
)
DEFAULT_GOLD_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "annotations"
    / "pipeline_v1"
    / "gold"
    / "gold_tasks.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_dataset import write_jsonl  # noqa: E402
from ml.evidence_corpus import combine_reviewed_sources  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dir", type=Path, default=DEFAULT_HUMAN_DIR)
    parser.add_argument("--gold-path", type=Path, default=DEFAULT_GOLD_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    human_tasks = load_jsonl(args.human_dir / "annotated_tasks.jsonl")
    human_pairs = load_jsonl(args.human_dir / "training_pairs.jsonl")
    gold_tasks = load_jsonl(args.gold_path)
    if not human_tasks or not human_pairs:
        raise SystemExit("Export the reviewed v3 human seed before combining sources.")
    if not gold_tasks:
        raise SystemExit("No accepted consensus gold tasks were found.")
    tasks, pairs, manifest = combine_reviewed_sources(
        human_tasks,
        human_pairs,
        gold_tasks,
    )
    write_jsonl(tasks, args.output_dir / "annotated_tasks.jsonl")
    write_jsonl(pairs, args.output_dir / "training_pairs.jsonl")
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Combined tasks: {manifest['task_count']}")
    print(f"Combined pairs: {manifest['pair_count']}")
    print(f"Task sources: {manifest['task_source_counts']}")
    print(f"Pair labels: {manifest['pair_label_counts']}")
    print(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
