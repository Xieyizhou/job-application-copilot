"""Build a local, diagnostic-only queue for teacher ranking-error review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_PAIRS = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "processed"
    / "reviewed_evidence_training_v4_batch1"
    / "training_pairs.jsonl"
)
DISTILLATION_DIR = PROJECT_ROOT / "data" / "ml" / "distillation"
DEFAULT_EMBEDDING = DISTILLATION_DIR / "embedding_baseline_v1" / "teacher_scores.jsonl"
DEFAULT_RERANKER = DISTILLATION_DIR / "qwen_reranker_v1" / "teacher_scores.jsonl"
DEFAULT_OUTPUT = DISTILLATION_DIR / "teacher_error_audit_v1" / "queue.jsonl"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_dataset import write_jsonl  # noqa: E402
from ml.evidence_distillation import (  # noqa: E402
    assert_allowed_input_path,
    build_teacher_error_audit_queue,
    select_balanced_task_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--embedding-scores", type=Path, default=DEFAULT_EMBEDDING)
    parser.add_argument("--reranker-scores", type=Path, default=DEFAULT_RERANKER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int, default=100)
    parser.add_argument("--random-state", type=int, default=20260731)
    parser.add_argument("--max-business-inversions", type=int, default=8)
    return parser.parse_args()


def aligned_scores(path: Path, sample: list[dict[str, Any]]) -> list[float]:
    by_pair = {str(row["pair_id"]): float(row["score"]) for row in load_jsonl(path)}
    try:
        return [by_pair[str(pair["pair_id"])] for pair in sample]
    except KeyError as error:
        raise SystemExit(f"Missing sampled pair score in {path}: {error}") from error


def main() -> None:
    args = parse_args()
    assert_allowed_input_path(str(args.pairs), input_role="human_reviewed_training")
    if args.output.exists():
        raise SystemExit("Teacher-error audit queue exists; refusing overwrite.")
    pairs = load_jsonl(args.pairs)
    sample = select_balanced_task_sample(
        pairs,
        max_tasks=args.max_tasks,
        random_state=args.random_state,
    )
    queue = build_teacher_error_audit_queue(
        sample,
        aligned_scores(args.embedding_scores, sample),
        aligned_scores(args.reranker_scores, sample),
        max_business_inversions=args.max_business_inversions,
    )
    args.output.parent.mkdir(parents=True, exist_ok=False)
    write_jsonl(queue, args.output)
    args.output.chmod(0o600)
    manifest = {
        "schema_version": 1,
        "tasks": len(queue),
        "qwen_top1_errors": sum(
            "qwen_top1_error" in row["audit_reasons"] for row in queue
        ),
        "business_pair_inversions": sum(
            "business_pair_inversion" in row["audit_reasons"] for row in queue
        ),
        "diagnostic_only": True,
        "human_gold_changes_allowed": False,
        "model_selection_allowed": False,
        "holdout_accessed": False,
    }
    manifest_path = args.output.parent / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.chmod(0o600)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
