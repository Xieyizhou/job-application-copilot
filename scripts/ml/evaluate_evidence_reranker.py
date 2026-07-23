"""Evaluate the local v3 reranker on the curated de-identified semantic set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models" / "evidence_pairwise_reranker_v3.joblib"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT
    / "reports"
    / "ml"
    / "generated"
    / "evidence_reranker_v3_external_diagnostic.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.evidence_validation import (  # noqa: E402
    DEFAULT_SEMANTIC_MANIFEST,
    evaluate_reranker_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--manifest-path", type=Path, default=DEFAULT_SEMANTIC_MANIFEST)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    training_pairs = load_jsonl(args.dataset_dir / "training_pairs.jsonl")
    report = evaluate_reranker_artifact(
        manifest_path=args.manifest_path,
        model_path=args.model_path,
        training_pairs=training_pairs,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    metrics = report["metrics"]
    print(
        f"External diagnostic: {metrics['examples']} cases, "
        f"F1={metrics['f1']:.3f}, balanced accuracy={metrics['balanced_accuracy']:.3f}, "
        f"AP={metrics['average_precision']:.3f}"
    )
    print(f"Failed cases: {', '.join(report['failed_case_ids']) or 'none'}")
    print("Promotion eligible: no; a frozen 40+ task real holdout is still required.")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
