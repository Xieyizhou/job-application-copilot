"""Evaluate and fit an experimental model on reviewed evidence annotations."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_v2"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "ml" / "models" / "evidence_support_pilot.joblib"
DEFAULT_PORTABLE_PATH = PROJECT_ROOT / "data" / "ml" / "models" / "evidence_support_pilot.json"
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "evidence_support_pilot_metrics.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_experiment import run_annotation_experiment  # noqa: E402
from ml.relevance import MODEL_SCHEMA_VERSION, PairRelevanceModel  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--portable-path", type=Path, default=DEFAULT_PORTABLE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.dataset_dir / "annotated_tasks.jsonl")
    pairs = load_jsonl(args.dataset_dir / "training_pairs.jsonl")
    if not tasks or not pairs:
        raise SystemExit("Export the reviewed annotation dataset before training.")
    report = run_annotation_experiment(tasks, pairs, random_state=args.random_state)
    threshold = float(report["trained_threshold_median"])
    model = PairRelevanceModel(max_features=5_000, random_state=args.random_state)
    model.fit(
        [str(pair["evidence"]) for pair in pairs],
        [str(pair["requirement"]) for pair in pairs],
        [int(pair["binary_label"]) for pair in pairs],
    )
    metadata: dict[str, object] = {
        "model_version": "reviewed-evidence-pilot-v2",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_source": "locally_reviewed_fictional_evidence_pairs",
        "status": "experimental_not_used_by_application",
        "threshold": threshold,
        "evaluation": report,
        "feature_manifest": model.feature_manifest(),
    }
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": model,
        "threshold": threshold,
        "metadata": metadata,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    args.portable_path.write_text(
        json.dumps(
            model.export_portable(threshold=threshold, metadata=metadata),
            separators=(",", ":"),
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved experimental model: {args.model_path}")
    print(f"Saved aggregate report: {args.report_path}")
    for name, result in report["methods"].items():
        pair = result["pair_classification"]
        retrieval = result["retrieval"]
        print(
            f"{name}: pair F1={pair['f1']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"MRR={retrieval['mean_reciprocal_rank']:.3f}, "
            f"no-support rejection={retrieval['no_support_rejection_rate']:.3f}"
        )
    print("Status: experimental only; application inference is unchanged.")


if __name__ == "__main__":
    main()
