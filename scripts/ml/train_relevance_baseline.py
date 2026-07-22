"""Train and evaluate the optional local resume/job relevance model."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "data" / "ml" / "raw" / "candidate_matching_synthetic"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "data" / "ml" / "models" / "relevance_baseline.joblib"
DEFAULT_INFERENCE_PATH = PROJECT_ROOT / "data" / "ml" / "models" / "relevance_baseline.json"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "ml" / "generated" / "relevance_baseline_metrics.json"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evaluation import classification_metrics, mean_job_average_precision, select_f1_threshold
from ml.relevance import MODEL_SCHEMA_VERSION, PairRelevanceModel
from ml.synthetic import SyntheticPair, load_synthetic_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--inference-path", type=Path, default=DEFAULT_INFERENCE_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--max-jobs", type=int, default=None, help="Optional development-only job cap.")
    parser.add_argument("--negative-ratio", type=float, default=1.0)
    parser.add_argument("--max-features", type=int, default=20_000)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def subset(pairs: list[SyntheticPair], name: str) -> list[SyntheticPair]:
    result = [pair for pair in pairs if pair.subset == name]
    if not result:
        raise ValueError(f"No examples were assigned to {name}.")
    return result


def predict(model: PairRelevanceModel, pairs: list[SyntheticPair]):
    return model.predict_proba(
        [pair.resume_text for pair in pairs],
        [pair.job_text for pair in pairs],
    )


def main() -> None:
    args = parse_args()
    pairs = load_synthetic_pairs(
        args.dataset_dir,
        negatives_per_positive=args.negative_ratio,
        random_state=args.random_state,
        max_jobs=args.max_jobs,
    )
    train_pairs = subset(pairs, "train")
    validation_pairs = subset(pairs, "validation")
    test_pairs = subset(pairs, "test")
    model = PairRelevanceModel(max_features=args.max_features, random_state=args.random_state)
    model.fit(
        [pair.resume_text for pair in train_pairs],
        [pair.job_text for pair in train_pairs],
        [pair.label for pair in train_pairs],
    )
    validation_probabilities = predict(model, validation_pairs)
    threshold = select_f1_threshold(
        [pair.label for pair in validation_pairs],
        validation_probabilities,
    )
    test_probabilities = predict(model, test_pairs)
    validation_metrics = classification_metrics(
        [pair.label for pair in validation_pairs],
        validation_probabilities,
        threshold=threshold,
    )
    test_metrics = classification_metrics(
        [pair.label for pair in test_pairs],
        test_probabilities,
        threshold=threshold,
    )
    test_metrics["mean_job_average_precision"] = mean_job_average_precision(
        [pair.job_id for pair in test_pairs],
        [pair.label for pair in test_pairs],
        test_probabilities,
    )
    trained_at = datetime.now(timezone.utc).isoformat()
    metadata = {
        "model_version": "synthetic-unseen-job-v1",
        "trained_at": trained_at,
        "training_source": "candidate_matching_synthetic",
        "split_protocol": "stable hash by job_id (70/15/15)",
        "decision_role": "auxiliary only; does not change Role Fit, eligibility, or recommendation",
        "row_counts": dict(Counter(pair.subset for pair in pairs)),
        "job_counts": {
            name: len({pair.job_id for pair in pairs if pair.subset == name})
            for name in ("train", "validation", "test")
        },
        "feature_manifest": model.feature_manifest(),
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
    }
    artifact = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "model": model,
        "threshold": threshold,
        "metadata": metadata,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    portable_artifact = model.export_portable(threshold=threshold, metadata=metadata)
    args.inference_path.parent.mkdir(parents=True, exist_ok=True)
    args.inference_path.write_text(
        json.dumps(portable_artifact, separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Saved local model: {args.model_path}")
    print(f"Saved portable inference model: {args.inference_path}")
    print(f"Saved aggregate report: {args.report_path}")
    print(json.dumps(test_metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
