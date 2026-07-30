"""Select and train the winning reviewed-evidence reranker as a local experiment."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models" / "evidence_reranker_v3.joblib"
)
DEFAULT_REPORT_PATH = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "evidence_reranker_v3_metrics.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_experiment import run_annotation_experiment  # noqa: E402
from ml.evidence_artifact import (  # noqa: E402
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    SUPPORTED_ARTIFACT_MODEL_TYPES,
    training_corpus_fingerprint,
)
from ml.evidence_models import fit_selected_reranker  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def _write_report(report: dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    tasks = load_jsonl(args.dataset_dir / "annotated_tasks.jsonl")
    pairs = load_jsonl(args.dataset_dir / "training_pairs.jsonl")
    if not tasks or not pairs:
        raise SystemExit("Build the reviewed v3 training corpus before fitting.")
    report = run_annotation_experiment(tasks, pairs, random_state=args.random_state)
    selected_method = str(report["model_selection"]["selected_method"])
    corpus_fingerprint = training_corpus_fingerprint(tasks, pairs)
    report["training_corpus"] = {
        "fingerprint": corpus_fingerprint,
        "tasks": len(tasks),
        "pairs": len(pairs),
    }
    report["artifact_generation"] = {
        "status": "comparison_complete_not_fitted",
        "selected_method": selected_method,
    }
    _write_report(report, args.report_path)
    if selected_method not in SUPPORTED_ARTIFACT_MODEL_TYPES:
        raise SystemExit(
            f"Selected {selected_method}, which is not an artifact-capable reranker. "
            f"The comparison report was saved to {args.report_path}; no model was fitted."
        )
    threshold = float(report["method_threshold_medians"][selected_method])
    requirements = [str(pair["requirement"]) for pair in pairs]
    evidence = [str(pair["evidence"]) for pair in pairs]
    labels = [int(pair["binary_label"]) for pair in pairs]
    model = fit_selected_reranker(
        selected_method,
        requirements,
        evidence,
        labels,
        tasks,
        random_state=args.random_state,
    )
    metadata = {
        "model_version": f"reviewed-evidence-v3-{selected_method}",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_source": "reviewed_evidence_training_v3",
        "training_task_count": len(tasks),
        "training_pair_count": len(pairs),
        "training_corpus_fingerprint": corpus_fingerprint,
        "status": "experimental_not_used_by_application",
        "threshold": threshold,
        "evaluation_protocol": report["evaluation_protocol"],
        "selection": report["model_selection"],
        "feature_manifest": model.feature_manifest(),
    }
    artifact = {
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "model_type": selected_method,
        "model": model,
        "threshold": threshold,
        "metadata": metadata,
    }
    args.model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, args.model_path)
    report["artifact_generation"] = {
        "status": "saved_experimental_artifact",
        "selected_method": selected_method,
        "model_path": str(args.model_path),
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    }
    _write_report(report, args.report_path)
    print(f"Saved experimental reranker: {args.model_path}")
    print(f"Saved grouped evaluation report: {args.report_path}")
    for name, result in report["methods"].items():
        pair_metrics = result["pair_classification"]
        retrieval = result["retrieval"]
        print(
            f"{name}: pair F1={pair_metrics['f1']:.3f}, "
            f"balanced accuracy={pair_metrics['balanced_accuracy']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"MRR={retrieval['mean_reciprocal_rank']:.3f}, "
            f"no-support rejection={retrieval['no_support_rejection_rate']:.3f}"
        )
    print("Status: local experiment only; product scoring and inference are unchanged.")


if __name__ == "__main__":
    main()
