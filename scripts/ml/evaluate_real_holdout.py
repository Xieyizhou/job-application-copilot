"""Evaluate frozen real-text holdout with thresholds fixed before inspection."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
from pathlib import Path
import sys

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
DEFAULT_HOLDOUT_DIR = (
    PROJECT_ROOT / "data" / "ml" / "annotations" / "real_holdout_v1"
)
DEFAULT_DATASET_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
DEFAULT_MODEL_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models" / "evidence_reranker_v3.joblib"
)
DEFAULT_TRAINING_REPORT = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "evidence_reranker_v3_metrics.json"
)
DEFAULT_REPORT = (
    PROJECT_ROOT / "reports" / "ml" / "generated" / "real_holdout_v1_metrics.json"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_experiment import NO_PORTABLE_MODEL  # noqa: E402
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair  # noqa: E402
from ml.evidence_artifact import validate_evidence_artifact  # noqa: E402
from ml.evidence_models import LsaEmbeddingScorer  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_holdout_method,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout-dir", type=Path, default=DEFAULT_HOLDOUT_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--training-report", type=Path, default=DEFAULT_TRAINING_REPORT)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gold_path = args.holdout_dir / "real_holdout_gold_v1.jsonl"
    manifest_path = args.holdout_dir / "real_holdout_frozen_manifest_v1.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(gold_path.read_bytes()).hexdigest() != manifest["gold_sha256"]:
        raise SystemExit("Frozen holdout checksum mismatch; evaluation blocked.")
    tasks = load_jsonl(gold_path)
    training_tasks = load_jsonl(args.dataset_dir / "annotated_tasks.jsonl")
    training_pairs = load_jsonl(args.dataset_dir / "training_pairs.jsonl")
    assert_holdout_isolated(tasks, training_tasks, training_pairs)

    training_report = json.loads(args.training_report.read_text(encoding="utf-8"))
    train_requirements = [str(pair["requirement"]) for pair in training_pairs]
    train_evidence = [str(pair["evidence"]) for pair in training_pairs]
    lsa = LsaEmbeddingScorer(random_state=42).fit(
        train_requirements,
        train_evidence,
    )
    artifact = validate_evidence_artifact(
        joblib.load(args.model_path),
        tasks=training_tasks,
        pairs=training_pairs,
    )
    model = artifact["model"]

    def concept_score(
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> list[float]:
        return [
            float(
                score_evidence_pair(
                    requirement,
                    candidate,
                    model_path=NO_PORTABLE_MODEL,
                )["similarity"]
            )
            for requirement, candidate in zip(requirements, evidence, strict=True)
        ]

    methods = {
        "concept_lexical_rule": evaluate_holdout_method(
            tasks,
            score=concept_score,
            threshold=MIN_ACCEPTED_SIMILARITY,
        ),
        "lsa_embedding": evaluate_holdout_method(
            tasks,
            score=lsa.score,
            threshold=float(
                training_report["method_threshold_medians"]["lsa_embedding"]
            ),
        ),
        str(artifact["model_type"]): evaluate_holdout_method(
            tasks,
            score=model.predict_proba,
            threshold=float(artifact["threshold"]),
        ),
    }
    baseline = methods["concept_lexical_rule"]["retrieval"]
    artifact_name = str(artifact["model_type"])
    artifact_retrieval = methods[artifact_name]["retrieval"]
    artifact_beats_baseline = (
        artifact_retrieval["task_decision_accuracy"]
        > baseline["task_decision_accuracy"]
        and artifact_retrieval["recall_at_1"] >= baseline["recall_at_1"]
        and artifact_retrieval["no_support_rejection_rate"]
        >= baseline["no_support_rejection_rate"]
    )
    report = {
        "schema_version": 1,
        "dataset": "real_holdout_v1",
        "gold_sha256": manifest["gold_sha256"],
        "tasks": len(tasks),
        "training_tasks": len(training_tasks),
        "training_pairs": len(training_pairs),
        "exact_training_overlap": 0,
        "source_group_overlap": 0,
        "threshold_policy": "fixed from training-only evaluation before holdout scoring",
        "methods": methods,
        "status": "independent_real_holdout_evaluation",
        "promotion_gate": {
            "artifact": artifact_name,
            "baseline": "concept_lexical_rule",
            "criteria": [
                "higher task_decision_accuracy",
                "non-inferior recall_at_1",
                "non-inferior no_support_rejection_rate",
            ],
            "passed": artifact_beats_baseline,
            "status": (
                "candidate_for_further_review"
                if artifact_beats_baseline
                else "blocked_baseline_not_beaten"
            ),
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    args.report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in methods.items():
        pair = result["pair_classification"]
        retrieval = result["retrieval"]
        print(
            f"{name}: F1={pair['f1']:.3f}, AP={pair['average_precision']:.3f}, "
            f"Recall@1={retrieval['recall_at_1']:.3f}, "
            f"MRR={retrieval['mean_reciprocal_rank']:.3f}, "
            f"No-support rejection={retrieval['no_support_rejection_rate']:.3f}, "
            f"task accuracy={retrieval['task_decision_accuracy']:.3f}"
        )
    print(f"Promotion gate: {report['promotion_gate']['status']}")
    print(f"Report: {args.report_path}")


if __name__ == "__main__":
    main()
