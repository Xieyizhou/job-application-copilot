"""Train and evaluate a constraint-aware top-candidate acceptance gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
import sys
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"
TRAINING_DIR = (
    PROJECT_ROOT / "data" / "ml" / "processed" / "reviewed_evidence_training_v3"
)
ARTIFACT_PATH = (
    PROJECT_ROOT
    / "data"
    / "ml"
    / "models"
    / "evidence_acceptance_candidate_v1.joblib"
)
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.annotation import load_jsonl  # noqa: E402
from ml.annotation_experiment import NO_PORTABLE_MODEL  # noqa: E402
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair  # noqa: E402
from ml.evidence_acceptance import ConstraintAwareSupportClassifier  # noqa: E402
from ml.evidence_artifact import training_corpus_fingerprint  # noqa: E402
from ml.evidence_constraints import (  # noqa: E402
    constraint_features,
    hard_constraints_supported,
)
from ml.evidence_decision import (  # noqa: E402
    SupportSignal,
    paired_stratified_bootstrap_delta,
    select_support_gate_threshold,
)
from ml.evidence_models import WordTfidfCosineScorer  # noqa: E402
from ml.real_holdout_evaluation import (  # noqa: E402
    assert_holdout_isolated,
    evaluate_holdout_method,
    score_holdout_tasks,
    select_validation_threshold,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-version", type=int, choices=(2, 3), default=2)
    parser.add_argument("--annotation-dir", type=Path)
    parser.add_argument("--training-dir", type=Path, default=TRAINING_DIR)
    parser.add_argument("--report-path", type=Path)
    parser.add_argument("--artifact-path", type=Path, default=ARTIFACT_PATH)
    return parser.parse_args()


def _load_development(
    development_dir: Path,
    version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    gold_path = development_dir / f"real_development_gold_{version}.jsonl"
    manifest_path = (
        development_dir / f"real_development_frozen_manifest_{version}.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checksum = hashlib.sha256(gold_path.read_bytes()).hexdigest()
    if checksum != manifest["gold_sha256"]:
        raise SystemExit("Frozen development checksum mismatch; evaluation blocked.")
    return load_jsonl(gold_path), manifest


def _score_tasks(
    tasks: list[dict[str, Any]],
    *,
    ranker: WordTfidfCosineScorer,
    gate: ConstraintAwareSupportClassifier,
) -> tuple[dict[str, list[float]], dict[str, list[SupportSignal]]]:
    rankings: dict[str, list[float]] = {}
    signals: dict[str, list[SupportSignal]] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        candidates = list(task["candidates"])
        requirements = [str(task["requirement"])] * len(candidates)
        evidence = [str(candidate["evidence"]) for candidate in candidates]
        rankings[task_id] = [
            float(value) for value in ranker.score(requirements, evidence)
        ]
        probabilities = gate.predict_proba(requirements, evidence)
        task_signals: list[SupportSignal] = []
        for requirement, candidate, probability in zip(
            requirements,
            evidence,
            probabilities,
            strict=True,
        ):
            constraints = constraint_features(requirement, candidate)
            task_signals.append(
                {
                    "similarity": float(probability),
                    "has_overlap": True,
                    "numeric_constraint_supported": bool(
                        constraints["numeric_constraint_supported"]
                    ),
                    "compound_requirement_supported": True,
                    "hard_constraint_supported": hard_constraints_supported(
                        constraints
                    ),
                    "explicit_negation": bool(constraints["explicit_negation"]),
                }
            )
        signals[task_id] = task_signals
    return rankings, signals


def main() -> None:
    args = parse_args()
    version = f"v{args.dataset_version}"
    development_dir = args.annotation_dir or (
        PROJECT_ROOT
        / "data"
        / "ml"
        / "annotations"
        / f"real_development_{version}"
    )
    report_path = args.report_path or (
        PROJECT_ROOT
        / "reports"
        / "ml"
        / "generated"
        / f"constraint_gate_development_{version}.json"
    )
    development_tasks, manifest = _load_development(
        development_dir,
        version,
    )
    training_tasks = load_jsonl(args.training_dir / "annotated_tasks.jsonl")
    training_pairs = load_jsonl(args.training_dir / "training_pairs.jsonl")
    assert_holdout_isolated(
        development_tasks,
        training_tasks,
        training_pairs,
    )
    requirements = [str(pair["requirement"]) for pair in training_pairs]
    evidence = [str(pair["evidence"]) for pair in training_pairs]
    labels = [int(pair["binary_label"]) for pair in training_pairs]

    ranker = WordTfidfCosineScorer().fit(requirements, evidence)
    gate = ConstraintAwareSupportClassifier(random_state=42).fit(
        requirements,
        evidence,
        labels,
    )
    ranking_scores, support_signals = _score_tasks(
        development_tasks,
        ranker=ranker,
        gate=gate,
    )
    gate_threshold, gate_result = select_support_gate_threshold(
        development_tasks,
        ranking_scores,
        support_signals,
    )
    tfidf_scores = score_holdout_tasks(
        development_tasks,
        score=ranker.score,
    )
    tfidf_threshold, tfidf_result = select_validation_threshold(
        development_tasks,
        tfidf_scores,
    )

    def concept_score(
        aligned_requirements: Sequence[str],
        aligned_evidence: Sequence[str],
    ) -> list[float]:
        return [
            float(
                score_evidence_pair(
                    requirement,
                    candidate,
                    model_path=NO_PORTABLE_MODEL,
                )["similarity"]
            )
            for requirement, candidate in zip(
                aligned_requirements,
                aligned_evidence,
                strict=True,
            )
        ]

    fixed = evaluate_holdout_method(
        development_tasks,
        score=concept_score,
        threshold=MIN_ACCEPTED_SIMILARITY,
    )
    gate_metrics = gate_result["retrieval"]
    tfidf_metrics = tfidf_result["retrieval"]
    gate_failures = {
        str(failure["task_id"]) for failure in gate_result["failures"]
    }
    tfidf_failures = {
        str(failure["task_id"]) for failure in tfidf_result["failures"]
    }
    support_mask = [
        str(task["support_label"]) != "No Support"
        for task in development_tasks
    ]
    tfidf_outcomes = [
        str(task["task_id"]) not in tfidf_failures
        for task in development_tasks
    ]
    gate_outcomes = [
        str(task["task_id"]) not in gate_failures
        for task in development_tasks
    ]
    bootstrap = paired_stratified_bootstrap_delta(
        support_mask,
        tfidf_outcomes,
        gate_outcomes,
    )
    point_estimate_wins = (
        gate_metrics["task_balanced_accuracy"]
        > tfidf_metrics["task_balanced_accuracy"]
        and gate_metrics["recall_at_1"] >= tfidf_metrics["recall_at_1"]
        and gate_metrics["no_support_rejection_rate"]
        >= tfidf_metrics["no_support_rejection_rate"]
    )
    wins_development = (
        point_estimate_wins and float(bootstrap["lower_90"]) > 0
    )
    training_fingerprint = training_corpus_fingerprint(
        training_tasks,
        training_pairs,
    )
    artifact_status = "blocked_development_not_won"
    if wins_development:
        artifact = {
            "schema_version": 1,
            "model_type": "tfidf_constraint_acceptance",
            "ranker": ranker,
            "gate": gate,
            "support_threshold": gate_threshold,
            "metadata": {
                "status": "candidate_for_fresh_source_isolated_reserve",
                "training_corpus_fingerprint": training_fingerprint,
                "development_gold_sha256": manifest["gold_sha256"],
                "feature_manifest": gate.feature_manifest(),
            },
        }
        args.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact, args.artifact_path)
        artifact_status = "saved_candidate_for_fresh_reserve"
    elif point_estimate_wins:
        artifact_status = "blocked_improvement_not_robust"
        if args.artifact_path.exists():
            stale = joblib.load(args.artifact_path)
            metadata = stale.get("metadata", {}) if isinstance(stale, dict) else {}
            if (
                metadata.get("training_corpus_fingerprint")
                == training_fingerprint
                and metadata.get("development_gold_sha256")
                == manifest["gold_sha256"]
            ):
                args.artifact_path.unlink()

    report = {
        "schema_version": 1,
        "dataset": f"real_development_{version}",
        "dataset_role": "development_model_selection",
        "training_tasks": len(training_tasks),
        "training_pairs": len(training_pairs),
        "exact_training_overlap": 0,
        "source_group_overlap": 0,
        "ranker": "word_tfidf_cosine",
        "fixed_baseline": fixed,
        "tfidf_rank_and_accept_reference": {
            **tfidf_result,
            "threshold": tfidf_threshold,
        },
        "constraint_acceptance_gate": gate_result,
        "feature_manifest": gate.feature_manifest(),
        "selection": {
            "wins_development": wins_development,
            "point_estimate_wins": point_estimate_wins,
            "paired_stratified_bootstrap": bootstrap,
            "status": artifact_status,
            "artifact_path": str(args.artifact_path) if wins_development else None,
        },
        "usage_boundary": (
            "Development-only model selection. A new source-isolated reserve is "
            "required before product integration."
        ),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    for name, result in (
        ("fixed", fixed),
        ("tfidf", tfidf_result),
        ("constraint_gate", gate_result),
    ):
        metrics = result["retrieval"]
        print(
            f"{name}: balanced={metrics['task_balanced_accuracy']:.3f}, "
            f"task accuracy={metrics['task_decision_accuracy']:.3f}, "
            f"Recall@1={metrics['recall_at_1']:.3f}, "
            f"No-support={metrics['no_support_rejection_rate']:.3f}"
        )
    print(f"Constraint threshold: {gate_threshold:.4f}")
    print(
        "Balanced delta 90% interval: "
        f"[{bootstrap['lower_90']:.3f}, {bootstrap['upper_90']:.3f}]"
    )
    print(f"Selection: {artifact_status}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
