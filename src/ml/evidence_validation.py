"""External diagnostics for local evidence reranker artifacts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import joblib

from ml.annotation_generation import normalize_text
from ml.annotation_metrics import decision_metrics
from ml.evidence_artifact import EvidenceArtifactError, validate_evidence_artifact
from ml.validation import (
    DEFAULT_SEMANTIC_MANIFEST,
    load_manifest,
    validate_semantic_manifest,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RERANKER_PATH = (
    PROJECT_ROOT / "data" / "ml" / "models" / "evidence_reranker_v3.joblib"
)


class EvidenceValidationError(ValueError):
    """Raised when an experiment artifact or validation boundary is unsafe."""


def _content_keys(rows: Sequence[dict[str, Any]]) -> set[tuple[str, str]]:
    return {
        (
            normalize_text(str(row["requirement"])),
            normalize_text(str(row["evidence"])),
        )
        for row in rows
    }


def evaluate_reranker_cases(
    cases: Sequence[dict[str, Any]],
    *,
    predict: Callable[[Sequence[str], Sequence[str]], Sequence[float]],
    threshold: float,
    training_pairs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Evaluate de-identified cases and reject exact train/evaluation overlap."""
    evaluation_keys = _content_keys(cases)
    overlap = evaluation_keys & _content_keys(training_pairs)
    if overlap:
        raise EvidenceValidationError(
            "Validation requirement/evidence pairs overlap with the training corpus."
        )
    requirements = [str(case["requirement"]) for case in cases]
    evidence = [str(case["evidence"]) for case in cases]
    labels = [int(bool(case["expected_accepted"])) for case in cases]
    scores = [float(value) for value in predict(requirements, evidence)]
    if len(scores) != len(cases):
        raise EvidenceValidationError("Reranker returned the wrong number of scores.")
    predictions = [int(score >= threshold) for score in scores]
    metrics = decision_metrics(labels, predictions, scores)
    rows = [
        {
            "case_id": case["case_id"],
            "review_tag": case.get("review_tag"),
            "expected": bool(label),
            "predicted": bool(prediction),
            "score": score,
            "passed": label == prediction,
        }
        for case, label, prediction, score in zip(
            cases,
            labels,
            predictions,
            scores,
            strict=True,
        )
    ]
    return {
        "status": "diagnostic_only_small_curated_set",
        "promotion_eligible": False,
        "threshold": threshold,
        "metrics": metrics,
        "failed_case_ids": [row["case_id"] for row in rows if not row["passed"]],
        "exact_training_overlap": 0,
        "rows": rows,
        "limitations": [
            "The curated set contains only 24 cases.",
            "It is not the frozen 40+ task real holdout required for promotion.",
            "Results must not be reported as production accuracy.",
        ],
    }


def evaluate_reranker_artifact(
    *,
    manifest_path: Path = DEFAULT_SEMANTIC_MANIFEST,
    model_path: Path = DEFAULT_RERANKER_PATH,
    training_tasks: Sequence[dict[str, Any]] = (),
    training_pairs: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    """Load one local artifact and evaluate it on the curated semantic manifest."""
    if not model_path.is_file():
        raise EvidenceValidationError(
            f"Evidence reranker artifact not found: {model_path}. Retrain first."
        )
    if not training_tasks or not training_pairs:
        raise EvidenceValidationError(
            "Current training tasks and pairs are required for artifact freshness checks."
        )
    manifest = load_manifest(manifest_path)
    cases = validate_semantic_manifest(manifest)
    try:
        artifact = validate_evidence_artifact(
            joblib.load(model_path),
            tasks=training_tasks,
            pairs=training_pairs,
        )
    except EvidenceArtifactError as error:
        raise EvidenceValidationError(str(error)) from error
    model = artifact["model"]
    report = evaluate_reranker_cases(
        cases,
        predict=model.predict_proba,
        threshold=float(artifact["threshold"]),
        training_pairs=training_pairs,
    )
    report.update(
        {
            "dataset_id": manifest.get("dataset_id", ""),
            "model_type": artifact["model_type"],
            "model_version": artifact.get("metadata", {}).get("model_version"),
        }
    )
    return report
