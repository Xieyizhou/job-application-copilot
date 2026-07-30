from __future__ import annotations

from pathlib import Path
import sys

import joblib
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_artifact import (
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    training_corpus_fingerprint,
)
from ml.evidence_validation import (
    EvidenceValidationError,
    evaluate_reranker_artifact,
    evaluate_reranker_cases,
)


CASES = [
    {
        "case_id": "case-positive",
        "requirement": "Build SQL data pipelines.",
        "evidence": "Automated recurring SQL ETL workflows.",
        "expected_accepted": True,
        "review_tag": "direct",
    },
    {
        "case_id": "case-negative",
        "requirement": "Deploy containerized services.",
        "evidence": "Prepared monthly accounting reports.",
        "expected_accepted": False,
        "review_tag": "unrelated",
    },
]
TRAINING_TASKS = [
    {
        "task_id": "training-task",
        "requirement": "Prepare quarterly financial forecasts.",
        "support_label": "Direct",
        "selected_candidate_id": "training-candidate",
        "evaluation_group": "semantic:forecasting",
        "candidates": [
            {
                "candidate_id": "training-candidate",
                "evidence": "Owned quarterly revenue forecasting models.",
            }
        ],
    }
]
TRAINING_PAIRS = [
    {
        "pair_id": "training-pair",
        "task_id": "training-task",
        "requirement": "Prepare quarterly financial forecasts.",
        "evidence": "Owned quarterly revenue forecasting models.",
        "binary_label": 1,
        "support_label": "Direct",
        "evaluation_group": "semantic:forecasting",
    }
]


class _ConstantModel:
    def predict_proba(
        self,
        requirements: list[str],
        evidence: list[str],
    ) -> list[float]:
        return [0.1] * len(requirements)


def test_external_diagnostic_reports_metrics_without_promotion_claim() -> None:
    report = evaluate_reranker_cases(
        CASES,
        predict=lambda requirements, evidence: [0.9, 0.1],
        threshold=0.5,
    )

    assert report["metrics"]["f1"] == 1.0
    assert report["promotion_eligible"] is False
    assert report["exact_training_overlap"] == 0


def test_external_diagnostic_rejects_exact_training_overlap() -> None:
    with pytest.raises(EvidenceValidationError, match="overlap"):
        evaluate_reranker_cases(
            CASES,
            predict=lambda requirements, evidence: [0.9, 0.1],
            threshold=0.5,
            training_pairs=[CASES[0]],
        )


def test_artifact_diagnostic_accepts_current_hybrid_and_rejects_legacy(
    tmp_path: Path,
) -> None:
    fingerprint = training_corpus_fingerprint(TRAINING_TASKS, TRAINING_PAIRS)
    artifact = {
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "model_type": "hybrid_lsa_reranker",
        "model": _ConstantModel(),
        "threshold": 0.5,
        "metadata": {
            "model_version": "test-hybrid",
            "selection": {"selected_method": "hybrid_lsa_reranker"},
            "training_corpus_fingerprint": fingerprint,
        },
    }
    model_path = tmp_path / "model.joblib"
    joblib.dump(artifact, model_path)

    report = evaluate_reranker_artifact(
        model_path=model_path,
        training_tasks=TRAINING_TASKS,
        training_pairs=TRAINING_PAIRS,
    )
    assert report["model_type"] == "hybrid_lsa_reranker"

    artifact["schema_version"] = 1
    joblib.dump(artifact, model_path)
    with pytest.raises(EvidenceValidationError, match="Legacy"):
        evaluate_reranker_artifact(
            model_path=model_path,
            training_tasks=TRAINING_TASKS,
            training_pairs=TRAINING_PAIRS,
        )
