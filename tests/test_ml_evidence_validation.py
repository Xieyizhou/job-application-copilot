from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_validation import EvidenceValidationError, evaluate_reranker_cases


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
