from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_artifact import (
    EVIDENCE_ARTIFACT_SCHEMA_VERSION,
    EvidenceArtifactError,
    training_corpus_fingerprint,
    validate_evidence_artifact,
)


class _Predictor:
    def predict_proba(
        self,
        requirements: list[str],
        evidence: list[str],
    ) -> list[float]:
        return [0.5] * len(requirements)


TASKS = [
    {
        "task_id": "task-1",
        "requirement": "Build recurring SQL reports.",
        "support_label": "Direct",
        "selected_candidate_id": "candidate-a",
        "evaluation_group": "semantic:sql-1",
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "evidence": "Automated monthly SQL reports.",
            },
            {
                "candidate_id": "candidate-b",
                "evidence": "Prepared meeting notes.",
            },
        ],
    }
]
PAIRS = [
    {
        "pair_id": "pair-1",
        "task_id": "task-1",
        "requirement": "Build recurring SQL reports.",
        "evidence": "Automated monthly SQL reports.",
        "binary_label": 1,
        "support_label": "Direct",
        "evaluation_group": "semantic:sql-1",
    }
]


def _artifact() -> dict[str, object]:
    return {
        "schema_version": EVIDENCE_ARTIFACT_SCHEMA_VERSION,
        "model_type": "hybrid_lsa_reranker",
        "model": _Predictor(),
        "threshold": 0.5,
        "metadata": {
            "selection": {"selected_method": "hybrid_lsa_reranker"},
            "training_corpus_fingerprint": training_corpus_fingerprint(
                TASKS,
                PAIRS,
            ),
        },
    }


def test_current_artifact_matches_its_training_corpus() -> None:
    artifact = _artifact()

    assert (
        validate_evidence_artifact(artifact, tasks=TASKS, pairs=PAIRS)
        is artifact
    )


def test_artifact_rejects_legacy_schema_and_stale_training_data() -> None:
    legacy = _artifact()
    legacy["schema_version"] = 1
    with pytest.raises(EvidenceArtifactError, match="Legacy"):
        validate_evidence_artifact(legacy, tasks=TASKS, pairs=PAIRS)

    changed_pairs = deepcopy(PAIRS)
    changed_pairs[0]["binary_label"] = 0
    with pytest.raises(EvidenceArtifactError, match="different corpus"):
        validate_evidence_artifact(
            _artifact(),
            tasks=TASKS,
            pairs=changed_pairs,
        )


def test_artifact_model_type_must_match_recorded_selection() -> None:
    artifact = _artifact()
    artifact["model_type"] = "pairwise_hybrid_reranker"

    with pytest.raises(EvidenceArtifactError, match="recorded model selection"):
        validate_evidence_artifact(artifact, tasks=TASKS, pairs=PAIRS)
