"""Tests for the frozen sentence-embedding shadow artifact contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from ml.evidence_sentence_artifact import (
    ARTIFACT_MODEL_TYPE,
    ARTIFACT_SCHEMA_VERSION,
    SentenceArtifactError,
    file_sha256,
    validate_sentence_artifact,
)


class _Classifier:
    def predict_proba(self, values: object) -> list[list[float]]:
        return [[1.0, 0.0, 0.0]]


class _WordScorer:
    def score(self, left: object, right: object) -> list[float]:
        return [0.0]


def _artifact(spec_path: Path, training_dir: Path) -> dict[str, object]:
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "model_type": ARTIFACT_MODEL_TYPE,
        "pure_classifier": _Classifier(),
        "hybrid_classifier": _Classifier(),
        "hybrid_word_scorer": _WordScorer(),
        "metadata": {
            "candidate_spec_sha256": file_sha256(spec_path),
            "training": {
                "annotated_tasks_sha256": file_sha256(
                    training_dir / "annotated_tasks.jsonl"
                ),
                "training_pairs_sha256": file_sha256(
                    training_dir / "training_pairs.jsonl"
                ),
            },
        },
    }


def test_artifact_rejects_changed_candidate_or_training(
    tmp_path: Path,
) -> None:
    spec = tmp_path / "spec.json"
    spec.write_text("{}", encoding="utf-8")
    training = tmp_path / "training"
    training.mkdir()
    (training / "annotated_tasks.jsonl").write_text("{}\n", encoding="utf-8")
    (training / "training_pairs.jsonl").write_text("{}\n", encoding="utf-8")
    artifact = _artifact(spec, training)

    assert (
        validate_sentence_artifact(
            artifact,
            candidate_spec_path=spec,
            training_dir=training,
        )
        is artifact
    )
    spec.write_text('{"changed": true}', encoding="utf-8")
    with pytest.raises(SentenceArtifactError, match="specification"):
        validate_sentence_artifact(
            artifact,
            candidate_spec_path=spec,
            training_dir=training,
        )
