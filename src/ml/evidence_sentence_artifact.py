"""Contracts and inference restoration for the frozen two-stage artifact."""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ml.evidence_multiclass import SUPPORT_CLASSES
from ml.evidence_sentence_embedding import FrozenSentenceEmbeddingClassifier


ARTIFACT_SCHEMA_VERSION = 1
ARTIFACT_MODEL_TYPE = "frozen_sentence_embedding_two_stage"


class SentenceArtifactError(ValueError):
    """Raised when a frozen sentence artifact is incompatible or stale."""


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of one local file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_sentence_artifact(
    artifact: Any,
    *,
    candidate_spec_path: Path,
    training_dir: Path,
) -> dict[str, Any]:
    """Validate schema, candidate commitment, and training-corpus identity."""
    if not isinstance(artifact, dict):
        raise SentenceArtifactError("Sentence artifact must be an object.")
    if int(artifact.get("schema_version", 0)) != ARTIFACT_SCHEMA_VERSION:
        raise SentenceArtifactError("Unsupported sentence artifact schema.")
    if artifact.get("model_type") != ARTIFACT_MODEL_TYPE:
        raise SentenceArtifactError("Unexpected sentence artifact model type.")
    metadata = artifact.get("metadata")
    if not isinstance(metadata, dict):
        raise SentenceArtifactError("Sentence artifact metadata is missing.")
    if metadata.get("candidate_spec_sha256") != file_sha256(candidate_spec_path):
        raise SentenceArtifactError("Candidate specification commitment mismatch.")
    expected_training = {
        "annotated_tasks_sha256": file_sha256(
            training_dir / "annotated_tasks.jsonl"
        ),
        "training_pairs_sha256": file_sha256(
            training_dir / "training_pairs.jsonl"
        ),
    }
    if metadata.get("training") != expected_training:
        raise SentenceArtifactError("Training-corpus commitment mismatch.")
    for key in ("pure_classifier", "hybrid_classifier", "hybrid_word_scorer"):
        if artifact.get(key) is None:
            raise SentenceArtifactError(f"Sentence artifact lacks {key}.")
    if not callable(getattr(artifact["pure_classifier"], "predict_proba", None)):
        raise SentenceArtifactError("Pure classifier is invalid.")
    if not callable(getattr(artifact["hybrid_classifier"], "predict_proba", None)):
        raise SentenceArtifactError("Hybrid classifier is invalid.")
    if not callable(getattr(artifact["hybrid_word_scorer"], "score", None)):
        raise SentenceArtifactError("Hybrid word scorer is invalid.")
    return artifact


def restore_two_stage_models(
    artifact: dict[str, Any],
    encoder: Any,
) -> tuple[FrozenSentenceEmbeddingClassifier, FrozenSentenceEmbeddingClassifier]:
    """Restore fitted heads around a separately loaded pinned encoder."""
    random_state = int(artifact["metadata"]["random_state"])
    pure = FrozenSentenceEmbeddingClassifier(
        encoder,
        random_state=random_state,
    )
    pure.classifier = artifact["pure_classifier"]
    pure.is_fitted = True
    hybrid = FrozenSentenceEmbeddingClassifier(
        encoder,
        include_transparent_features=True,
        random_state=random_state,
    )
    hybrid.classifier = artifact["hybrid_classifier"]
    hybrid.word = artifact["hybrid_word_scorer"]
    hybrid.is_fitted = True
    return pure, hybrid


def predict_two_stage(
    artifact: dict[str, Any],
    encoder: Any,
    requirements: Sequence[str],
    evidence: Sequence[str],
) -> dict[str, Any]:
    """Return frozen acceptance labels and hybrid ranking scores."""
    pure, hybrid = restore_two_stage_models(artifact, encoder)
    pure_probabilities = np.asarray(
        pure.predict_class_proba(requirements, evidence),
        dtype=np.float64,
    )
    hybrid_probabilities = np.asarray(
        hybrid.predict_class_proba(requirements, evidence),
        dtype=np.float64,
    )
    predictions = np.asarray(SUPPORT_CLASSES)[
        np.argmax(pure_probabilities, axis=1)
    ]
    support_scores = (
        hybrid_probabilities[:, 0] + 0.5 * hybrid_probabilities[:, 1]
    )
    return {
        "predictions": [str(value) for value in predictions],
        "pure_probabilities": pure_probabilities.tolist(),
        "hybrid_support_scores": support_scores.astype(float).tolist(),
    }
