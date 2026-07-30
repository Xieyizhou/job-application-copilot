"""Frozen sentence-embedding classifiers for development-only evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.evidence import score_transparent_evidence_pair
from ml.evidence_multiclass import SUPPORT_CLASSES
from ml.evidence_models import WordTfidfCosineScorer
from ml.features import pair_feature_matrix


DEFAULT_SENTENCE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_SENTENCE_MODEL_REVISION = "b8903db39f65d93ae28d49a37c4f3fa90c5f94e0"


class SentenceEncoder(Protocol):
    """Minimal encoder contract used by the frozen classifiers."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> Any: ...


class CachedSentenceEncoder:
    """Cache normalized embeddings so grouped folds encode each text once."""

    def __init__(self, encoder: SentenceEncoder) -> None:
        self.encoder = encoder
        self.cache: dict[str, np.ndarray] = {}

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        normalize_embeddings: bool,
        show_progress_bar: bool,
        convert_to_numpy: bool,
    ) -> np.ndarray:
        if not normalize_embeddings or not convert_to_numpy:
            raise ValueError("cached encoder requires normalized NumPy embeddings")
        missing = list(dict.fromkeys(text for text in sentences if text not in self.cache))
        if missing:
            encoded = np.asarray(
                self.encoder.encode(
                    missing,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                ),
                dtype=np.float64,
            )
            if encoded.ndim != 2 or encoded.shape[0] != len(missing):
                raise ValueError("sentence encoder returned an invalid matrix")
            self.cache.update(
                {
                    text: row
                    for text, row in zip(missing, encoded, strict=True)
                }
            )
        return np.vstack([self.cache[text] for text in sentences])


def load_sentence_encoder(
    *,
    model_name: str = DEFAULT_SENTENCE_MODEL,
    revision: str = DEFAULT_SENTENCE_MODEL_REVISION,
    local_files_only: bool = False,
) -> SentenceEncoder:
    """Load a pinned local-or-Hugging-Face sentence encoder lazily."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as error:
        raise RuntimeError(
            "Install requirements-ml.txt to evaluate sentence embeddings."
        ) from error
    return SentenceTransformer(
        model_name,
        revision=revision,
        local_files_only=local_files_only,
    )


class FrozenSentenceEmbeddingClassifier:
    """Fit a small multiclass head while keeping sentence weights frozen."""

    def __init__(
        self,
        encoder: SentenceEncoder,
        *,
        include_transparent_features: bool = False,
        batch_size: int = 64,
        random_state: int = 42,
    ) -> None:
        self.encoder = encoder
        self.include_transparent_features = include_transparent_features
        self.batch_size = batch_size
        self.random_state = random_state
        self.word = WordTfidfCosineScorer()
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=1_000,
            random_state=random_state,
            solver="lbfgs",
        )
        self.is_fitted = False

    @staticmethod
    def _validate(
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> None:
        if len(requirements) != len(evidence):
            raise ValueError("requirements and evidence must have equal lengths")
        if not requirements:
            raise ValueError("at least one requirement/evidence pair is required")

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        embeddings = self.encoder.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        matrix = np.asarray(embeddings, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError("sentence encoder returned an invalid matrix")
        return matrix

    def _embedding_features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        left = self._encode(requirements)
        right = self._encode(evidence)
        if left.shape != right.shape:
            raise ValueError("requirement and evidence embeddings must align")
        cosine = np.sum(left * right, axis=1, keepdims=True)
        return np.column_stack(
            [
                cosine,
                np.abs(left - right),
                left * right,
            ]
        )

    def _transparent_features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        lexical = self.word.score(requirements, evidence).reshape(-1, 1)
        overlap = np.asarray(
            pair_feature_matrix(evidence, requirements),
            dtype=np.float64,
        )
        transparent_rows = []
        for requirement, candidate in zip(requirements, evidence, strict=True):
            score = score_transparent_evidence_pair(requirement, candidate)
            transparent_rows.append(
                [
                    float(score["similarity"]) / 100.0,
                    float(bool(score["numeric_constraint_supported"])),
                    float(bool(score["compound_requirement_supported"])),
                ]
            )
        return np.column_stack(
            [
                lexical,
                overlap,
                np.asarray(transparent_rows, dtype=np.float64),
            ]
        )

    def _features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        self._validate(requirements, evidence)
        embedding = self._embedding_features(requirements, evidence)
        if not self.include_transparent_features:
            return embedding
        return np.column_stack(
            [
                embedding,
                self._transparent_features(requirements, evidence),
            ]
        )

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[str],
    ) -> FrozenSentenceEmbeddingClassifier:
        """Fit only the local classification head."""
        self._validate(requirements, evidence)
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        if set(labels) != set(SUPPORT_CLASSES):
            raise ValueError(
                f"training labels must contain exactly {sorted(SUPPORT_CLASSES)}"
            )
        if self.include_transparent_features:
            self.word.fit(requirements, evidence)
        self.classifier.fit(self._features(requirements, evidence), labels)
        self.is_fitted = True
        return self

    def predict(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        probabilities = self.predict_class_proba(requirements, evidence)
        return np.asarray(SUPPORT_CLASSES)[np.argmax(probabilities, axis=1)]

    def predict_class_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        """Return probabilities in the stable support-class order."""
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        raw = self.classifier.predict_proba(
            self._features(requirements, evidence)
        )
        class_positions = {
            str(label): index
            for index, label in enumerate(self.classifier.classes_)
        }
        return np.column_stack(
            [raw[:, class_positions[label]] for label in SUPPORT_CLASSES]
        )

    def feature_manifest(self) -> dict[str, Any]:
        """Describe the frozen-encoder experiment without source text."""
        return {
            "model_type": "frozen_sentence_embedding_multiclass",
            "encoder_frozen": True,
            "include_transparent_features": self.include_transparent_features,
            "classes": list(SUPPORT_CLASSES),
            "classifier": "balanced_logistic_regression",
            "input_features": int(self.classifier.coef_.shape[1]),
        }
