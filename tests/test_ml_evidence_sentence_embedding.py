"""Tests for development-only frozen sentence-embedding classifiers."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import pytest

from ml.evidence_multiclass import SUPPORT_CLASSES
from ml.evidence_sentence_embedding import (
    CachedSentenceEncoder,
    FrozenSentenceEmbeddingClassifier,
)


class FakeSentenceEncoder:
    """Return deterministic normalized vectors without model downloads."""

    def __init__(self) -> None:
        self.encoded_texts: list[str] = []

    def encode(
        self,
        sentences: Sequence[str],
        **_: Any,
    ) -> np.ndarray:
        self.encoded_texts.extend(sentences)
        rows = []
        for sentence in sentences:
            digest = hashlib.sha256(sentence.encode()).digest()[:8]
            row = np.frombuffer(digest, dtype=np.uint8).astype(np.float64) - 127.5
            rows.append(row / np.linalg.norm(row))
        return np.vstack(rows)


def training_rows() -> tuple[list[str], list[str], list[str]]:
    return (
        [
            "Build production data pipelines",
            "Use Python for analysis",
            "Deploy models to production",
            "Use Docker in production",
            "Manage client support requests",
            "Hold a PhD in computer science",
        ],
        [
            "Built and monitored production ETL data pipelines.",
            "Completed a guided Python tutorial.",
            "Trained models locally without deployment.",
            "Deployed a containerized service with Docker.",
            "Answered customer requests through chat and email.",
            "Built a search API using Scala.",
        ],
        ["Direct", "Partial", "Partial", "Direct", "Direct", "No Support"],
    )


@pytest.mark.parametrize("hybrid", [False, True])
def test_frozen_classifier_predicts_stable_classes(hybrid: bool) -> None:
    requirements, evidence, labels = training_rows()
    model = FrozenSentenceEmbeddingClassifier(
        CachedSentenceEncoder(FakeSentenceEncoder()),
        include_transparent_features=hybrid,
    ).fit(requirements, evidence, labels)

    predictions = model.predict(
        ["Use Docker"],
        ["Deployed a service with Docker."],
    )

    assert predictions.shape == (1,)
    assert predictions[0] in SUPPORT_CLASSES
    assert model.feature_manifest()["encoder_frozen"] is True
    assert model.feature_manifest()["include_transparent_features"] is hybrid


def test_cached_encoder_encodes_each_unique_text_once() -> None:
    source = FakeSentenceEncoder()
    cached = CachedSentenceEncoder(source)

    first = cached.encode(
        ["SQL", "Python", "SQL"],
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    second = cached.encode(
        ["Python", "Docker"],
        batch_size=8,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )

    assert first.shape == (3, 8)
    assert second.shape == (2, 8)
    assert source.encoded_texts == ["SQL", "Python", "Docker"]


def test_cached_encoder_requires_normalized_numpy_output() -> None:
    cached = CachedSentenceEncoder(FakeSentenceEncoder())

    with pytest.raises(ValueError, match="normalized"):
        cached.encode(
            ["SQL"],
            batch_size=8,
            normalize_embeddings=False,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
