"""Constraint-aware classifier for accepting a ranker's top evidence candidate."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evidence import score_transparent_evidence_pair
from ml.evidence_constraints import (
    CONSTRAINT_FEATURE_NAMES,
    constraint_feature_vector,
)
from ml.evidence_models import LsaEmbeddingScorer, WordTfidfCosineScorer
from ml.features import FEATURE_NAMES, pair_feature_matrix


ACCEPTANCE_FEATURE_NAMES = (
    "word_tfidf_cosine",
    "lsa_embedding_cosine",
    *FEATURE_NAMES,
    "transparent_similarity",
    "has_lexical_or_concept_overlap",
    "shared_term_count",
    "shared_concept_count",
    *CONSTRAINT_FEATURE_NAMES,
)


class ConstraintAwareSupportClassifier:
    """Estimate support while exposing the constraint features used."""

    def __init__(self, *, max_components: int = 96, random_state: int = 42) -> None:
        self.random_state = random_state
        self.word = WordTfidfCosineScorer()
        self.embedding = LsaEmbeddingScorer(
            max_components=max_components,
            random_state=random_state,
        )
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=900,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
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

    def _features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        self._validate(requirements, evidence)
        word_scores = self.word.score(requirements, evidence)
        embedding_scores = self.embedding.score(requirements, evidence)
        lexical = np.asarray(
            pair_feature_matrix(evidence, requirements),
            dtype=np.float64,
        )
        transparent_rows: list[list[float]] = []
        constraint_rows: list[list[float]] = []
        for requirement, candidate in zip(requirements, evidence, strict=True):
            score = score_transparent_evidence_pair(requirement, candidate)
            transparent_rows.append(
                [
                    float(score["similarity"]),
                    float(bool(score["shared_terms"] or score["shared_concepts"])),
                    float(len(score["shared_terms"])),
                    float(len(score["shared_concepts"])),
                ]
            )
            constraint_rows.append(
                constraint_feature_vector(requirement, candidate)
            )
        return np.column_stack(
            [
                word_scores,
                embedding_scores,
                lexical,
                np.asarray(transparent_rows, dtype=np.float64),
                np.asarray(constraint_rows, dtype=np.float64),
            ]
        )

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[int],
    ) -> "ConstraintAwareSupportClassifier":
        """Fit on explicitly reviewed positive and negative evidence pairs."""
        self._validate(requirements, evidence)
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        if set(int(label) for label in labels) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")
        self.word.fit(requirements, evidence)
        self.embedding.fit(requirements, evidence)
        self.pipeline.fit(self._features(requirements, evidence), labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        """Return aligned probabilities that each pair offers usable support."""
        self._validate(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("constraint-aware support classifier is not fitted")
        return self.pipeline.predict_proba(
            self._features(requirements, evidence)
        )[:, 1]

    def feature_manifest(self) -> dict[str, Any]:
        """Expose standardized logistic coefficients and preprocessing metadata."""
        if not self.is_fitted:
            raise RuntimeError("constraint-aware support classifier is not fitted")
        classifier = self.pipeline.named_steps["classifier"]
        scaler = self.pipeline.named_steps["scaler"]
        return {
            "model_type": "constraint_aware_support_logistic",
            "features": {
                name: float(value)
                for name, value in zip(
                    ACCEPTANCE_FEATURE_NAMES,
                    classifier.coef_[0],
                    strict=True,
                )
            },
            "feature_means": {
                name: float(value)
                for name, value in zip(
                    ACCEPTANCE_FEATURE_NAMES,
                    scaler.mean_,
                    strict=True,
                )
            },
            "intercept": float(classifier.intercept_[0]),
            "word": self.word.feature_manifest(),
            "embedding": self.embedding.feature_manifest(),
        }
