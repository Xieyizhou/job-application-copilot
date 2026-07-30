"""Three-class evidence support model for local development experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.evidence import score_transparent_evidence_pair
from ml.evidence_models import LsaEmbeddingScorer, WordTfidfCosineScorer
from ml.features import FEATURE_NAMES, pair_feature_matrix


SUPPORT_CLASSES = ("Direct", "Partial", "No Support")
TRANSPARENT_FEATURE_NAMES = (
    "transparent_similarity",
    "has_overlap",
    "numeric_constraint_supported",
    "compound_requirement_supported",
)


class PairTextTfidfClassifier:
    """Three-class word/character TF-IDF baseline over aligned text pairs."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.vectorizer = FeatureUnion(
            [
                (
                    "word",
                    TfidfVectorizer(
                        lowercase=True,
                        strip_accents="unicode",
                        ngram_range=(1, 2),
                        min_df=1,
                        sublinear_tf=True,
                    ),
                ),
                (
                    "character",
                    TfidfVectorizer(
                        analyzer="char_wb",
                        lowercase=True,
                        ngram_range=(3, 5),
                        min_df=2,
                        sublinear_tf=True,
                        max_features=16_000,
                    ),
                ),
            ]
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=700,
            random_state=random_state,
            solver="lbfgs",
        )
        self.is_fitted = False

    @staticmethod
    def _pair_text(
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> list[str]:
        MulticlassEvidenceReranker._validate(requirements, evidence)
        return [
            f"requirement {requirement} evidence {candidate}"
            for requirement, candidate in zip(
                requirements,
                evidence,
                strict=True,
            )
        ]

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[str],
    ) -> "PairTextTfidfClassifier":
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        if set(labels) != set(SUPPORT_CLASSES):
            raise ValueError(
                f"training labels must contain exactly {sorted(SUPPORT_CLASSES)}"
            )
        matrix = self.vectorizer.fit_transform(
            self._pair_text(requirements, evidence)
        )
        self.classifier.fit(matrix, labels)
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
        """Return probabilities in the stable SUPPORT_CLASSES order."""
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        matrix = self.vectorizer.transform(
            self._pair_text(requirements, evidence)
        )
        raw = self.classifier.predict_proba(matrix)
        class_positions = {
            str(label): index
            for index, label in enumerate(self.classifier.classes_)
        }
        return np.column_stack(
            [raw[:, class_positions[label]] for label in SUPPORT_CLASSES]
        )

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "pair_text_tfidf_multiclass",
            "classes": list(SUPPORT_CLASSES),
            "features": int(
                sum(
                    len(transformer.get_feature_names_out())
                    for _, transformer in self.vectorizer.transformer_list
                )
            ),
        }


class MulticlassEvidenceReranker:
    """Estimate Direct, Partial, and No Support probabilities for aligned pairs."""

    FEATURE_NAMES = (
        "word_tfidf_cosine",
        "lsa_embedding_cosine",
        *FEATURE_NAMES,
        *TRANSPARENT_FEATURE_NAMES,
    )

    def __init__(self, *, max_components: int = 96, random_state: int = 42) -> None:
        self.random_state = random_state
        self.word = WordTfidfCosineScorer()
        self.embedding = LsaEmbeddingScorer(
            max_components=max_components,
            random_state=random_state,
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=700,
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

    def _features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        self._validate(requirements, evidence)
        lexical = self.word.score(requirements, evidence)
        latent = self.embedding.score(requirements, evidence)
        overlap = np.asarray(
            pair_feature_matrix(evidence, requirements),
            dtype=np.float64,
        )
        transparent_rows = []
        for requirement, candidate in zip(requirements, evidence, strict=True):
            score = score_transparent_evidence_pair(requirement, candidate)
            transparent_rows.append(
                [
                    float(score["similarity"]),
                    float(bool(score["shared_terms"] or score["shared_concepts"])),
                    float(bool(score["numeric_constraint_supported"])),
                    float(bool(score["compound_requirement_supported"])),
                ]
            )
        transparent = np.asarray(transparent_rows, dtype=np.float64)
        return np.column_stack([lexical, latent, overlap, transparent])

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[str],
    ) -> "MulticlassEvidenceReranker":
        """Fit only on explicitly labeled reviewed pairs."""
        self._validate(requirements, evidence)
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        observed = set(labels)
        expected = set(SUPPORT_CLASSES)
        if observed != expected:
            raise ValueError(
                f"training labels must contain exactly {sorted(expected)}"
            )
        self.word.fit(requirements, evidence)
        self.embedding.fit(requirements, evidence)
        self.classifier.fit(self._features(requirements, evidence), labels)
        self.is_fitted = True
        return self

    def predict_class_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        """Return probabilities in the stable SUPPORT_CLASSES order."""
        self._validate(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        raw = self.classifier.predict_proba(self._features(requirements, evidence))
        class_positions = {
            str(label): index
            for index, label in enumerate(self.classifier.classes_)
        }
        return np.column_stack(
            [raw[:, class_positions[label]] for label in SUPPORT_CLASSES]
        )

    def predict(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        """Return the highest-probability support class."""
        probabilities = self.predict_class_proba(requirements, evidence)
        return np.asarray(SUPPORT_CLASSES)[np.argmax(probabilities, axis=1)]

    def feature_manifest(self) -> dict[str, Any]:
        """Describe the local model without retaining source text."""
        coefficients = {
            label: {
                feature: float(value)
                for feature, value in zip(
                    self.FEATURE_NAMES,
                    self.classifier.coef_[index],
                    strict=True,
                )
            }
            for index, label in enumerate(self.classifier.classes_)
        }
        return {
            "model_type": "multiclass_evidence_reranker",
            "classes": list(SUPPORT_CLASSES),
            "features": list(self.FEATURE_NAMES),
            "embedding": self.embedding.feature_manifest(),
            "coefficients": coefficients,
        }
