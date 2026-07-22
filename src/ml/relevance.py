"""Trainable local resume/job relevance model.

The classifier only sees shared TF-IDF terms and pair-level overlap features.
This prevents it from learning that a resume or job is globally "good" in
isolation, which is a common leakage-prone baseline for pair matching.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from ml.features import FEATURE_NAMES, pair_feature_matrix


MODEL_SCHEMA_VERSION = 1


class PairRelevanceModel:
    """Sparse, interpretable binary relevance model for resume/job pairs."""

    def __init__(self, *, max_features: int = 20_000, random_state: int = 42) -> None:
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=2,
            max_features=max_features,
            sublinear_tf=True,
            norm="l2",
        )
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
            solver="liblinear",
        )
        self.is_fitted = False

    @staticmethod
    def _validate_inputs(resume_texts: Sequence[str], job_texts: Sequence[str]) -> None:
        if len(resume_texts) != len(job_texts):
            raise ValueError("resume_texts and job_texts must have equal lengths")
        if not resume_texts:
            raise ValueError("at least one resume/job pair is required")
        if any(not str(text).strip() for text in [*resume_texts, *job_texts]):
            raise ValueError("resume and job texts must be non-empty")

    def _transform(self, resume_texts: Sequence[str], job_texts: Sequence[str]):
        resume_matrix = self.vectorizer.transform(resume_texts)
        job_matrix = self.vectorizer.transform(job_texts)
        shared_terms = resume_matrix.multiply(job_matrix)
        numeric = csr_matrix(pair_feature_matrix(resume_texts, job_texts))
        return hstack([shared_terms, numeric], format="csr")

    def fit(
        self,
        resume_texts: Sequence[str],
        job_texts: Sequence[str],
        labels: Sequence[int],
    ) -> "PairRelevanceModel":
        """Fit the vectorizer and binary classifier."""
        self._validate_inputs(resume_texts, job_texts)
        if len(labels) != len(resume_texts):
            raise ValueError("labels must align with resume/job pairs")
        if set(int(value) for value in labels) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")
        corpus = list(dict.fromkeys([*resume_texts, *job_texts]))
        self.vectorizer.fit(corpus)
        self.classifier.fit(self._transform(resume_texts, job_texts), labels)
        self.is_fitted = True
        return self

    def predict_proba(self, resume_texts: Sequence[str], job_texts: Sequence[str]) -> np.ndarray:
        """Return the positive-class relevance probability for each pair."""
        self._validate_inputs(resume_texts, job_texts)
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        return self.classifier.predict_proba(self._transform(resume_texts, job_texts))[:, 1]

    def feature_manifest(self) -> dict[str, object]:
        """Return non-sensitive model feature metadata."""
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "shared_tfidf_features": len(self.vectorizer.get_feature_names_out()),
            "numeric_features": list(FEATURE_NAMES),
        }

    def export_portable(self, *, threshold: float, metadata: dict[str, object]) -> dict[str, object]:
        """Export a JSON-safe artifact that does not require sklearn at runtime."""
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        vocabulary = {
            term: int(index)
            for term, index in self.vectorizer.vocabulary_.items()
        }
        return {
            "schema_version": MODEL_SCHEMA_VERSION,
            "model_type": "portable_pair_relevance",
            "threshold": float(threshold),
            "metadata": metadata,
            "vectorizer": {
                "vocabulary": vocabulary,
                "idf": [float(value) for value in self.vectorizer.idf_],
                "ngram_range": [1, 2],
                "sublinear_tf": True,
                "norm": "l2",
                "strip_accents": "unicode",
                "token_pattern": r"(?u)\b\w\w+\b",
            },
            "classifier": {
                "coefficients": [float(value) for value in self.classifier.coef_[0]],
                "intercept": float(self.classifier.intercept_[0]),
            },
            "numeric_features": list(FEATURE_NAMES),
        }
