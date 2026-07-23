"""Local retrieval and reranking models for requirement-to-evidence experiments."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import normalize

from ml.features import FEATURE_NAMES, pair_feature_matrix


def _validate_aligned(requirements: Sequence[str], evidence: Sequence[str]) -> None:
    if len(requirements) != len(evidence):
        raise ValueError("requirements and evidence must have equal lengths")
    if not requirements:
        raise ValueError("at least one requirement/evidence pair is required")


def _aligned_sparse_cosine(left: Any, right: Any) -> np.ndarray:
    normalized_left = normalize(left, norm="l2", copy=True)
    normalized_right = normalize(right, norm="l2", copy=True)
    return np.asarray(normalized_left.multiply(normalized_right).sum(axis=1)).ravel()


class WordTfidfCosineScorer:
    """Word unigram/bigram cosine baseline fitted on the training fold only."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> "WordTfidfCosineScorer":
        _validate_aligned(requirements, evidence)
        self.vectorizer.fit(list(dict.fromkeys([*requirements, *evidence])))
        return self

    def score(self, requirements: Sequence[str], evidence: Sequence[str]) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        requirement_matrix = self.vectorizer.transform(requirements)
        evidence_matrix = self.vectorizer.transform(evidence)
        return _aligned_sparse_cosine(requirement_matrix, evidence_matrix)

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "word_tfidf_cosine",
            "features": len(self.vectorizer.get_feature_names_out()),
            "ngram_range": [1, 2],
        }


class LsaEmbeddingScorer:
    """Word/character TF-IDF projected into a local latent semantic space."""

    def __init__(self, *, max_components: int = 96, random_state: int = 42) -> None:
        self.max_components = max_components
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
                        max_features=12_000,
                    ),
                ),
            ]
        )
        self.reducer: TruncatedSVD | None = None
        self.components = 0

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> "LsaEmbeddingScorer":
        _validate_aligned(requirements, evidence)
        corpus = list(dict.fromkeys([*requirements, *evidence]))
        matrix = self.vectorizer.fit_transform(corpus)
        components = min(
            self.max_components,
            max(matrix.shape[0] - 1, 0),
            max(matrix.shape[1] - 1, 0),
        )
        if components >= 2:
            self.reducer = TruncatedSVD(
                n_components=components,
                n_iter=7,
                random_state=self.random_state,
            )
            self.reducer.fit(matrix)
            self.components = components
        return self

    def _transform(self, texts: Sequence[str]) -> Any:
        matrix = self.vectorizer.transform(texts)
        if self.reducer is None:
            return normalize(matrix, norm="l2", copy=True)
        return normalize(self.reducer.transform(matrix), norm="l2", copy=True)

    def score(self, requirements: Sequence[str], evidence: Sequence[str]) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        requirement_matrix = self._transform(requirements)
        evidence_matrix = self._transform(evidence)
        if self.reducer is None:
            return np.asarray(
                requirement_matrix.multiply(evidence_matrix).sum(axis=1)
            ).ravel()
        return np.sum(requirement_matrix * evidence_matrix, axis=1)

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "local_lsa_embedding",
            "components": self.components,
            "word_ngrams": [1, 2],
            "character_ngrams": [3, 5],
        }


class HybridEvidenceReranker:
    """Train a calibrated lexical/latent-feature reranker on reviewed pairs."""

    FEATURE_NAMES = (
        "word_tfidf_cosine",
        "lsa_embedding_cosine",
        *FEATURE_NAMES,
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
            max_iter=500,
            random_state=random_state,
            solver="liblinear",
        )
        self.is_fitted = False

    def _features(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        lexical = self.word.score(requirements, evidence)
        latent = self.embedding.score(requirements, evidence)
        numeric = np.asarray(
            pair_feature_matrix(evidence, requirements),
            dtype=np.float64,
        )
        return np.column_stack([lexical, latent, numeric])

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[int],
    ) -> "HybridEvidenceReranker":
        _validate_aligned(requirements, evidence)
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        if set(int(label) for label in labels) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")
        self.word.fit(requirements, evidence)
        self.embedding.fit(requirements, evidence)
        self.classifier.fit(self._features(requirements, evidence), labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        return self.classifier.predict_proba(self._features(requirements, evidence))[:, 1]

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "hybrid_lsa_reranker",
            "features": list(self.FEATURE_NAMES),
            "embedding": self.embedding.feature_manifest(),
        }


class PairwiseHybridReranker:
    """Blend binary support confidence with human strongest-evidence preferences."""

    def __init__(
        self,
        *,
        ranking_weight: float = 0.35,
        max_components: int = 96,
        random_state: int = 42,
    ) -> None:
        if not 0.0 <= ranking_weight <= 1.0:
            raise ValueError("ranking_weight must be between zero and one")
        self.ranking_weight = ranking_weight
        self.binary = HybridEvidenceReranker(
            max_components=max_components,
            random_state=random_state,
        )
        self.ranker = LogisticRegression(
            class_weight="balanced",
            fit_intercept=False,
            max_iter=500,
            random_state=random_state,
            solver="liblinear",
        )
        self.comparisons = 0
        self.is_fitted = False

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[int],
        tasks: Sequence[dict[str, Any]],
    ) -> "PairwiseHybridReranker":
        """Fit binary support and pairwise strongest-evidence objectives."""
        self.binary.fit(requirements, evidence, labels)
        differences: list[np.ndarray] = []
        ranking_labels: list[int] = []
        for task in tasks:
            if str(task.get("support_label")) == "No Support":
                continue
            candidates = list(task.get("candidates", []))
            candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
            selected_id = str(task.get("selected_candidate_id", ""))
            if selected_id not in candidate_ids or len(candidates) < 2:
                continue
            features = self.binary._features(
                [str(task["requirement"])] * len(candidates),
                [str(candidate["evidence"]) for candidate in candidates],
            )
            selected_index = candidate_ids.index(selected_id)
            for index in range(len(candidates)):
                if index == selected_index:
                    continue
                difference = features[selected_index] - features[index]
                differences.extend([difference, -difference])
                ranking_labels.extend([1, 0])
                self.comparisons += 1
        if not differences:
            raise ValueError("pairwise training needs supported tasks with alternatives")
        self.ranker.fit(np.vstack(differences), ranking_labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        """Return a blended support/ranking score for aligned candidates."""
        _validate_aligned(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        binary_score = self.binary.predict_proba(requirements, evidence)
        features = self.binary._features(requirements, evidence)
        ranking_logit = self.ranker.decision_function(features)
        ranking_score = 1.0 / (1.0 + np.exp(-ranking_logit))
        return (
            (1.0 - self.ranking_weight) * binary_score
            + self.ranking_weight * ranking_score
        )

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "pairwise_hybrid_reranker",
            "ranking_weight": self.ranking_weight,
            "pairwise_comparisons": self.comparisons,
            "binary_features": self.binary.feature_manifest(),
        }


class LexicalGuardedReranker:
    """Preserve strong lexical ranking while adding learned support calibration."""

    def __init__(
        self,
        *,
        lexical_weight: float = 0.70,
        max_components: int = 96,
        random_state: int = 42,
    ) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between zero and one")
        self.lexical_weight = lexical_weight
        self.hybrid = HybridEvidenceReranker(
            max_components=max_components,
            random_state=random_state,
        )
        self.is_fitted = False

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[int],
    ) -> "LexicalGuardedReranker":
        self.hybrid.fit(requirements, evidence, labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("model is not fitted")
        lexical = self.hybrid.word.score(requirements, evidence)
        support = self.hybrid.predict_proba(requirements, evidence)
        return self.lexical_weight * lexical + (1.0 - self.lexical_weight) * support

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "lexical_guarded_reranker",
            "lexical_weight": self.lexical_weight,
            "support_features": self.hybrid.feature_manifest(),
        }
