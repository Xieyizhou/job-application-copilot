"""Explainable feature model for accepting a ranked resume evidence pair."""

from __future__ import annotations

from ml.evidence_metrics import _validate_aligned

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from ml.evidence import score_transparent_evidence_pair
from ml.features import FEATURE_NAMES, pair_feature_matrix


SUPPORT_FEATURE_NAMES = (
    *FEATURE_NAMES,
    "transparent_similarity",
    "has_lexical_or_concept_overlap",
    "numeric_constraint_supported",
    "compound_requirement_supported",
    "shared_term_count",
    "shared_concept_count",
)


class TransparentSupportClassifier:
    """Fit support acceptance from a small set of inspectable pair features."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = random_state
        self.classifier = LogisticRegression(
            class_weight="balanced",
            max_iter=500,
            random_state=random_state,
            solver="liblinear",
        )
        self.is_fitted = False

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
        labels: Sequence[int],
    ) -> "TransparentSupportClassifier":
        _validate_aligned(requirements, evidence)
        if len(labels) != len(requirements):
            raise ValueError("labels must align with requirement/evidence pairs")
        if set(int(label) for label in labels) != {0, 1}:
            raise ValueError("training labels must contain both 0 and 1")
        self.classifier.fit(_support_feature_matrix(requirements, evidence), labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        if not self.is_fitted:
            raise RuntimeError("support classifier is not fitted")
        features = _support_feature_matrix(requirements, evidence)
        return self.classifier.predict_proba(features)[:, 1]

    def feature_manifest(self) -> dict[str, Any]:
        """Expose every learned coefficient for local inspection."""
        if not self.is_fitted:
            raise RuntimeError("support classifier is not fitted")
        coefficients = self.classifier.coef_[0]
        return {
            "model_type": "transparent_support_logistic",
            "features": {
                name: float(coefficient)
                for name, coefficient in zip(
                    SUPPORT_FEATURE_NAMES,
                    coefficients,
                    strict=True,
                )
            },
            "intercept": float(self.classifier.intercept_[0]),
        }


def _support_feature_matrix(
    requirements: Sequence[str],
    evidence: Sequence[str],
) -> np.ndarray:
    lexical = np.asarray(
        pair_feature_matrix(evidence, requirements),
        dtype=np.float64,
    )
    transparent = []
    for requirement, candidate in zip(requirements, evidence, strict=True):
        score = score_transparent_evidence_pair(requirement, candidate)
        transparent.append(
            [
                float(score["similarity"]),
                float(bool(score["shared_terms"] or score["shared_concepts"])),
                float(bool(score["numeric_constraint_supported"])),
                float(bool(score["compound_requirement_supported"])),
                float(len(score["shared_terms"])),
                float(len(score["shared_concepts"])),
            ]
        )
    return np.column_stack(
        [
            lexical,
            np.asarray(transparent, dtype=np.float64),
        ]
    )


