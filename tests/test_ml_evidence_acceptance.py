"""Tests for the constraint-aware support classifier."""

from __future__ import annotations

import pytest

from ml.evidence_acceptance import (
    ACCEPTANCE_FEATURE_NAMES,
    ConstraintAwareSupportClassifier,
)


def _training_rows() -> tuple[list[str], list[str], list[int]]:
    requirements = [
        "Use Spring Boot",
        "Use Spring Boot",
        "Manage candidates in an ATS",
        "Manage candidates in an ATS",
        "Build data pipelines",
        "Build data pipelines",
    ]
    evidence = [
        "Built Spring Boot services for production.",
        "Implemented ETL workflows with Airflow.",
        "Managed recruiting workflows in an ATS.",
        "Worked with product marketing.",
        "Built and monitored ETL data pipelines.",
        "Created sales presentations.",
    ]
    return requirements, evidence, [1, 0, 1, 0, 1, 0]


def test_constraint_classifier_exposes_features_and_probabilities() -> None:
    requirements, evidence, labels = _training_rows()
    model = ConstraintAwareSupportClassifier(max_components=4).fit(
        requirements,
        evidence,
        labels,
    )

    probabilities = model.predict_proba(requirements, evidence)
    manifest = model.feature_manifest()

    assert len(probabilities) == len(labels)
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert set(manifest["features"]) == set(ACCEPTANCE_FEATURE_NAMES)


def test_constraint_classifier_rejects_unfitted_inference() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        ConstraintAwareSupportClassifier().predict_proba(
            ["Use SQL"],
            ["Built SQL reports"],
        )
