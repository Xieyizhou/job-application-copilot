"""Tests for the explainable support acceptance model."""

from __future__ import annotations

import pytest

from ml.evidence_support import (
    SUPPORT_FEATURE_NAMES,
    TransparentSupportClassifier,
)


def test_support_classifier_exposes_features_and_probabilities() -> None:
    requirements = [
        "Build Python data pipelines",
        "Build Python data pipelines",
        "Create Tableau dashboards",
        "Create Tableau dashboards",
    ]
    evidence = [
        "Built Python ETL pipelines for daily reporting.",
        "Designed marketing brochures.",
        "Created Tableau dashboards for finance leaders.",
        "Maintained Java web services.",
    ]
    model = TransparentSupportClassifier().fit(
        requirements,
        evidence,
        [1, 0, 1, 0],
    )

    probabilities = model.predict_proba(requirements, evidence)
    manifest = model.feature_manifest()

    assert len(probabilities) == 4
    assert all(0.0 <= value <= 1.0 for value in probabilities)
    assert set(manifest["features"]) == set(SUPPORT_FEATURE_NAMES)


def test_support_classifier_rejects_unfitted_inference() -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        TransparentSupportClassifier().predict_proba(["SQL"], ["Used SQL"])
