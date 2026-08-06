"""Tests for text-free MiniLM support and strength diagnostics."""

from __future__ import annotations

import pytest

from ml.evidence_multitask_diagnostics import (
    diagnostic_findings,
    distribution_summary,
    grouped_strength_confusion,
    grouped_support_distributions,
    strength_confusion,
    support_distribution_diagnostic,
)


def test_support_diagnostic_separates_prior_shift_from_auc() -> None:
    labels = ["Direct", "Partial", "No Support", "No Support"]
    report = support_distribution_diagnostic(
        labels,
        [0.8, 0.7, 0.6, 0.4],
        [0.4, 0.3, 0.2, 0.1],
    )

    assert report["raw"]["binary_support_auc"] == 1.0
    assert report["corrected"]["binary_support_auc"] == 1.0
    assert report["raw"]["predicted_supported_rate_at_0_50"] == 0.75
    assert report["corrected"]["predicted_supported_rate_at_0_50"] == 0.0
    assert report["raw"]["supported_minus_no_support_median"] > 0.0


def test_strength_confusion_excludes_no_support_rows() -> None:
    report = strength_confusion(
        ["Direct", "Partial", "No Support", "Direct"],
        [0.2, 0.1, 0.9, 0.8],
    )

    assert report["rows"] == 3
    assert report["labels"] == ["Partial", "Direct"]
    assert report["confusion_matrix"] == [[1, 0], [1, 1]]
    assert report["direct_recall"] == 0.5
    assert report["partial_recall"] == 1.0


def test_grouped_reports_skip_groups_missing_both_binary_classes() -> None:
    groups = ["Data", "Data", "ML", "ML", "Software"]
    labels = ["Direct", "No Support", "Partial", "Direct", "No Support"]
    raw = [0.8, 0.2, 0.6, 0.7, 0.1]
    corrected = [0.4, 0.1, 0.3, 0.4, 0.05]

    support = grouped_support_distributions(
        groups,
        labels,
        raw,
        corrected,
    )
    strength = grouped_strength_confusion(groups, labels, raw)

    assert list(support) == ["Data"]
    assert list(strength) == ["ML"]


def test_findings_do_not_claim_encoder_or_data_attribution() -> None:
    support = support_distribution_diagnostic(
        ["Direct", "Partial", "No Support", "No Support"],
        [0.55, 0.52, 0.51, 0.50],
        [0.30, 0.28, 0.27, 0.26],
    )
    strength = strength_confusion(
        ["Direct", "Partial", "No Support", "Direct"],
        [0.2, 0.1, 0.9, 0.3],
    )
    findings = diagnostic_findings(support, strength)

    assert findings["flags"][
        "prior_correction_zeroed_all_candidate_acceptance"
    ] is True
    assert findings["flags"]["strength_direct_recall_below_0_50"] is True
    assert findings["encoder_objective_data_attribution_resolved"] is False
    assert findings["model_selection_allowed"] is False


def test_diagnostics_reject_invalid_or_misaligned_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        distribution_summary([1.0, float("nan")])
    with pytest.raises(ValueError, match="align"):
        support_distribution_diagnostic(
            ["Direct", "No Support"],
            [0.5],
            [0.4],
        )
    with pytest.raises(ValueError, match="Direct and Partial"):
        strength_confusion(["Direct", "No Support"], [0.8, 0.2])
