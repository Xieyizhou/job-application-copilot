"""Text-free diagnostics for support calibration and strength separation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from ml.evidence_multiclass import SUPPORT_CLASSES


def _probabilities(values: Sequence[float], *, name: str) -> np.ndarray:
    probabilities = np.asarray(values, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or not len(probabilities)
        or not np.all(np.isfinite(probabilities))
        or np.any(probabilities < 0.0)
        or np.any(probabilities > 1.0)
    ):
        raise ValueError(f"{name} must be a non-empty finite probability vector.")
    return probabilities


def distribution_summary(values: Sequence[float]) -> dict[str, float | int]:
    """Return stable aggregate statistics without retaining row-level values."""
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("Distribution values must be non-empty and finite.")
    quantiles = np.quantile(array, [0.05, 0.25, 0.5, 0.75, 0.95])
    return {
        "count": int(len(array)),
        "mean": float(np.mean(array)),
        "standard_deviation": float(np.std(array)),
        "minimum": float(np.min(array)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "maximum": float(np.max(array)),
    }


def _logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(
        probabilities,
        np.finfo(np.float64).eps,
        1.0 - np.finfo(np.float64).eps,
    )
    return np.log(clipped / (1.0 - clipped))


def support_distribution_diagnostic(
    labels: Sequence[str],
    raw_probabilities: Sequence[float],
    corrected_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Compare raw and fixed-prior-corrected support without selecting a threshold."""
    raw = _probabilities(raw_probabilities, name="Raw support")
    corrected = _probabilities(corrected_probabilities, name="Corrected support")
    label_values = [str(label) for label in labels]
    if (
        len(label_values) != len(raw)
        or len(corrected) != len(raw)
        or any(label not in SUPPORT_CLASSES for label in label_values)
    ):
        raise ValueError("Support diagnostic labels and probabilities must align.")
    supported = np.asarray(
        [label != "No Support" for label in label_values],
        dtype=bool,
    )
    if set(supported.tolist()) != {False, True}:
        raise ValueError("Support diagnostic needs supported and No Support rows.")

    def report(probabilities: np.ndarray) -> dict[str, Any]:
        supported_summary = distribution_summary(probabilities[supported].tolist())
        no_support_summary = distribution_summary(
            probabilities[~supported].tolist()
        )
        return {
            "probability": distribution_summary(probabilities.tolist()),
            "logit": distribution_summary(_logits(probabilities).tolist()),
            "predicted_supported_rate_at_0_50": float(
                np.mean(probabilities >= 0.5)
            ),
            "binary_support_auc": float(
                roc_auc_score(supported.astype(np.int64), probabilities)
            ),
            "supported": supported_summary,
            "no_support": no_support_summary,
            "supported_minus_no_support_median": float(
                supported_summary["median"] - no_support_summary["median"]
            ),
        }

    return {
        "rows": len(label_values),
        "label_counts": dict(Counter(label_values)),
        "raw": report(raw),
        "corrected": report(corrected),
    }


def grouped_support_distributions(
    group_values: Sequence[str],
    labels: Sequence[str],
    raw_probabilities: Sequence[float],
    corrected_probabilities: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Report anonymous support distributions for each non-empty group."""
    raw = _probabilities(raw_probabilities, name="Raw support")
    corrected = _probabilities(corrected_probabilities, name="Corrected support")
    groups = [str(value) for value in group_values]
    label_values = [str(label) for label in labels]
    if (
        len(groups) != len(raw)
        or len(label_values) != len(raw)
        or len(corrected) != len(raw)
        or any(not value for value in groups)
    ):
        raise ValueError("Grouped support diagnostic inputs must align.")
    return {
        group: support_distribution_diagnostic(
            [label_values[index] for index in indices],
            raw[indices].tolist(),
            corrected[indices].tolist(),
        )
        for group in sorted(set(groups))
        if set(
            label_values[index] != "No Support"
            for index, value in enumerate(groups)
            if value == group
        )
        == {False, True}
        for indices in (
            np.asarray(
                [index for index, value in enumerate(groups) if value == group],
                dtype=np.int64,
            ),
        )
    }


def strength_confusion(
    labels: Sequence[str],
    direct_probabilities: Sequence[float],
) -> dict[str, Any]:
    """Evaluate Direct versus Partial only on human-supported candidates."""
    probabilities = _probabilities(
        direct_probabilities,
        name="Direct strength",
    )
    label_values = [str(label) for label in labels]
    if len(label_values) != len(probabilities) or any(
        label not in SUPPORT_CLASSES for label in label_values
    ):
        raise ValueError("Strength labels and probabilities must align.")
    supported_indices = np.asarray(
        [
            index
            for index, label in enumerate(label_values)
            if label in {"Direct", "Partial"}
        ],
        dtype=np.int64,
    )
    supported_labels = [label_values[index] for index in supported_indices]
    if set(supported_labels) != {"Direct", "Partial"}:
        raise ValueError("Strength diagnostic needs Direct and Partial rows.")
    predicted = [
        "Direct" if probabilities[index] >= 0.5 else "Partial"
        for index in supported_indices
    ]
    matrix = [
        [
            sum(
                actual == row_label and guess == column_label
                for actual, guess in zip(
                    supported_labels,
                    predicted,
                    strict=True,
                )
            )
            for column_label in ("Partial", "Direct")
        ]
        for row_label in ("Partial", "Direct")
    ]
    partial_total = sum(matrix[0])
    direct_total = sum(matrix[1])
    return {
        "rows": len(supported_indices),
        "labels": ["Partial", "Direct"],
        "label_counts": dict(Counter(supported_labels)),
        "prediction_counts": dict(Counter(predicted)),
        "confusion_matrix": matrix,
        "partial_recall": matrix[0][0] / partial_total,
        "direct_recall": matrix[1][1] / direct_total,
        "direct_probability": {
            "overall": distribution_summary(
                probabilities[supported_indices].tolist()
            ),
            "Partial": distribution_summary(
                [
                    probabilities[index]
                    for index in supported_indices
                    if label_values[index] == "Partial"
                ]
            ),
            "Direct": distribution_summary(
                [
                    probabilities[index]
                    for index in supported_indices
                    if label_values[index] == "Direct"
                ]
            ),
        },
    }


def grouped_strength_confusion(
    group_values: Sequence[str],
    labels: Sequence[str],
    direct_probabilities: Sequence[float],
) -> dict[str, dict[str, Any]]:
    """Return strength confusion for groups containing both supported classes."""
    probabilities = _probabilities(
        direct_probabilities,
        name="Direct strength",
    )
    groups = [str(value) for value in group_values]
    label_values = [str(label) for label in labels]
    if (
        len(groups) != len(probabilities)
        or len(label_values) != len(probabilities)
        or any(not value for value in groups)
    ):
        raise ValueError("Grouped strength diagnostic inputs must align.")
    result: dict[str, dict[str, Any]] = {}
    for group in sorted(set(groups)):
        indices = [
            index
            for index, value in enumerate(groups)
            if value == group
        ]
        group_labels = [label_values[index] for index in indices]
        if {"Direct", "Partial"}.issubset(group_labels):
            result[group] = strength_confusion(
                group_labels,
                probabilities[indices].tolist(),
            )
    return result


def diagnostic_findings(
    support: Mapping[str, Any],
    strength: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate aggregate measurements into bounded diagnostic findings."""
    raw = support["raw"]
    corrected = support["corrected"]
    raw_positive = float(raw["predicted_supported_rate_at_0_50"])
    corrected_positive = float(
        corrected["predicted_supported_rate_at_0_50"]
    )
    raw_auc = float(raw["binary_support_auc"])
    raw_median_gap = float(raw["supported_minus_no_support_median"])
    direct_recall = float(strength["direct_recall"])
    flags = {
        "prior_correction_zeroed_all_candidate_acceptance": (
            raw_positive > 0.0 and corrected_positive == 0.0
        ),
        "raw_support_auc_below_0_60": raw_auc < 0.60,
        "raw_supported_median_not_above_no_support": raw_median_gap <= 0.0,
        "strength_direct_recall_below_0_50": direct_recall < 0.50,
    }
    findings = []
    if flags["prior_correction_zeroed_all_candidate_acceptance"]:
        findings.append("fixed_prior_correction_caused_threshold_collapse")
    if (
        flags["raw_support_auc_below_0_60"]
        or flags["raw_supported_median_not_above_no_support"]
    ):
        findings.append("raw_support_separation_is_weak")
    if flags["strength_direct_recall_below_0_50"]:
        findings.append("strength_head_under_recalls_direct")
    return {
        "flags": flags,
        "findings": findings,
        "encoder_objective_data_attribution_resolved": False,
        "model_selection_allowed": False,
    }
