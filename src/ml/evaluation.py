"""Evaluation helpers for leakage-aware relevance experiments."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def select_f1_threshold(labels: Sequence[int], probabilities: Sequence[float]) -> float:
    """Select the validation threshold with the highest F1 score."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    if not len(thresholds):
        return 0.5
    f1_values = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.argmax(f1_values))])


def classification_metrics(
    labels: Sequence[int],
    probabilities: Sequence[float],
    *,
    threshold: float,
) -> dict[str, object]:
    """Return JSON-safe binary classification metrics."""
    predictions = np.asarray(probabilities) >= threshold
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "examples": len(labels),
        "positive_rate": float(np.mean(labels)),
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "average_precision": float(average_precision_score(labels, probabilities)),
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def mean_job_average_precision(
    job_ids: Sequence[str],
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Measure ranking quality within each sampled unseen-job candidate pool."""
    grouped: defaultdict[str, list[tuple[int, float]]] = defaultdict(list)
    for job_id, label, probability in zip(job_ids, labels, probabilities):
        grouped[str(job_id)].append((int(label), float(probability)))
    scores = [
        average_precision_score(
            [label for label, _ in values],
            [probability for _, probability in values],
        )
        for values in grouped.values()
        if len({label for label, _ in values}) > 1
    ]
    return float(np.mean(scores)) if scores else 0.0
