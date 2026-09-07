"""LSA retrieval followed by three-class evidence reranking."""

from __future__ import annotations

from ml.evidence_metrics import _mean

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any, TypedDict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ml.evidence_constraints import named_terms
from ml.evidence_multiclass import SUPPORT_CLASSES, MulticlassEvidenceReranker


class CachedTaskPrediction(TypedDict):
    """Model outputs cached before development-parameter selection."""

    retrieval_scores: list[float]
    class_probabilities: list[list[float]]


@dataclass(frozen=True)
class TwoStageParameters:
    """Development-only ranking and acceptance parameters."""

    retrieval_weight: float
    partial_weight: float
    support_threshold: float
    top_k: int = 3


@dataclass(frozen=True)
class RankPreservingGateParameters:
    """Acceptance thresholds applied after fixed candidate ranking."""

    retrieval_threshold: float
    support_threshold: float
    named_term_rescue: bool = True


def reviewed_multiclass_rows(
    tasks: Sequence[dict[str, Any]],
) -> list[tuple[str, str, str]]:
    """Return only explicitly labeled pairs from reviewed task decisions."""
    rows: list[tuple[str, str, str]] = []
    for task in tasks:
        label = str(task["support_label"])
        candidates = list(task["candidates"])
        candidate_labels = [
            str(candidate.get("support_label", ""))
            for candidate in candidates
        ]
        if candidate_labels and all(
            candidate_label in SUPPORT_CLASSES
            for candidate_label in candidate_labels
        ):
            rows.extend(
                (
                    str(task["requirement"]),
                    str(candidate["evidence"]),
                    candidate_label,
                )
                for candidate, candidate_label in zip(
                    candidates,
                    candidate_labels,
                    strict=True,
                )
            )
            continue
        if label == "No Support":
            rows.extend(
                (
                    str(task["requirement"]),
                    str(candidate["evidence"]),
                    label,
                )
                for candidate in candidates
            )
            continue
        selected_id = str(task["selected_candidate_id"])
        selected = next(
            candidate
            for candidate in candidates
            if str(candidate["candidate_id"]) == selected_id
        )
        rows.append(
            (
                str(task["requirement"]),
                str(selected["evidence"]),
                label,
            )
        )
    return rows


def cache_two_stage_predictions(
    model: MulticlassEvidenceReranker,
    tasks: Sequence[dict[str, Any]],
    *,
    retrieval_score: (
        Callable[[Sequence[str], Sequence[str]], Iterable[float]] | None
    ) = None,
) -> dict[str, CachedTaskPrediction]:
    """Score each task once so parameter selection cannot refit the model."""
    score = retrieval_score or model.embedding.score
    cached: dict[str, CachedTaskPrediction] = {}
    for task in tasks:
        candidates = list(task["candidates"])
        requirements = [str(task["requirement"])] * len(candidates)
        evidence = [str(candidate["evidence"]) for candidate in candidates]
        cached[str(task["task_id"])] = {
            "retrieval_scores": [
                float(value)
                for value in score(requirements, evidence)
            ],
            "class_probabilities": model.predict_class_proba(
                requirements,
                evidence,
            ).tolist(),
        }
    return cached


def _scaled_retrieval(scores: Any) -> Any:
    values = np.asarray(scores, dtype=np.float64)
    spread = float(np.max(values) - np.min(values))
    if spread <= 1e-12:
        return np.zeros_like(values)
    return (values - np.min(values)) / spread


def _rank_task(
    prediction: CachedTaskPrediction,
    parameters: TwoStageParameters,
) -> tuple[list[int], str, float]:
    retrieval = np.asarray(prediction["retrieval_scores"], dtype=np.float64)
    probabilities = np.asarray(
        prediction["class_probabilities"],
        dtype=np.float64,
    )
    top_k = min(parameters.top_k, len(retrieval))
    retrieval_order = np.argsort(-retrieval, kind="stable")
    shortlist = retrieval_order[:top_k]
    support_strength = (
        probabilities[:, SUPPORT_CLASSES.index("Direct")]
        + parameters.partial_weight
        * probabilities[:, SUPPORT_CLASSES.index("Partial")]
    )
    rerank_score: Any = (
        (1.0 - parameters.retrieval_weight) * support_strength
        + parameters.retrieval_weight * _scaled_retrieval(retrieval)
    )
    shortlist_order = shortlist[
        np.argsort(-rerank_score[shortlist], kind="stable")
    ]
    order = [*shortlist_order.tolist(), *retrieval_order[top_k:].tolist()]
    top_index = order[0]
    support_probability = float(
        1.0 - probabilities[top_index, SUPPORT_CLASSES.index("No Support")]
    )
    if support_probability < parameters.support_threshold:
        predicted_label = "No Support"
    else:
        predicted_label = (
            "Direct"
            if probabilities[top_index, SUPPORT_CLASSES.index("Direct")]
            >= probabilities[top_index, SUPPORT_CLASSES.index("Partial")]
            else "Partial"
        )
    return order, predicted_label, support_probability


def evaluate_two_stage(
    tasks: Sequence[dict[str, Any]],
    cached: dict[str, CachedTaskPrediction],
    parameters: TwoStageParameters,
) -> dict[str, Any]:
    """Evaluate ranking, rejection, and exact support-class decisions."""
    ranks: list[int] = []
    supported_successes: list[bool] = []
    rejection_successes: list[bool] = []
    task_successes: list[bool] = []
    labels: list[str] = []
    predictions: list[str] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        order, predicted_label, support_probability = _rank_task(
            cached[task_id],
            parameters,
        )
        candidates = list(task["candidates"])
        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in candidates
        ]
        label = str(task["support_label"])
        labels.append(label)
        predictions.append(predicted_label)
        if label == "No Support":
            passed = predicted_label == "No Support"
            rejection_successes.append(passed)
            failure = "false_accept"
        else:
            gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
            rank = order.index(gold_index) + 1
            ranks.append(rank)
            passed = rank == 1 and predicted_label != "No Support"
            supported_successes.append(passed)
            failure = "wrong_rank" if rank != 1 else "support_reject"
        task_successes.append(passed)
        if not passed:
            failures.append(
                {
                    "task_id": task_id,
                    "failure": failure,
                    "predicted_label": predicted_label,
                    "support_probability": support_probability,
                    "top_candidate_id": candidate_ids[order[0]],
                }
            )
    supported_rate = _mean(supported_successes)
    rejection_rate = _mean(rejection_successes)
    return {
        "parameters": {
            "retrieval_weight": parameters.retrieval_weight,
            "partial_weight": parameters.partial_weight,
            "support_threshold": parameters.support_threshold,
            "top_k": parameters.top_k,
        },
        "retrieval": {
            "support_tasks": len(ranks),
            "recall_at_1": _mean([rank == 1 for rank in ranks]),
            "recall_at_3": _mean([rank <= 3 for rank in ranks]),
            "mean_reciprocal_rank": _mean([1.0 / rank for rank in ranks]),
            "supported_task_success_rate": supported_rate,
            "no_support_tasks": len(rejection_successes),
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": _mean(task_successes),
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
        },
        "support_classification": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "macro_f1": float(
                f1_score(
                    labels,
                    predictions,
                    labels=list(SUPPORT_CLASSES),
                    average="macro",
                    zero_division=0,
                )
            ),
            "labels": list(SUPPORT_CLASSES),
            "confusion_matrix": confusion_matrix(
                labels,
                predictions,
                labels=list(SUPPORT_CLASSES),
            ).astype(int).tolist(),
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def select_two_stage_parameters(
    tasks: Sequence[dict[str, Any]],
    cached: dict[str, CachedTaskPrediction],
) -> tuple[TwoStageParameters, dict[str, Any]]:
    """Select parameters on development tasks without refitting the model."""
    candidates = (
        TwoStageParameters(*values)
        for values in product(
            (0.0, 0.2, 0.4, 0.6),
            (0.6, 0.8, 1.0),
            (0.40, 0.50, 0.60, 0.70, 0.80),
        )
    )
    evaluated = [
        (parameters, evaluate_two_stage(tasks, cached, parameters))
        for parameters in candidates
    ]

    def key(
        item: tuple[TwoStageParameters, dict[str, Any]],
    ) -> tuple[float, ...]:
        parameters, result = item
        retrieval = result["retrieval"]
        support = result["support_classification"]
        return (
            float(retrieval["task_balanced_accuracy"]),
            float(retrieval["task_decision_accuracy"]),
            float(retrieval["recall_at_1"]),
            float(retrieval["no_support_rejection_rate"]),
            float(support["macro_f1"]),
            -parameters.retrieval_weight,
        )

    return max(evaluated, key=key)


def evaluate_rank_preserving_gate(
    tasks: Sequence[dict[str, Any]],
    cached: dict[str, CachedTaskPrediction],
    parameters: RankPreservingGateParameters,
) -> dict[str, Any]:
    """Evaluate a gate that cannot alter the supplied retrieval order."""
    ranks: list[int] = []
    supported_successes: list[bool] = []
    rejection_successes: list[bool] = []
    task_successes: list[bool] = []
    failures: list[dict[str, Any]] = []
    no_support_index = SUPPORT_CLASSES.index("No Support")
    for task in tasks:
        task_id = str(task["task_id"])
        prediction = cached[task_id]
        retrieval = np.asarray(prediction["retrieval_scores"], dtype=np.float64)
        order = np.argsort(-retrieval, kind="stable")
        top_index = int(order[0])
        support_probability = float(
            1.0
            - prediction["class_probabilities"][top_index][no_support_index]
        )
        candidates = list(task["candidates"])
        shared_named_term = bool(
            named_terms(str(task["requirement"]))
            & named_terms(str(candidates[top_index]["evidence"]))
        )
        accepted = (
            float(retrieval[top_index]) >= parameters.retrieval_threshold
            and (
                support_probability >= parameters.support_threshold
                or parameters.named_term_rescue
                and shared_named_term
            )
        )
        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in candidates
        ]
        if str(task["support_label"]) == "No Support":
            passed = not accepted
            rejection_successes.append(passed)
            failure = "false_accept"
        else:
            gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
            rank = int(np.flatnonzero(order == gold_index)[0]) + 1
            ranks.append(rank)
            passed = rank == 1 and accepted
            supported_successes.append(passed)
            failure = "wrong_rank" if rank != 1 else "support_reject"
        task_successes.append(passed)
        if not passed:
            failures.append(
                {
                    "task_id": task_id,
                    "failure": failure,
                    "top_candidate_id": candidate_ids[top_index],
                    "retrieval_score": float(retrieval[top_index]),
                    "support_probability": support_probability,
                    "shared_named_term": shared_named_term,
                }
            )
    supported_rate = _mean(supported_successes)
    rejection_rate = _mean(rejection_successes)
    return {
        "parameters": {
            "retrieval_threshold": parameters.retrieval_threshold,
            "support_threshold": parameters.support_threshold,
            "named_term_rescue": parameters.named_term_rescue,
        },
        "retrieval": {
            "support_tasks": len(ranks),
            "recall_at_1": _mean([rank == 1 for rank in ranks]),
            "recall_at_3": _mean([rank <= 3 for rank in ranks]),
            "mean_reciprocal_rank": _mean([1.0 / rank for rank in ranks]),
            "supported_task_success_rate": supported_rate,
            "no_support_tasks": len(rejection_successes),
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": _mean(task_successes),
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
        },
        "failure_count": len(failures),
        "failures": failures,
    }


def select_rank_preserving_gate(
    tasks: Sequence[dict[str, Any]],
    cached: dict[str, CachedTaskPrediction],
    *,
    retrieval_threshold: float,
    reference_supported_success: float,
    reference_rejection_rate: float,
) -> tuple[RankPreservingGateParameters, dict[str, Any]]:
    """Tune an extra gate without sacrificing reference training outcomes."""
    top_support_probabilities: list[float] = []
    no_support_index = SUPPORT_CLASSES.index("No Support")
    for task in tasks:
        prediction = cached[str(task["task_id"])]
        top_index = int(np.argmax(np.asarray(prediction["retrieval_scores"])))
        top_support_probabilities.append(
            1.0
            - prediction["class_probabilities"][top_index][no_support_index]
        )
    thresholds = sorted(
        {
            min(top_support_probabilities),
            max(top_support_probabilities) + 1e-12,
            *(
                float(np.nextafter(probability, float("inf")))
                for probability in top_support_probabilities
            ),
        }
    )
    evaluated = [
        (
            RankPreservingGateParameters(
                retrieval_threshold=retrieval_threshold,
                support_threshold=threshold,
                named_term_rescue=True,
            ),
            evaluate_rank_preserving_gate(
                tasks,
                cached,
                RankPreservingGateParameters(
                    retrieval_threshold=retrieval_threshold,
                    support_threshold=threshold,
                    named_term_rescue=True,
                ),
            ),
        )
        for threshold in thresholds
    ]

    def key(
        item: tuple[RankPreservingGateParameters, dict[str, Any]],
    ) -> tuple[float, ...]:
        parameters, result = item
        retrieval = result["retrieval"]
        preserves_reference = (
            float(retrieval["supported_task_success_rate"])
            + 1e-12
            >= reference_supported_success
            and float(retrieval["no_support_rejection_rate"])
            + 1e-12
            >= reference_rejection_rate
        )
        return (
            float(preserves_reference),
            float(retrieval["task_balanced_accuracy"]),
            float(retrieval["task_decision_accuracy"]),
            float(retrieval["no_support_rejection_rate"]),
            float(retrieval["supported_task_success_rate"]),
            parameters.support_threshold,
        )

    return max(evaluated, key=key)


