"""Separate evidence ranking from transparent support acceptance."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, NotRequired, TypedDict

import numpy as np


class SupportSignal(TypedDict):
    """Auditable support checks for one requirement/evidence pair."""

    similarity: float
    has_overlap: bool
    numeric_constraint_supported: bool
    compound_requirement_supported: bool
    hard_constraint_supported: NotRequired[bool]
    explicit_negation: NotRequired[bool]


class EvidenceDecisionError(ValueError):
    """Raised when ranking scores and support signals do not align."""


def support_signal_from_score(score: dict[str, Any]) -> SupportSignal:
    """Reduce the transparent scorer output to the acceptance-gate contract."""
    return {
        "similarity": float(score["similarity"]),
        "has_overlap": bool(score["shared_terms"] or score["shared_concepts"]),
        "numeric_constraint_supported": bool(
            score["numeric_constraint_supported"]
        ),
        "compound_requirement_supported": bool(
            score["compound_requirement_supported"]
        ),
    }


def support_signal_from_probability(probability: float) -> SupportSignal:
    """Adapt an explicit-feature support probability to the gate contract."""
    return {
        "similarity": float(probability),
        "has_overlap": True,
        "numeric_constraint_supported": True,
        "compound_requirement_supported": True,
    }


def gate_accepts(signal: SupportSignal, *, threshold: float) -> bool:
    """Accept support only when similarity and hard checks all pass."""
    return (
        signal["similarity"] >= threshold
        and signal["has_overlap"]
        and signal["numeric_constraint_supported"]
        and signal["compound_requirement_supported"]
        and signal.get("hard_constraint_supported", True)
        and not signal.get("explicit_negation", False)
    )


def evaluate_ranked_support_gate(
    tasks: Sequence[dict[str, Any]],
    ranking_scores: dict[str, list[float]],
    support_signals: dict[str, list[SupportSignal]],
    *,
    support_threshold: float,
) -> dict[str, Any]:
    """Evaluate independent candidate ranking and top-candidate acceptance."""
    reciprocal_ranks: list[float] = []
    top_rank_successes: list[bool] = []
    supported_successes: list[bool] = []
    rejection_successes: list[bool] = []
    task_successes: list[bool] = []
    failures: list[dict[str, Any]] = []
    decisions: list[dict[str, Any]] = []

    for task in tasks:
        task_id = str(task["task_id"])
        candidates = list(task["candidates"])
        scores = ranking_scores.get(task_id, [])
        signals = support_signals.get(task_id, [])
        if len(scores) != len(candidates) or len(signals) != len(candidates):
            raise EvidenceDecisionError(
                f"Scores or support signals have wrong length for {task_id}."
            )

        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in candidates
        ]
        top_index = int(np.argmax(np.asarray(scores)))
        top_signal = signals[top_index]
        accepted = gate_accepts(top_signal, threshold=support_threshold)
        selected = task.get("selected_candidate_id")
        top_is_gold: bool | None = None

        if str(task["support_label"]) == "No Support":
            passed = not accepted
            rejection_successes.append(passed)
            failure = "false_accept"
        else:
            gold_index = candidate_ids.index(str(selected))
            ranked_indices = np.argsort(-np.asarray(scores), kind="stable")
            rank = int(np.flatnonzero(ranked_indices == gold_index)[0]) + 1
            reciprocal_ranks.append(1.0 / rank)
            top_is_gold = top_index == gold_index
            top_rank_successes.append(top_is_gold)
            passed = top_is_gold and accepted
            supported_successes.append(passed)
            failure = "wrong_rank" if not top_is_gold else "gate_reject"

        task_successes.append(passed)
        decisions.append(
            {
                "task_id": task_id,
                "support_label": str(task["support_label"]),
                "top_candidate_id": candidate_ids[top_index],
                "top_is_gold": top_is_gold,
                "accepted": accepted,
                "passed": passed,
                "ranking_score": float(scores[top_index]),
                "support_score": top_signal["similarity"],
            }
        )
        if not passed:
            failures.append(
                {
                    "task_id": task_id,
                    "failure": failure,
                    "top_candidate_id": candidate_ids[top_index],
                    "ranking_score": float(scores[top_index]),
                    "support_score": top_signal["similarity"],
                    "support_checks": {
                        key: value
                        for key, value in top_signal.items()
                        if key != "similarity"
                    },
                }
            )

    supported_rate = _mean(supported_successes)
    rejection_rate = _mean(rejection_successes)
    return {
        "support_threshold": support_threshold,
        "retrieval": {
            "recall_at_1": _mean(top_rank_successes),
            "mean_reciprocal_rank": _mean(reciprocal_ranks),
            "supported_task_success_rate": supported_rate,
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": _mean(task_successes),
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
        },
        "accepted_task_count": sum(
            gate_accepts(
                support_signals[str(task["task_id"])][
                    int(
                        np.argmax(
                            np.asarray(
                                ranking_scores[str(task["task_id"])]
                            )
                        )
                    )
                ],
                threshold=support_threshold,
            )
            for task in tasks
        ),
        "failure_count": len(failures),
        "failures": failures,
        "decisions": decisions,
    }


def select_support_gate_threshold(
    tasks: Sequence[dict[str, Any]],
    ranking_scores: dict[str, list[float]],
    support_signals: dict[str, list[SupportSignal]],
) -> tuple[float, dict[str, Any]]:
    """Select one transparent gate threshold on development tasks only."""
    top_similarities = []
    for task in tasks:
        task_id = str(task["task_id"])
        scores = ranking_scores[task_id]
        signals = support_signals[task_id]
        top_index = int(np.argmax(np.asarray(scores)))
        top_similarities.append(signals[top_index]["similarity"])
    thresholds = sorted(
        {
            min(top_similarities),
            max(top_similarities) + 1e-12,
            *(
                float(np.nextafter(similarity, float("inf")))
                for similarity in top_similarities
            ),
        }
    )
    evaluated = [
        (
            threshold,
            evaluate_ranked_support_gate(
                tasks,
                ranking_scores,
                support_signals,
                support_threshold=threshold,
            ),
        )
        for threshold in thresholds
    ]

    def selection_key(item: tuple[float, dict[str, Any]]) -> tuple[float, ...]:
        threshold, result = item
        retrieval = result["retrieval"]
        return (
            float(retrieval["task_balanced_accuracy"]),
            float(retrieval["task_decision_accuracy"]),
            float(retrieval["no_support_rejection_rate"]),
            float(retrieval["supported_task_success_rate"]),
            threshold,
        )

    return max(evaluated, key=selection_key)


def paired_stratified_bootstrap_delta(
    support_mask: Sequence[bool],
    reference_outcomes: Sequence[bool],
    candidate_outcomes: Sequence[bool],
    *,
    samples: int = 5_000,
    random_state: int = 42,
) -> dict[str, float | int]:
    """Estimate uncertainty in task-balanced accuracy improvement."""
    if not (
        len(support_mask)
        == len(reference_outcomes)
        == len(candidate_outcomes)
    ):
        raise ValueError("bootstrap inputs must have equal lengths")
    if samples < 100:
        raise ValueError("bootstrap requires at least 100 samples")
    support_indices = np.flatnonzero(np.asarray(support_mask, dtype=bool))
    no_support_indices = np.flatnonzero(
        ~np.asarray(support_mask, dtype=bool)
    )
    if not len(support_indices) or not len(no_support_indices):
        raise ValueError("bootstrap requires support and No Support tasks")
    reference = np.asarray(reference_outcomes, dtype=np.float64)
    candidate = np.asarray(candidate_outcomes, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    deltas = np.empty(samples, dtype=np.float64)
    for sample_index in range(samples):
        sampled_support = rng.choice(
            support_indices,
            size=len(support_indices),
            replace=True,
        )
        sampled_no_support = rng.choice(
            no_support_indices,
            size=len(no_support_indices),
            replace=True,
        )
        reference_balanced = (
            float(np.mean(reference[sampled_support]))
            + float(np.mean(reference[sampled_no_support]))
        ) / 2
        candidate_balanced = (
            float(np.mean(candidate[sampled_support]))
            + float(np.mean(candidate[sampled_no_support]))
        ) / 2
        deltas[sample_index] = candidate_balanced - reference_balanced
    return {
        "samples": samples,
        "mean_delta": float(np.mean(deltas)),
        "lower_90": float(np.quantile(deltas, 0.05)),
        "upper_90": float(np.quantile(deltas, 0.95)),
        "probability_positive": float(np.mean(deltas > 0)),
    }


def _mean(values: Sequence[bool] | Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0
