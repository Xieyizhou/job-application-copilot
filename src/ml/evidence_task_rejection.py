"""Task-level rejection over candidate-complete evidence probabilities."""

from __future__ import annotations

from ml.evidence_metrics import _mean

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.evidence_multiclass import SUPPORT_CLASSES


TASK_REJECTION_FEATURE_NAMES = (
    "top_support_probability",
    "second_support_probability",
    "support_probability_margin",
    "mean_support_probability",
    "support_probability_std",
    "top_direct_probability",
    "top_partial_probability",
    "top_no_support_probability",
    "max_direct_probability",
    "max_partial_probability",
    "support_count_gte_050",
    "support_count_gte_070",
    "top_rank_score",
    "rank_score_margin",
    "rank_score_std",
)


def task_rejection_features(
    class_probabilities: Sequence[Sequence[float]],
) -> np.ndarray:
    """Return auditable distribution features for one candidate-complete task."""
    probabilities = np.asarray(class_probabilities, dtype=np.float64)
    if probabilities.ndim != 2 or probabilities.shape[0] < 2:
        raise ValueError("Task rejection needs at least two candidate rows.")
    if probabilities.shape[1] != len(SUPPORT_CLASSES):
        raise ValueError("Candidate probabilities have the wrong class count.")
    direct = probabilities[:, SUPPORT_CLASSES.index("Direct")]
    partial = probabilities[:, SUPPORT_CLASSES.index("Partial")]
    no_support = probabilities[:, SUPPORT_CLASSES.index("No Support")]
    support = 1.0 - no_support
    rank_scores = direct + 0.5 * partial
    support_order = np.sort(support)[::-1]
    rank_order = np.sort(rank_scores)[::-1]
    top_index = int(np.argmax(rank_scores))
    return np.asarray(
        [
            support_order[0],
            support_order[1],
            support_order[0] - support_order[1],
            float(np.mean(support)),
            float(np.std(support)),
            direct[top_index],
            partial[top_index],
            no_support[top_index],
            float(np.max(direct)),
            float(np.max(partial)),
            float(np.sum(support >= 0.50)),
            float(np.sum(support >= 0.70)),
            rank_order[0],
            rank_order[0] - rank_order[1],
            float(np.std(rank_scores)),
        ],
        dtype=np.float64,
    )


class TaskSupportRejector:
    """Predict whether any candidate provides usable support."""

    def __init__(self, *, random_state: int = 42) -> None:
        self.pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1_000,
                        random_state=random_state,
                        solver="liblinear",
                    ),
                ),
            ]
        )
        self.is_fitted = False

    def fit(
        self,
        features: Sequence[Sequence[float]],
        labels: Sequence[int],
    ) -> "TaskSupportRejector":
        """Fit on out-of-fold candidate-distribution features only."""
        matrix = np.asarray(features, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(
            TASK_REJECTION_FEATURE_NAMES
        ):
            raise ValueError("Task rejection feature matrix is invalid.")
        if len(labels) != len(matrix) or set(int(value) for value in labels) != {
            0,
            1,
        }:
            raise ValueError("Task labels must align and contain both classes.")
        self.pipeline.fit(matrix, labels)
        self.is_fitted = True
        return self

    def predict_proba(
        self,
        features: Sequence[Sequence[float]],
    ) -> np.ndarray:
        """Return the probability that the task contains usable support."""
        if not self.is_fitted:
            raise RuntimeError("Task rejector is not fitted.")
        matrix = np.asarray(features, dtype=np.float64)
        return self.pipeline.predict_proba(matrix)[:, 1]

    def feature_manifest(self) -> dict[str, Any]:
        """Expose the fitted standardized linear coefficients."""
        if not self.is_fitted:
            raise RuntimeError("Task rejector is not fitted.")
        classifier = self.pipeline.named_steps["classifier"]
        scaler = self.pipeline.named_steps["scaler"]
        return {
            "model_type": "task_support_logistic_rejector",
            "features": {
                name: float(value)
                for name, value in zip(
                    TASK_REJECTION_FEATURE_NAMES,
                    classifier.coef_[0],
                    strict=True,
                )
            },
            "feature_means": {
                name: float(value)
                for name, value in zip(
                    TASK_REJECTION_FEATURE_NAMES,
                    scaler.mean_,
                    strict=True,
                )
            },
            "intercept": float(classifier.intercept_[0]),
        }


def select_task_rejection_threshold(
    labels: Sequence[int],
    probabilities: Sequence[float],
) -> float:
    """Select a conservative balanced-accuracy threshold on OOF predictions."""
    y_true = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(probabilities, dtype=np.float64)
    if len(y_true) != len(scores) or set(y_true.tolist()) != {0, 1}:
        raise ValueError("Threshold labels and probabilities are invalid.")
    thresholds = sorted(
        {
            0.0,
            1.0,
            *scores.tolist(),
            *(
                float((left + right) / 2.0)
                for left, right in zip(
                    np.sort(np.unique(scores))[:-1],
                    np.sort(np.unique(scores))[1:],
                    strict=True,
                )
            ),
        }
    )

    def selection_key(threshold: float) -> tuple[float, float, float]:
        predictions = (scores >= threshold).astype(int)
        balanced = float(balanced_accuracy_score(y_true, predictions))
        rejection = float(
            np.mean(predictions[y_true == 0] == 0)
        )
        return balanced, rejection, threshold

    return max(thresholds, key=selection_key)


def evaluate_task_policy(
    tasks: Sequence[dict[str, Any]],
    probabilities_by_task: Mapping[str, Sequence[Sequence[float]]],
    *,
    acceptance_probabilities: Mapping[str, float] | None = None,
    acceptance_threshold: float = 0.5,
) -> dict[str, Any]:
    """Evaluate fixed ranking with either candidate or task-level acceptance."""
    ranks: list[int] = []
    supported_successes: list[bool] = []
    rejection_successes: list[bool] = []
    failures: list[dict[str, str]] = []
    outcomes: dict[str, bool] = {}
    for task in tasks:
        task_id = str(task["task_id"])
        candidates = list(task["candidates"])
        probabilities = np.asarray(
            probabilities_by_task[task_id],
            dtype=np.float64,
        )
        if probabilities.shape != (len(candidates), len(SUPPORT_CLASSES)):
            raise ValueError(f"Candidate probabilities misalign for {task_id}.")
        rank_scores = (
            probabilities[:, SUPPORT_CLASSES.index("Direct")]
            + 0.5
            * probabilities[:, SUPPORT_CLASSES.index("Partial")]
        )
        order = np.argsort(-rank_scores, kind="stable").tolist()
        if acceptance_probabilities is None:
            accepted = bool(
                np.any(
                    np.argmax(probabilities, axis=1)
                    != SUPPORT_CLASSES.index("No Support")
                )
            )
        else:
            accepted = (
                float(acceptance_probabilities[task_id])
                >= acceptance_threshold
            )
        if str(task["support_label"]) == "No Support":
            passed = not accepted
            rejection_successes.append(passed)
            if not passed:
                failures.append(
                    {"task_id": task_id, "failure": "false_accept"}
                )
            outcomes[task_id] = passed
            continue
        candidate_ids = [
            str(candidate["candidate_id"]) for candidate in candidates
        ]
        gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
        rank = order.index(gold_index) + 1
        ranks.append(rank)
        passed = accepted and rank == 1
        supported_successes.append(passed)
        if not passed:
            failures.append(
                {
                    "task_id": task_id,
                    "failure": (
                        "support_reject" if not accepted else "wrong_rank"
                    ),
                }
            )
        outcomes[task_id] = passed
    supported_rate = _mean(supported_successes)
    rejection_rate = _mean(rejection_successes)
    return {
        "retrieval": {
            "support_tasks": len(ranks),
            "no_support_tasks": len(rejection_successes),
            "recall_at_1": _mean([rank == 1 for rank in ranks]),
            "recall_at_3": _mean([rank <= 3 for rank in ranks]),
            "mean_reciprocal_rank": _mean([1.0 / rank for rank in ranks]),
            "supported_task_success_rate": supported_rate,
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": (
                sum(outcomes.values()) / len(outcomes)
            ),
            "task_balanced_accuracy": (
                supported_rate + rejection_rate
            )
            / 2.0,
        },
        "failures": failures,
        "outcomes": outcomes,
    }


