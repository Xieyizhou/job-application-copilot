"""Template-grouped evaluation for reviewed requirement/evidence pairs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.evaluation import select_f1_threshold
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair
from ml.annotation_metrics import decision_metrics, record_task_scores, retrieval_metrics
from ml.relevance import PairRelevanceModel


NO_PORTABLE_MODEL = Path("/missing/reviewed-evidence-model.json")
POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}


def _binary_task_label(task: dict[str, Any]) -> int:
    return int(str(task["support_label"]) in POSITIVE_SUPPORT_LABELS)


def _mixed_validation_group(
    tasks: list[dict[str, Any]],
    test_group: str,
    *,
    random_state: int,
) -> str:
    grouped: dict[str, set[int]] = {}
    for task in tasks:
        grouped.setdefault(str(task["template_group"]), set()).add(_binary_task_label(task))
    candidates = [
        group
        for group, labels in grouped.items()
        if group != test_group and labels == {0, 1}
    ]
    if not candidates:
        raise ValueError("At least two mixed-label template groups are required.")
    return min(
        candidates,
        key=lambda group: hashlib.sha256(
            f"{random_state}:{test_group}:{group}".encode("utf-8")
        ).hexdigest(),
    )


def _evaluate_fixed_rule(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    pair_scores = [
        float(
            score_evidence_pair(
                str(pair["requirement"]),
                str(pair["evidence"]),
                model_path=NO_PORTABLE_MODEL,
            )["similarity"]
        )
        for pair in pairs
    ]
    labels = [int(pair["binary_label"]) for pair in pairs]
    predictions = [int(score >= MIN_ACCEPTED_SIMILARITY) for score in pair_scores]
    ranks: list[int] = []
    rejected: list[bool] = []
    task_decisions: list[bool] = []
    for task in tasks:
        scores = [
            float(
                score_evidence_pair(
                    str(task["requirement"]),
                    str(candidate["evidence"]),
                    model_path=NO_PORTABLE_MODEL,
                )["similarity"]
            )
            for candidate in task["candidates"]
        ]
        record_task_scores(
            task,
            scores,
            MIN_ACCEPTED_SIMILARITY,
            ranks=ranks,
            rejected=rejected,
            task_decisions=task_decisions,
        )
    return {
        "threshold_protocol": f"fixed transparent threshold {MIN_ACCEPTED_SIMILARITY}",
        "pair_classification": decision_metrics(labels, predictions, pair_scores),
        "retrieval": retrieval_metrics(ranks, rejected, task_decisions),
    }


class _CosineScorer:
    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            ngram_range=(1, 2),
            min_df=1,
            sublinear_tf=True,
            norm="l2",
        )

    def fit(self, requirements: Sequence[str], evidence: Sequence[str]) -> "_CosineScorer":
        self.vectorizer.fit(list(dict.fromkeys([*requirements, *evidence])))
        return self

    def score(self, requirements: Sequence[str], evidence: Sequence[str]) -> np.ndarray:
        requirement_matrix = self.vectorizer.transform(requirements)
        evidence_matrix = self.vectorizer.transform(evidence)
        return np.asarray(requirement_matrix.multiply(evidence_matrix).sum(axis=1)).ravel()


def _score_model(
    method: str,
    training_pairs: list[dict[str, Any]],
    *,
    random_state: int,
) -> tuple[Callable[[Sequence[str], Sequence[str]], np.ndarray], dict[str, Any]]:
    train_requirements = [str(pair["requirement"]) for pair in training_pairs]
    train_evidence = [str(pair["evidence"]) for pair in training_pairs]
    if method == "tfidf_cosine":
        scorer = _CosineScorer().fit(train_requirements, train_evidence)
        return scorer.score, {"features": "word unigram and bigram cosine"}
    model = PairRelevanceModel(max_features=5_000, random_state=random_state)
    model.fit(
        train_evidence,
        train_requirements,
        [int(pair["binary_label"]) for pair in training_pairs],
    )
    return (
        lambda reqs, evs: model.predict_proba(evs, reqs),
        model.feature_manifest(),
    )


def _evaluate_cross_validated(
    method: str,
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    random_state: int,
) -> tuple[dict[str, Any], list[float]]:
    groups = sorted({str(task["template_group"]) for task in tasks})
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    ranks: list[int] = []
    rejected: list[bool] = []
    task_decisions: list[bool] = []
    thresholds: list[float] = []
    fold_details: list[dict[str, Any]] = []
    feature_manifest: dict[str, Any] = {}
    for fold_index, test_group in enumerate(groups):
        validation_group = _mixed_validation_group(
            tasks,
            test_group,
            random_state=random_state,
        )
        train_pairs = [
            pair
            for pair in pairs
            if pair["template_group"] not in {test_group, validation_group}
        ]
        validation_pairs = [
            pair for pair in pairs if pair["template_group"] == validation_group
        ]
        test_pairs = [pair for pair in pairs if pair["template_group"] == test_group]
        score, feature_manifest = _score_model(
            method,
            train_pairs,
            random_state=random_state + fold_index,
        )
        validation_scores = score(
            [str(pair["requirement"]) for pair in validation_pairs],
            [str(pair["evidence"]) for pair in validation_pairs],
        )
        threshold = select_f1_threshold(
            [int(pair["binary_label"]) for pair in validation_pairs],
            validation_scores.tolist(),
        )
        thresholds.append(threshold)
        test_scores = score(
            [str(pair["requirement"]) for pair in test_pairs],
            [str(pair["evidence"]) for pair in test_pairs],
        )
        labels.extend(int(pair["binary_label"]) for pair in test_pairs)
        scores.extend(float(value) for value in test_scores)
        predictions.extend(int(value >= threshold) for value in test_scores)
        test_tasks = [task for task in tasks if task["template_group"] == test_group]
        for task in test_tasks:
            candidate_scores = score(
                [str(task["requirement"])] * len(task["candidates"]),
                [str(candidate["evidence"]) for candidate in task["candidates"]],
            )
            record_task_scores(
                task,
                candidate_scores.tolist(),
                threshold,
                ranks=ranks,
                rejected=rejected,
                task_decisions=task_decisions,
            )
        fold_details.append(
            {
                "test_group": test_group,
                "validation_group": validation_group,
                "train_pairs": len(train_pairs),
                "test_pairs": len(test_pairs),
                "threshold": float(threshold),
            }
        )
    return (
        {
            "threshold_protocol": "selected on a separate mixed-label template group",
            "threshold_range": [float(min(thresholds)), float(max(thresholds))],
            "pair_classification": decision_metrics(labels, predictions, scores),
            "retrieval": retrieval_metrics(ranks, rejected, task_decisions),
            "folds": fold_details,
            "feature_manifest": feature_manifest,
        },
        thresholds,
    )


def run_annotation_experiment(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare transparent and trained baselines without template leakage."""
    cosine, _ = _evaluate_cross_validated(
        "tfidf_cosine",
        tasks,
        pairs,
        random_state=random_state,
    )
    trained, trained_thresholds = _evaluate_cross_validated(
        "trained_pair_classifier",
        tasks,
        pairs,
        random_state=random_state,
    )
    return {
        "experiment": "reviewed_evidence_pilot_v2",
        "evaluation_protocol": "leave one requirement template group out",
        "task_count": len(tasks),
        "pair_count": len(pairs),
        "template_group_count": len({task["template_group"] for task in tasks}),
        "methods": {
            "concept_lexical_rule": _evaluate_fixed_rule(tasks, pairs),
            "tfidf_cosine": cosine,
            "trained_pair_classifier": trained,
        },
        "trained_threshold_median": float(np.median(trained_thresholds)),
        "pair_label_counts": dict(Counter(str(pair["binary_label"]) for pair in pairs)),
        "interpretation": (
            "Experimental pilot diagnostic only. Do not replace the current evidence "
            "decision path until an independent real-data holdout confirms improvement."
        ),
    }
