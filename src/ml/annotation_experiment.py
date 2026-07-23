"""Template-grouped evaluation for reviewed requirement/evidence pairs."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from ml.evaluation import select_f1_threshold
from ml.evidence import MIN_ACCEPTED_SIMILARITY, score_evidence_pair
from ml.annotation_metrics import decision_metrics, record_task_scores, retrieval_metrics
from ml.evidence_models import (
    HybridEvidenceReranker,
    LexicalGuardedReranker,
    LsaEmbeddingScorer,
    PairwiseHybridReranker,
    WordTfidfCosineScorer,
)
from ml.relevance import PairRelevanceModel


NO_PORTABLE_MODEL = Path("/missing/reviewed-evidence-model.json")
POSITIVE_SUPPORT_LABELS = {"Direct", "Partial"}


def _evaluation_group(row: dict[str, Any]) -> str:
    return str(row.get("evaluation_group") or row["template_group"])


def _mixed_validation_group(
    pairs: list[dict[str, Any]],
    test_group: str,
    *,
    random_state: int,
) -> str:
    grouped: dict[str, set[int]] = {}
    for pair in pairs:
        grouped.setdefault(_evaluation_group(pair), set()).add(int(pair["binary_label"]))
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


def _score_model(
    method: str,
    training_pairs: list[dict[str, Any]],
    training_tasks: list[dict[str, Any]],
    *,
    random_state: int,
) -> tuple[Callable[[Sequence[str], Sequence[str]], np.ndarray], dict[str, Any]]:
    train_requirements = [str(pair["requirement"]) for pair in training_pairs]
    train_evidence = [str(pair["evidence"]) for pair in training_pairs]
    if method == "tfidf_cosine":
        scorer = WordTfidfCosineScorer().fit(train_requirements, train_evidence)
        return scorer.score, scorer.feature_manifest()
    if method == "lsa_embedding":
        embedding = LsaEmbeddingScorer(random_state=random_state).fit(
            train_requirements,
            train_evidence,
        )
        return embedding.score, embedding.feature_manifest()
    labels = [int(pair["binary_label"]) for pair in training_pairs]
    if method == "hybrid_lsa_reranker":
        reranker = HybridEvidenceReranker(random_state=random_state).fit(
            train_requirements,
            train_evidence,
            labels,
        )
        return reranker.predict_proba, reranker.feature_manifest()
    if method == "lexical_guarded_reranker":
        guarded = LexicalGuardedReranker(random_state=random_state).fit(
            train_requirements,
            train_evidence,
            labels,
        )
        return guarded.predict_proba, guarded.feature_manifest()
    if method == "pairwise_hybrid_reranker":
        pairwise = PairwiseHybridReranker(random_state=random_state).fit(
            train_requirements,
            train_evidence,
            labels,
            training_tasks,
        )
        return pairwise.predict_proba, pairwise.feature_manifest()
    if method == "trained_pair_classifier":
        model = PairRelevanceModel(max_features=5_000, random_state=random_state)
        model.fit(train_evidence, train_requirements, labels)
        return (
            lambda reqs, evs: model.predict_proba(evs, reqs),
            model.feature_manifest(),
        )
    raise ValueError(f"Unsupported annotation experiment method: {method}")


def _evaluate_cross_validated(
    method: str,
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    random_state: int,
) -> tuple[dict[str, Any], list[float]]:
    groups = sorted({_evaluation_group(task) for task in tasks})
    labels: list[int] = []
    scores: list[float] = []
    predictions: list[int] = []
    ranks: list[int] = []
    rejected: list[bool] = []
    task_decisions: list[bool] = []
    thresholds: list[float] = []
    fold_details: list[dict[str, Any]] = []
    pair_errors: list[dict[str, Any]] = []
    retrieval_failures: list[dict[str, Any]] = []
    feature_manifest: dict[str, Any] = {}
    for fold_index, test_group in enumerate(groups):
        validation_group = _mixed_validation_group(
            pairs,
            test_group,
            random_state=random_state,
        )
        train_pairs = [
            pair
            for pair in pairs
            if _evaluation_group(pair) not in {test_group, validation_group}
        ]
        validation_pairs = [
            pair for pair in pairs if _evaluation_group(pair) == validation_group
        ]
        test_pairs = [pair for pair in pairs if _evaluation_group(pair) == test_group]
        train_tasks = [
            task
            for task in tasks
            if _evaluation_group(task) not in {test_group, validation_group}
        ]
        score, feature_manifest = _score_model(
            method,
            train_pairs,
            train_tasks,
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
        fold_labels = [int(pair["binary_label"]) for pair in test_pairs]
        fold_scores = [float(value) for value in test_scores]
        fold_predictions = [int(value >= threshold) for value in test_scores]
        labels.extend(fold_labels)
        scores.extend(fold_scores)
        predictions.extend(fold_predictions)
        for pair, label, prediction, pair_score in zip(
            test_pairs,
            fold_labels,
            fold_predictions,
            fold_scores,
            strict=True,
        ):
            if label != prediction:
                pair_errors.append(
                    {
                        "pair_id": pair.get("pair_id"),
                        "task_id": pair.get("task_id"),
                        "evaluation_group": test_group,
                        "label": label,
                        "prediction": prediction,
                        "score": pair_score,
                        "threshold": float(threshold),
                        "confidence_margin": abs(pair_score - threshold),
                    }
                )
        test_tasks = [task for task in tasks if _evaluation_group(task) == test_group]
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
            score_values = candidate_scores.tolist()
            task_label = str(task["support_label"])
            if task_label == "No Support":
                if max(score_values) >= threshold:
                    retrieval_failures.append(
                        {
                            "task_id": task["task_id"],
                            "evaluation_group": test_group,
                            "failure": "false_accept",
                            "top_score": float(max(score_values)),
                            "threshold": float(threshold),
                        }
                    )
            else:
                candidate_ids = [
                    str(candidate["candidate_id"])
                    for candidate in task["candidates"]
                ]
                gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
                order = np.argsort(-np.asarray(score_values))
                rank = int(np.where(order == gold_index)[0][0]) + 1
                if rank != 1 or max(score_values) < threshold:
                    retrieval_failures.append(
                        {
                            "task_id": task["task_id"],
                            "evaluation_group": test_group,
                            "failure": (
                                "wrong_rank"
                                if rank != 1
                                else "below_threshold"
                            ),
                            "gold_rank": rank,
                            "top_score": float(max(score_values)),
                            "threshold": float(threshold),
                        }
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
            "threshold_protocol": "selected on a separate mixed-label evaluation group",
            "threshold_range": [float(min(thresholds)), float(max(thresholds))],
            "pair_classification": decision_metrics(labels, predictions, scores),
            "retrieval": retrieval_metrics(ranks, rejected, task_decisions),
            "folds": fold_details,
            "feature_manifest": feature_manifest,
            "error_analysis": {
                "pair_errors": sorted(
                    pair_errors,
                    key=lambda error: float(error["confidence_margin"]),
                    reverse=True,
                )[:25],
                "retrieval_failures": retrieval_failures[:25],
                "pair_error_count": len(pair_errors),
                "retrieval_failure_count": len(retrieval_failures),
            },
        },
        thresholds,
    )


def run_annotation_experiment(
    tasks: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    random_state: int = 42,
) -> dict[str, Any]:
    """Compare transparent, latent, and trained reranking baselines."""
    cosine, _ = _evaluate_cross_validated(
        "tfidf_cosine",
        tasks,
        pairs,
        random_state=random_state,
    )
    embedding, embedding_thresholds = _evaluate_cross_validated(
        "lsa_embedding",
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
    hybrid, hybrid_thresholds = _evaluate_cross_validated(
        "hybrid_lsa_reranker",
        tasks,
        pairs,
        random_state=random_state,
    )
    guarded, guarded_thresholds = _evaluate_cross_validated(
        "lexical_guarded_reranker",
        tasks,
        pairs,
        random_state=random_state,
    )
    pairwise, pairwise_thresholds = _evaluate_cross_validated(
        "pairwise_hybrid_reranker",
        tasks,
        pairs,
        random_state=random_state,
    )
    evaluation_groups = {_evaluation_group(task) for task in tasks}
    methods = {
        "concept_lexical_rule": _evaluate_fixed_rule(tasks, pairs),
        "tfidf_cosine": cosine,
        "lsa_embedding": embedding,
        "trained_pair_classifier": trained,
        "hybrid_lsa_reranker": hybrid,
        "lexical_guarded_reranker": guarded,
        "pairwise_hybrid_reranker": pairwise,
    }
    selection_order = (
        "task_decision_accuracy",
        "average_precision",
        "recall_at_1",
        "no_support_rejection_rate",
    )

    def selection_key(method_name: str) -> tuple[float, ...]:
        method = methods[method_name]
        return (
            float(method["retrieval"]["task_decision_accuracy"]),
            float(method["pair_classification"]["average_precision"]),
            float(method["retrieval"]["recall_at_1"]),
            float(method["retrieval"]["no_support_rejection_rate"]),
        )

    selected_method = max(methods, key=selection_key)
    return {
        "experiment": "reviewed_evidence_training_v3",
        "evaluation_protocol": "leave one semantic/template evaluation group out",
        "task_count": len(tasks),
        "pair_count": len(pairs),
        "template_group_count": len(evaluation_groups),
        "evaluation_group_count": len(evaluation_groups),
        "methods": methods,
        "model_selection": {
            "selected_method": selected_method,
            "lexicographic_metrics": list(selection_order),
            "selected_values": list(selection_key(selected_method)),
            "promotion_status": "blocked_until_fixed_real_holdout",
        },
        "trained_threshold_median": float(np.median(trained_thresholds)),
        "method_threshold_medians": {
            "lsa_embedding": float(np.median(embedding_thresholds)),
            "trained_pair_classifier": float(np.median(trained_thresholds)),
            "hybrid_lsa_reranker": float(np.median(hybrid_thresholds)),
            "lexical_guarded_reranker": float(np.median(guarded_thresholds)),
            "pairwise_hybrid_reranker": float(np.median(pairwise_thresholds)),
        },
        "pair_label_counts": dict(Counter(str(pair["binary_label"]) for pair in pairs)),
        "interpretation": (
            "Experimental pilot diagnostic only. Do not replace the current evidence "
            "decision path until an independent real-data holdout confirms improvement."
        ),
    }
