"""Conservative semantic ranking with an unchanged lexical acceptance gate."""

from __future__ import annotations

from ml.evidence_metrics import _validate_aligned

from ml.evidence_metrics import _mean

from collections.abc import Sequence
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from ml.evidence import score_transparent_evidence_pair


SEMANTIC_BLEND_CHARACTER_WEIGHTS = (0.0, 0.25, 0.5, 0.75, 1.0)


class CharacterTfidfCosineScorer:
    """Character n-gram cosine scorer for wording and spelling variation."""

    def __init__(self, *, max_features: int = 16_000) -> None:
        self.max_features = max_features
        self.vectorizer = TfidfVectorizer(
            analyzer="char_wb",
            lowercase=True,
            ngram_range=(3, 5),
            min_df=2,
            sublinear_tf=True,
            max_features=max_features,
            norm="l2",
        )

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> CharacterTfidfCosineScorer:
        _validate_aligned(requirements, evidence)
        self.vectorizer.fit(list(dict.fromkeys([*requirements, *evidence])))
        return self

    def score(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        _validate_aligned(requirements, evidence)
        left = self.vectorizer.transform(requirements)
        right = self.vectorizer.transform(evidence)
        return np.asarray(left.multiply(right).sum(axis=1)).ravel()

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "character_tfidf_cosine",
            "features": len(self.vectorizer.get_feature_names_out()),
            "ngram_range": [3, 5],
            "max_features": self.max_features,
        }


def transparent_concept_scores(
    requirements: Sequence[str],
    evidence: Sequence[str],
) -> np.ndarray:
    """Return transparent semantic scores on a stable zero-to-one scale."""
    _validate_aligned(requirements, evidence)
    return np.asarray(
        [
            float(score_transparent_evidence_pair(requirement, candidate)["similarity"])
            / 100.0
            for requirement, candidate in zip(requirements, evidence, strict=True)
        ],
        dtype=np.float64,
    )


def blend_semantic_scores(
    character_scores: Any,
    concept_scores: Any,
    *,
    character_weight: float,
) -> np.ndarray:
    """Blend character and concept scores without changing their scale."""
    if not 0.0 <= character_weight <= 1.0:
        raise ValueError("character_weight must be between zero and one")
    character = np.asarray(character_scores, dtype=np.float64)
    concept = np.asarray(concept_scores, dtype=np.float64)
    if character.shape != concept.shape:
        raise ValueError("character and concept scores must align")
    return character_weight * character + (1.0 - character_weight) * concept


class SemanticConceptBlendScorer:
    """Rank evidence using character TF-IDF and transparent semantic concepts."""

    def __init__(self, *, character_weight: float = 0.25) -> None:
        if not 0.0 <= character_weight <= 1.0:
            raise ValueError("character_weight must be between zero and one")
        self.character_weight = character_weight
        self.character = CharacterTfidfCosineScorer()

    def fit(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> SemanticConceptBlendScorer:
        self.character.fit(requirements, evidence)
        return self

    def score_components(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.character.score(requirements, evidence),
            transparent_concept_scores(requirements, evidence),
        )

    def score(
        self,
        requirements: Sequence[str],
        evidence: Sequence[str],
    ) -> np.ndarray:
        character, concept = self.score_components(requirements, evidence)
        return blend_semantic_scores(
            character,
            concept,
            character_weight=self.character_weight,
        )

    def feature_manifest(self) -> dict[str, Any]:
        return {
            "model_type": "semantic_concept_blend",
            "character_weight": self.character_weight,
            "concept_weight": 1.0 - self.character_weight,
            "character": self.character.feature_manifest(),
            "concept_score": "transparent_requirement_evidence_similarity",
        }


def evaluate_ranker_with_fixed_gate(
    tasks: Sequence[dict[str, Any]],
    ranking_scores: dict[str, list[float]],
    gate_scores: dict[str, list[float]],
    *,
    gate_threshold: float,
    ranking_floor: float = 0.0,
) -> dict[str, Any]:
    """Evaluate new ranking while preserving lexical task acceptance decisions."""
    if ranking_floor < 0.0:
        raise ValueError("ranking_floor must be non-negative")
    ranks: list[int] = []
    support_successes: list[bool] = []
    rejection_successes: list[bool] = []
    task_successes: list[bool] = []
    failures: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task["task_id"])
        candidates = list(task["candidates"])
        candidate_ids = [str(candidate["candidate_id"]) for candidate in candidates]
        ranking = ranking_scores.get(task_id, [])
        gate = gate_scores.get(task_id, [])
        if len(ranking) != len(candidates) or len(gate) != len(candidates):
            raise ValueError(f"Cached scores have wrong length for {task_id}.")
        semantic_top_index = int(np.argmax(np.asarray(ranking)))
        semantic_rerank_used = max(ranking) >= ranking_floor
        top_index = (
            semantic_top_index
            if semantic_rerank_used
            else int(np.argmax(np.asarray(gate)))
        )
        effective_ranking = ranking if semantic_rerank_used else gate
        accepted = max(gate) >= gate_threshold
        if str(task["support_label"]) == "No Support":
            passed = not accepted
            rejection_successes.append(passed)
            failure = "false_accept"
        else:
            gold_index = candidate_ids.index(str(task["selected_candidate_id"]))
            rank = (
                int(
                    np.flatnonzero(
                        np.argsort(-np.asarray(effective_ranking)) == gold_index
                    )[0]
                )
                + 1
            )
            ranks.append(rank)
            passed = rank == 1 and accepted
            support_successes.append(passed)
            failure = "wrong_rank" if rank != 1 else "below_fixed_gate"
        task_successes.append(passed)
        if not passed:
            failures.append(
                {
                    "task_id": task_id,
                    "failure": failure,
                    "top_candidate_id": candidate_ids[top_index],
                    "top_ranking_score": float(max(ranking)),
                    "max_gate_score": float(max(gate)),
                    "semantic_rerank_used": semantic_rerank_used,
                }
            )
    supported_rate = _mean(support_successes)
    rejection_rate = _mean(rejection_successes)
    return {
        "gate_threshold": gate_threshold,
        "ranking_floor": ranking_floor,
        "retrieval": {
            "support_tasks": len(support_successes),
            "no_support_tasks": len(rejection_successes),
            "recall_at_1": _mean([rank == 1 for rank in ranks]),
            "recall_at_3": _mean([rank <= 3 for rank in ranks]),
            "mean_reciprocal_rank": _mean([1.0 / rank for rank in ranks]),
            "supported_task_success_rate": supported_rate,
            "no_support_rejection_rate": rejection_rate,
            "task_decision_accuracy": _mean(task_successes),
            "task_balanced_accuracy": (supported_rate + rejection_rate) / 2,
        },
        "failure_count": len(failures),
        "failures": failures,
    }


