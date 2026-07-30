"""Tests for conservative semantic ranking with a fixed lexical gate."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evidence_semantic_blend import (
    CharacterTfidfCosineScorer,
    SemanticConceptBlendScorer,
    blend_semantic_scores,
    evaluate_ranker_with_fixed_gate,
)


REQUIREMENTS = [
    "Build production data pipelines",
    "Use Python for data analysis",
]
EVIDENCE = [
    "Automated recurring ETL workflows for warehouse delivery.",
    "Analyzed datasets with Python and pandas.",
]


def test_character_and_semantic_scorers_return_finite_aligned_scores() -> None:
    for scorer in (
        CharacterTfidfCosineScorer(),
        SemanticConceptBlendScorer(character_weight=0.25),
    ):
        scorer.fit(REQUIREMENTS, EVIDENCE)
        scores = scorer.score(REQUIREMENTS, EVIDENCE)

        assert scores.shape == (2,)
        assert np.isfinite(scores).all()
        assert scorer.feature_manifest()["model_type"]


def test_semantic_blend_validates_weights_and_shapes() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        SemanticConceptBlendScorer(character_weight=1.1)
    with pytest.raises(ValueError, match="must align"):
        blend_semantic_scores([0.1], [0.2, 0.3], character_weight=0.25)


def test_fixed_gate_changes_ranking_without_changing_task_acceptance() -> None:
    tasks = [
        {
            "task_id": "supported",
            "requirement": "Build data pipelines",
            "support_label": "Direct",
            "selected_candidate_id": "b",
            "candidates": [
                {"candidate_id": "a", "evidence": "Prepared data reports"},
                {"candidate_id": "b", "evidence": "Automated ETL workflows"},
            ],
        },
        {
            "task_id": "unsupported",
            "requirement": "Use Kubernetes",
            "support_label": "No Support",
            "selected_candidate_id": None,
            "candidates": [
                {"candidate_id": "c", "evidence": "Used spreadsheets"},
                {"candidate_id": "d", "evidence": "Wrote documentation"},
            ],
        },
    ]
    result = evaluate_ranker_with_fixed_gate(
        tasks,
        {
            "supported": [0.2, 0.9],
            "unsupported": [0.4, 0.1],
        },
        {
            "supported": [0.8, 0.4],
            "unsupported": [0.3, 0.2],
        },
        gate_threshold=0.5,
    )

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["supported_task_success_rate"] == 1.0
    assert result["retrieval"]["no_support_rejection_rate"] == 1.0


def test_fixed_gate_rejects_misaligned_cached_scores() -> None:
    task = {
        "task_id": "task",
        "requirement": "Use SQL",
        "support_label": "No Support",
        "selected_candidate_id": None,
        "candidates": [
            {"candidate_id": "a", "evidence": "Used spreadsheets"},
            {"candidate_id": "b", "evidence": "Wrote notes"},
        ],
    }

    with pytest.raises(ValueError, match="wrong length"):
        evaluate_ranker_with_fixed_gate(
            [task],
            {"task": [0.2]},
            {"task": [0.2, 0.1]},
            gate_threshold=0.5,
        )


def test_low_semantic_score_falls_back_to_lexical_ranking() -> None:
    task = {
        "task_id": "task",
        "requirement": "Coordinate the engineering team",
        "support_label": "Partial",
        "selected_candidate_id": "a",
        "candidates": [
            {"candidate_id": "a", "evidence": "Coordinated feature delivery"},
            {"candidate_id": "b", "evidence": "Wrote acceptance criteria"},
        ],
    }

    result = evaluate_ranker_with_fixed_gate(
        [task],
        {"task": [0.01, 0.02]},
        {"task": [0.8, 0.3]},
        gate_threshold=0.5,
        ranking_floor=0.1,
    )

    assert result["retrieval"]["supported_task_success_rate"] == 1.0
    with pytest.raises(ValueError, match="non-negative"):
        evaluate_ranker_with_fixed_gate(
            [task],
            {"task": [0.01, 0.02]},
            {"task": [0.8, 0.3]},
            gate_threshold=0.5,
            ranking_floor=-0.1,
        )
