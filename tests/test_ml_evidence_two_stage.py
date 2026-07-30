"""Tests for the local three-class, two-stage evidence experiment."""

from __future__ import annotations

import numpy as np
import pytest

from ml.evidence_multiclass import (
    SUPPORT_CLASSES,
    MulticlassEvidenceReranker,
    PairTextTfidfClassifier,
)
from ml.evidence_two_stage import (
    RankPreservingGateParameters,
    TwoStageParameters,
    evaluate_rank_preserving_gate,
    evaluate_two_stage,
    reviewed_multiclass_rows,
    select_rank_preserving_gate,
    select_two_stage_parameters,
)


def _training_rows() -> tuple[list[str], list[str], list[str]]:
    requirements = [
        "Build production data pipelines",
        "Use Python for analysis",
        "Deploy models to production",
        "Use Docker in production",
        "Manage client support requests",
        "Hold a PhD in computer science",
    ]
    evidence = [
        "Built and monitored production ETL data pipelines.",
        "Completed a guided Python tutorial without independent delivery.",
        "Trained models locally but did not deploy them.",
        "Deployed a containerized service with Docker.",
        "Answered customer requests through chat and email.",
        "Built a search API using Scala.",
    ]
    labels = [
        "Direct",
        "Partial",
        "Partial",
        "Direct",
        "Direct",
        "No Support",
    ]
    return requirements, evidence, labels


def test_multiclass_probabilities_use_stable_class_order() -> None:
    requirements, evidence, labels = _training_rows()
    model = MulticlassEvidenceReranker(max_components=4).fit(
        requirements,
        evidence,
        labels,
    )

    probabilities = model.predict_class_proba(
        ["Use Docker"],
        ["Deployed services with Docker"],
    )

    assert probabilities.shape == (1, len(SUPPORT_CLASSES))
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(1))
    assert model.feature_manifest()["classes"] == list(SUPPORT_CLASSES)


def test_multiclass_fit_requires_every_support_class() -> None:
    with pytest.raises(ValueError, match="exactly"):
        MulticlassEvidenceReranker(max_components=2).fit(
            ["Use SQL", "Use Python"],
            ["Built SQL reports", "Used Python"],
            ["Direct", "Partial"],
        )


def test_pair_text_tfidf_predicts_stable_support_classes() -> None:
    requirements, evidence, labels = _training_rows()
    model = PairTextTfidfClassifier().fit(requirements, evidence, labels)

    predictions = model.predict(
        ["Use Docker"],
        ["Deployed services with Docker"],
    )

    assert predictions.shape == (1,)
    assert predictions[0] in SUPPORT_CLASSES
    assert model.feature_manifest()["classes"] == list(SUPPORT_CLASSES)


def test_reviewed_rows_preserve_complete_candidate_labels() -> None:
    task = {
        "task_id": "complete",
        "requirement": "Deploy models",
        "support_label": "Direct",
        "selected_candidate_id": "a",
        "candidates": [
            {
                "candidate_id": "a",
                "evidence": "Deployed a model API.",
                "support_label": "Direct",
            },
            {
                "candidate_id": "b",
                "evidence": "Trained models offline.",
                "support_label": "Partial",
            },
            {
                "candidate_id": "c",
                "evidence": "Prepared status notes.",
                "support_label": "No Support",
            },
        ],
    }

    assert reviewed_multiclass_rows([task]) == [
        ("Deploy models", "Deployed a model API.", "Direct"),
        ("Deploy models", "Trained models offline.", "Partial"),
        ("Deploy models", "Prepared status notes.", "No Support"),
    ]


def test_two_stage_rejects_top_candidate_below_support_threshold() -> None:
    task = {
        "task_id": "unsupported",
        "requirement": "Use Kubernetes",
        "support_label": "No Support",
        "selected_candidate_id": None,
        "candidates": [
            {"candidate_id": "a", "evidence": "Used spreadsheets"},
            {"candidate_id": "b", "evidence": "Wrote documentation"},
        ],
    }
    cached = {
        "unsupported": {
            "retrieval_scores": [0.4, 0.3],
            "class_probabilities": [
                [0.10, 0.15, 0.75],
                [0.05, 0.10, 0.85],
            ],
        }
    }

    result = evaluate_two_stage(
        [task],
        cached,
        TwoStageParameters(
            retrieval_weight=0.2,
            partial_weight=0.8,
            support_threshold=0.6,
        ),
    )

    assert result["retrieval"]["no_support_rejection_rate"] == 1.0
    assert result["support_classification"]["confusion_matrix"][2][2] == 1


def test_two_stage_reranks_only_within_lsa_shortlist() -> None:
    task = {
        "task_id": "supported",
        "requirement": "Build pipelines",
        "support_label": "Direct",
        "selected_candidate_id": "b",
        "candidates": [
            {"candidate_id": "a", "evidence": "Made reports"},
            {"candidate_id": "b", "evidence": "Built ETL pipelines"},
            {"candidate_id": "c", "evidence": "Used spreadsheets"},
            {"candidate_id": "d", "evidence": "Deployed APIs"},
        ],
    }
    cached = {
        "supported": {
            "retrieval_scores": [0.9, 0.8, 0.7, 0.1],
            "class_probabilities": [
                [0.10, 0.20, 0.70],
                [0.80, 0.10, 0.10],
                [0.20, 0.20, 0.60],
                [0.99, 0.005, 0.005],
            ],
        }
    }

    result = evaluate_two_stage(
        [task],
        cached,
        TwoStageParameters(
            retrieval_weight=0.0,
            partial_weight=0.8,
            support_threshold=0.5,
            top_k=3,
        ),
    )

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["failure_count"] == 0


def test_parameter_selection_returns_evaluated_candidate() -> None:
    task = {
        "task_id": "direct",
        "requirement": "Use SQL",
        "support_label": "Direct",
        "selected_candidate_id": "a",
        "candidates": [
            {"candidate_id": "a", "evidence": "Built SQL reports"},
            {"candidate_id": "b", "evidence": "Used spreadsheets"},
        ],
    }
    cached = {
        "direct": {
            "retrieval_scores": [0.9, 0.2],
            "class_probabilities": [
                [0.8, 0.1, 0.1],
                [0.1, 0.1, 0.8],
            ],
        }
    }

    parameters, result = select_two_stage_parameters([task], cached)

    assert parameters.top_k == 3
    assert result["retrieval"]["task_decision_accuracy"] == 1.0


def test_rank_preserving_gate_keeps_retrieval_order() -> None:
    tasks = [
        {
            "task_id": "supported",
            "requirement": "Use SQL",
            "support_label": "Direct",
            "selected_candidate_id": "a",
            "candidates": [
                {"candidate_id": "a", "evidence": "Built SQL reports"},
                {"candidate_id": "b", "evidence": "Used spreadsheets"},
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
    cached = {
        "supported": {
            "retrieval_scores": [0.9, 0.2],
            "class_probabilities": [
                [0.6, 0.3, 0.1],
                [0.1, 0.1, 0.8],
            ],
        },
        "unsupported": {
            "retrieval_scores": [0.8, 0.1],
            "class_probabilities": [
                [0.1, 0.1, 0.8],
                [0.6, 0.3, 0.1],
            ],
        },
    }

    parameters, result = select_rank_preserving_gate(
        tasks,
        cached,
        retrieval_threshold=0.5,
        reference_supported_success=1.0,
        reference_rejection_rate=1.0,
    )

    assert parameters.retrieval_threshold == 0.5
    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["no_support_rejection_rate"] == 1.0
    assert result["retrieval"]["task_balanced_accuracy"] == 1.0


def test_rank_preserving_gate_requires_both_acceptance_thresholds() -> None:
    task = {
        "task_id": "supported",
        "requirement": "Use SQL",
        "support_label": "Direct",
        "selected_candidate_id": "a",
        "candidates": [
            {"candidate_id": "a", "evidence": "Built SQL reports"},
            {"candidate_id": "b", "evidence": "Used spreadsheets"},
        ],
    }
    cached = {
        "supported": {
            "retrieval_scores": [0.9, 0.2],
            "class_probabilities": [
                [0.1, 0.1, 0.8],
                [0.8, 0.1, 0.1],
            ],
        }
    }

    result = evaluate_rank_preserving_gate(
        [task],
        cached,
        RankPreservingGateParameters(
            retrieval_threshold=0.5,
            support_threshold=0.5,
            named_term_rescue=False,
        ),
    )

    assert result["retrieval"]["recall_at_1"] == 1.0
    assert result["retrieval"]["supported_task_success_rate"] == 0.0
    assert result["failures"][0]["failure"] == "support_reject"


def test_rank_preserving_gate_rescues_explicit_named_tool_overlap() -> None:
    task = {
        "task_id": "supported",
        "requirement": "Experience using Looker for product analytics.",
        "support_label": "Partial",
        "selected_candidate_id": "a",
        "candidates": [
            {
                "candidate_id": "a",
                "evidence": "Built operational dashboards in Looker Studio.",
            },
            {"candidate_id": "b", "evidence": "Used spreadsheets."},
        ],
    }
    cached = {
        "supported": {
            "retrieval_scores": [0.9, 0.2],
            "class_probabilities": [
                [0.1, 0.1, 0.8],
                [0.1, 0.1, 0.8],
            ],
        }
    }

    result = evaluate_rank_preserving_gate(
        [task],
        cached,
        RankPreservingGateParameters(
            retrieval_threshold=0.5,
            support_threshold=0.5,
        ),
    )

    assert result["retrieval"]["supported_task_success_rate"] == 1.0


def test_reviewed_rows_do_not_turn_unselected_support_into_negatives() -> None:
    supported = {
        "requirement": "Use SQL",
        "support_label": "Direct",
        "selected_candidate_id": "a",
        "candidates": [
            {"candidate_id": "a", "evidence": "Built SQL reports"},
            {"candidate_id": "b", "evidence": "Used spreadsheets"},
        ],
    }
    unsupported = {
        "requirement": "Use Kubernetes",
        "support_label": "No Support",
        "selected_candidate_id": None,
        "candidates": [
            {"candidate_id": "c", "evidence": "Used spreadsheets"},
            {"candidate_id": "d", "evidence": "Wrote documentation"},
        ],
    }

    rows = reviewed_multiclass_rows([supported, unsupported])

    assert rows == [
        ("Use SQL", "Built SQL reports", "Direct"),
        ("Use Kubernetes", "Used spreadsheets", "No Support"),
        ("Use Kubernetes", "Wrote documentation", "No Support"),
    ]
