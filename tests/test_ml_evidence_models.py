from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_models import (
    HybridEvidenceReranker,
    LexicalGuardedReranker,
    LsaEmbeddingScorer,
    PairwiseHybridReranker,
    WordTfidfCosineScorer,
)


REQUIREMENTS = [
    "Build recurring SQL reporting workflows.",
    "Deploy predictive models as reliable services.",
    "Build recurring SQL reporting workflows.",
    "Deploy predictive models as reliable services.",
]
EVIDENCE = [
    "Automated monthly reports with SQL and warehouse tables.",
    "Shipped a classification endpoint with health monitoring.",
    "Prepared weekly meeting agendas for project stakeholders.",
    "Compared algorithms in an offline notebook.",
]
LABELS = [1, 1, 0, 0]


def test_word_and_lsa_scorers_return_aligned_finite_scores() -> None:
    for scorer in (
        WordTfidfCosineScorer(),
        LsaEmbeddingScorer(max_components=4, random_state=7),
    ):
        scorer.fit(REQUIREMENTS, EVIDENCE)
        scores = scorer.score(REQUIREMENTS, EVIDENCE)
        assert scores.shape == (4,)
        assert np.isfinite(scores).all()
        assert scorer.feature_manifest()["model_type"]


def test_hybrid_reranker_fits_reviewed_binary_pairs() -> None:
    model = HybridEvidenceReranker(max_components=4, random_state=7)
    model.fit(REQUIREMENTS, EVIDENCE, LABELS)

    scores = model.predict_proba(REQUIREMENTS, EVIDENCE)

    assert scores.shape == (4,)
    assert np.all((scores >= 0.0) & (scores <= 1.0))
    assert model.feature_manifest()["model_type"] == "hybrid_lsa_reranker"


def test_pairwise_reranker_uses_strongest_evidence_preferences() -> None:
    tasks = [
        {
            "task_id": "task-1",
            "requirement": REQUIREMENTS[0],
            "support_label": "Direct",
            "selected_candidate_id": "candidate-a",
            "candidates": [
                {"candidate_id": "candidate-a", "evidence": EVIDENCE[0]},
                {"candidate_id": "candidate-b", "evidence": EVIDENCE[2]},
            ],
        },
        {
            "task_id": "task-2",
            "requirement": REQUIREMENTS[1],
            "support_label": "Direct",
            "selected_candidate_id": "candidate-a",
            "candidates": [
                {"candidate_id": "candidate-a", "evidence": EVIDENCE[1]},
                {"candidate_id": "candidate-b", "evidence": EVIDENCE[3]},
            ],
        },
    ]
    model = PairwiseHybridReranker(max_components=4, random_state=7)
    model.fit(REQUIREMENTS, EVIDENCE, LABELS, tasks)

    scores = model.predict_proba(REQUIREMENTS, EVIDENCE)

    assert scores.shape == (4,)
    assert model.feature_manifest()["pairwise_comparisons"] == 2


def test_lexical_guarded_reranker_blends_ranking_and_support_scores() -> None:
    model = LexicalGuardedReranker(max_components=4, random_state=7)
    model.fit(REQUIREMENTS, EVIDENCE, LABELS)

    scores = model.predict_proba(REQUIREMENTS, EVIDENCE)

    assert scores.shape == (4,)
    assert model.feature_manifest()["lexical_weight"] == 0.7


def test_models_reject_misaligned_inputs_and_one_class_training() -> None:
    with pytest.raises(ValueError, match="equal lengths"):
        WordTfidfCosineScorer().fit(["requirement"], [])
    with pytest.raises(ValueError, match="both 0 and 1"):
        HybridEvidenceReranker().fit(REQUIREMENTS, EVIDENCE, [1, 1, 1, 1])
