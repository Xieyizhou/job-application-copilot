from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.evidence_corpus import (
    EvidenceCorpusError,
    combine_reviewed_sources,
    gold_tasks_to_dataset,
    reviewed_real_tasks_to_dataset,
)


def _gold() -> dict[str, object]:
    return {
        "gold_id": "gold-1",
        "decision_source": "blind_consensus",
        "reviewer_count": 3,
        "semantic_case_group_id": "deployment-1",
        "support_label": "Direct",
        "best_candidate_id": "candidate-a",
        "role_family": "ML",
        "requirement": "Deploy predictive models into a production service.",
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "evidence": "Shipped a churn scorer behind a service endpoint.",
                "support_label": "Direct",
            },
            {
                "candidate_id": "candidate-b",
                "evidence": "Compared several algorithms in an offline notebook.",
                "support_label": "Partial",
            },
            {
                "candidate_id": "candidate-c",
                "evidence": "Prepared weekly project status notes.",
                "support_label": "No Support",
            },
        ],
    }


def test_gold_conversion_uses_candidate_level_reviewed_labels() -> None:
    tasks, pairs = gold_tasks_to_dataset([_gold()])

    assert tasks[0]["evaluation_group"] == "semantic:deployment-1"
    assert [pair["binary_label"] for pair in pairs] == [1, 1, 0]
    assert {pair["label_scope"] for pair in pairs} == {
        "reviewed_candidate_judgment"
    }
    assert all("producer" not in repr(pair).lower() for pair in pairs)


def test_blind_consensus_requires_three_reviewers() -> None:
    gold = _gold()
    gold["reviewer_count"] = 2

    with pytest.raises(EvidenceCorpusError, match="three reviewers"):
        gold_tasks_to_dataset([gold])


def test_human_adjudication_preserves_conservative_unlabeled_candidates() -> None:
    gold = _gold()
    gold["decision_source"] = "human_adjudication"
    gold["reviewer_count"] = 1
    gold["candidates"][1]["support_label"] = "Unlabeled"  # type: ignore[index]
    gold["candidates"][2]["support_label"] = "Unlabeled"  # type: ignore[index]

    tasks, pairs = gold_tasks_to_dataset([gold])

    assert tasks[0]["review_source"] == "human_adjudication"
    assert [pair["binary_label"] for pair in pairs] == [1]
    assert {pair["support_label"] for pair in pairs} == {"Direct"}


def test_unlabeled_candidates_are_not_allowed_for_blind_consensus() -> None:
    gold = _gold()
    gold["candidates"][1]["support_label"] = "Unlabeled"  # type: ignore[index]

    with pytest.raises(EvidenceCorpusError, match="Unsupported candidate"):
        gold_tasks_to_dataset([gold])


def test_combined_corpus_preserves_sources_and_groups() -> None:
    human_task = {
        "task_id": "human-1",
        "requirement": "Use SQL for recurring reporting.",
        "candidates": [
            {"candidate_id": "human-a", "evidence": "Built monthly SQL reports."},
            {"candidate_id": "human-b", "evidence": "Prepared meeting notes."},
        ],
        "selected_candidate_id": "human-a",
        "support_label": "Direct",
        "role_family": "Data",
        "template_group": "use sql for",
    }
    human_pair = {
        "pair_id": "pair-human",
        "task_id": "human-1",
        "requirement": human_task["requirement"],
        "evidence": "Built monthly SQL reports.",
        "binary_label": 1,
        "support_label": "Direct",
        "template_group": "use sql for",
    }

    tasks, pairs, manifest = combine_reviewed_sources(
        [human_task],
        [human_pair],
        [_gold()],
    )

    assert len(tasks) == 2
    assert len(pairs) == 4
    assert manifest["task_source_counts"] == {
        "human_annotation": 1,
        "blind_consensus": 1,
    }
    assert manifest["gold_policy"].startswith("Human annotations")


def test_real_training_supplement_keeps_only_reviewed_pair_labels() -> None:
    real = {
        "record_type": "real_training_supplement_gold_task",
        "task_id": "real-1",
        "decision_source": "human_adjudication",
        "support_label": "Partial",
        "selected_candidate_id": "a",
        "role_family": "Software",
        "requirement": "Use Java in production.",
        "source_job_hash": "job-1",
        "source_resume_hash": "resume-1",
        "source_dataset": "public_real_text",
        "candidates": [
            {"candidate_id": "a", "evidence": "Built a Java service."},
            {"candidate_id": "b", "evidence": "Prepared meeting notes."},
        ],
    }

    tasks, pairs = reviewed_real_tasks_to_dataset([real])

    assert tasks[0]["review_source"] == "acceptance_training_supplement"
    assert len(pairs) == 1
    assert pairs[0]["support_label"] == "Partial"
    assert pairs[0]["source_resume_hash"] == "resume-1"


def test_real_no_support_adds_all_candidates_as_negatives() -> None:
    real = {
        "record_type": "real_training_supplement_gold_task",
        "task_id": "real-2",
        "decision_source": "human_model_agreement",
        "support_label": "No Support",
        "selected_candidate_id": None,
        "role_family": "ML",
        "requirement": "Use PyTorch.",
        "source_job_hash": "job-2",
        "source_resume_hash": "resume-2",
        "source_dataset": "public_real_text",
        "candidates": [
            {"candidate_id": "a", "evidence": "Prepared reports."},
            {"candidate_id": "b", "evidence": "Managed a backlog."},
        ],
    }

    _, pairs = reviewed_real_tasks_to_dataset([real])

    assert len(pairs) == 2
    assert {pair["binary_label"] for pair in pairs} == {0}
