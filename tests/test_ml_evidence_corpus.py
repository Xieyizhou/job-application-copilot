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
