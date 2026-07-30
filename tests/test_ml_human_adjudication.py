from __future__ import annotations

from pathlib import Path
import sys

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from ml.human_adjudication import (
    HumanAdjudicationError,
    UNLABELED_CANDIDATE_LABEL,
    build_human_adjudication_gold,
)


def _task(task_id: str = "case-1") -> dict[str, object]:
    return {
        "schema_version": 1,
        "task_id": task_id,
        "role_family": "ML",
        "requirement": "Deploy a model through a production service.",
        "candidates": [
            {
                "candidate_id": "candidate-a",
                "evidence": "Released a fraud scorer behind a monitored API endpoint.",
            },
            {
                "candidate_id": "candidate-b",
                "evidence": "Compared model metrics in an offline notebook.",
            },
        ],
    }


def _source_case(task: dict[str, object]) -> dict[str, object]:
    return {
        "case_id": task["task_id"],
        "semantic_case_group_id": "deployment-001",
        "source_kind": "synthetic",
        "requirement": task["requirement"],
        "candidates": task["candidates"],
    }


def _event(
    task_id: str,
    *,
    label: str,
    selected_id: str | None,
) -> dict[str, object]:
    return {
        "event_id": "event-1",
        "task_id": task_id,
        "action": "label",
        "support_label": label,
        "selected_candidate_id": selected_id,
        "cover_letter_safe": True,
    }


def test_human_adjudication_keeps_unselected_supported_candidates_unlabeled() -> None:
    task = _task()
    gold, report = build_human_adjudication_gold(
        [task],
        [_event("case-1", label="Direct", selected_id="candidate-a")],
        [_source_case(task)],
    )

    assert report["accepted_gold_tasks"] == 1
    assert gold[0]["decision_source"] == "human_adjudication"
    assert gold[0]["semantic_case_group_id"] == "deployment-001"
    labels = {candidate["candidate_id"]: candidate["support_label"] for candidate in gold[0]["candidates"]}
    assert labels == {
        "candidate-a": "Direct",
        "candidate-b": UNLABELED_CANDIDATE_LABEL,
    }


def test_no_support_adjudication_labels_every_candidate_negative() -> None:
    task = _task()
    gold, _ = build_human_adjudication_gold(
        [task],
        [_event("case-1", label="No Support", selected_id=None)],
        [_source_case(task)],
    )

    assert gold[0]["best_candidate_id"] is None
    assert {candidate["support_label"] for candidate in gold[0]["candidates"]} == {
        "No Support"
    }


def test_complete_candidate_judgments_survive_human_gold_export() -> None:
    task = _task()
    event = _event("case-1", label="Direct", selected_id="candidate-a")
    event["candidate_labels"] = {
        "candidate-a": "Direct",
        "candidate-b": "Partial",
    }

    gold, report = build_human_adjudication_gold(
        [task],
        [event],
        [_source_case(task)],
    )

    assert report["accepted_gold_tasks"] == 1
    assert {
        candidate["candidate_id"]: candidate["support_label"]
        for candidate in gold[0]["candidates"]
    } == {
        "candidate-a": "Direct",
        "candidate-b": "Partial",
    }


def test_complete_export_rejects_unresolved_or_missing_tasks() -> None:
    task = _task()
    with pytest.raises(HumanAdjudicationError, match="Every adjudication task"):
        build_human_adjudication_gold([task], [], [_source_case(task)])

    gold, report = build_human_adjudication_gold(
        [task],
        [_event("case-1", label="Uncertain", selected_id=None)],
        [_source_case(task)],
        require_complete=False,
    )
    assert gold == []
    assert report["uncertain_task_ids"] == ["case-1"]


def test_adjudication_rejects_source_content_mismatch() -> None:
    task = _task()
    source = _source_case(task)
    source["requirement"] = "Build a financial forecast."
    with pytest.raises(HumanAdjudicationError, match="does not match"):
        build_human_adjudication_gold(
            [task],
            [_event("case-1", label="Direct", selected_id="candidate-a")],
            [source],
        )
