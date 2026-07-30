"""Tests for revealed shadow disagreement diagnostics."""

from __future__ import annotations

from ml.evidence_shadow_review import evaluate_shadow_review


def test_review_counts_shadow_false_accept_and_baseline_false_reject() -> None:
    task = {
        "task_id": "task-1",
        "role_family": "Data",
        "candidates": [
            {"candidate_id": "shadow", "evidence": "Adjacent evidence."},
            {"candidate_id": "filler", "evidence": "Direct evidence."},
        ],
    }
    state = {
        "support_label": "Direct",
        "selected_candidate_id": "filler",
        "candidate_labels": {
            "shadow": "No Support",
            "filler": "Direct",
        },
    }
    construction = {
        "task_id": "task-1",
        "comparison": "shadow_only_accept",
        "candidate_origins": {
            "shadow": "shadow",
            "filler": "filler",
        },
    }

    report = evaluate_shadow_review(
        [task],
        {"task-1": state},
        [construction],
    )

    assert report["methods"]["shadow"]["error_counts"] == {
        "false_accept": 1
    }
    assert report["methods"]["baseline"]["error_counts"] == {
        "observed_false_reject": 1
    }
    assert report["operational_safety_stop"]["triggered"] is True
    assert (
        report["operational_safety_stop"]["reviewed_sample_training_use"]
        == "prohibited"
    )
