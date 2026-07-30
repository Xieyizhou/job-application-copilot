from __future__ import annotations

from ml.candidate_completion import (
    merge_completed_candidate_labels,
    tasks_requiring_candidate_completion,
)


def _task(task_id: str, role: str, label: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "role_family": role,
        "requirement": f"Requirement {task_id}",
        "support_label": label,
        "selected_candidate_id": f"{task_id}-a",
        "candidates": [
            {"candidate_id": f"{task_id}-a", "evidence": f"Evidence {task_id} A"},
            {"candidate_id": f"{task_id}-b", "evidence": f"Evidence {task_id} B"},
        ],
    }


def test_completion_excludes_fully_labeled_tasks_and_balances_groups() -> None:
    tasks = [
        _task("data-direct", "Data", "Direct"),
        _task("data-partial", "Data", "Partial"),
        _task("ml-direct", "ML", "Direct"),
    ]
    pairs = [
        {
            "task_id": "data-direct",
            "evidence": "Evidence data-direct A",
            "support_label": "Direct",
        },
        {
            "task_id": "data-direct",
            "evidence": "Evidence data-direct B",
            "support_label": "No Support",
        },
    ]

    selected = tasks_requiring_candidate_completion(
        tasks,
        pairs,
        random_state=7,
    )

    assert {row["queue_task"]["task_id"] for row in selected} == {
        "data-partial",
        "ml-direct",
    }
    assert all(row["queue_task"]["schema_version"] == 1 for row in selected)
    assert all("support_label" not in row["queue_task"] for row in selected)


def test_completed_candidate_labels_replace_legacy_pairs_without_changing_tasks() -> None:
    base_task = _task("data-direct", "Data", "Direct")
    base_task["evaluation_group"] = "semantic:data"
    base_pair = {
        "pair_id": "old-pair",
        "task_id": "data-direct",
        "requirement": base_task["requirement"],
        "evidence": "Evidence data-direct A",
        "binary_label": 1,
        "support_label": "Direct",
    }
    completed_task = {
        **base_task,
        "candidate_label_coverage": "complete",
        "candidates": [
            {
                **base_task["candidates"][0],  # type: ignore[index]
                "support_label": "Direct",
            },
            {
                **base_task["candidates"][1],  # type: ignore[index]
                "support_label": "No Support",
            },
        ],
    }
    completed_pairs = [
        {
            "pair_id": "new-positive",
            "task_id": "data-direct",
            "requirement": base_task["requirement"],
            "evidence": "Evidence data-direct A",
            "binary_label": 1,
            "support_label": "Direct",
        },
        {
            "pair_id": "new-negative",
            "task_id": "data-direct",
            "requirement": base_task["requirement"],
            "evidence": "Evidence data-direct B",
            "binary_label": 0,
            "support_label": "No Support",
        },
    ]

    tasks, pairs, manifest = merge_completed_candidate_labels(
        [base_task],
        [base_pair],
        [completed_task],
        completed_pairs,
    )

    assert len(tasks) == 1
    assert {pair["pair_id"] for pair in pairs} == {
        "new-positive",
        "new-negative",
    }
    assert tasks[0]["review_source"] == "human_candidate_completion"
    assert manifest["candidate_label_coverage_counts"] == {"complete": 1}
