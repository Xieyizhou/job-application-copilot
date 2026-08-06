"""Contracts for operational development v3 construction and review."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from ml.annotation import write_queue
from ml.real_holdout import jsonl_bytes, sha256_bytes
from ml import real_development_sources
from ml.operational_development_v3 import (
    CONSTRUCTION_STRATA,
    REVIEWER_FORBIDDEN_FIELDS,
    build_self_adjudication_queue,
    build_single_reviewer_packet,
    build_blind_reviewer_queue,
    repeat_agreement_report,
    sufficiency_report,
    validate_reviewer_packet,
    validate_v3_contract,
)
from ml.real_development import ROLE_FAMILIES
from ml.real_development_isolation import build_development_isolation_index


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str) -> ModuleType:
    path = PROJECT_ROOT / "scripts" / "ml" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tasks() -> list[dict[str, object]]:
    tasks: list[dict[str, object]] = []
    for family_index, family in enumerate(ROLE_FAMILIES):
        for resume_index in range(6):
            resume_hash = f"resume-{family_index}-{resume_index}"
            for stratum_index, stratum in enumerate(CONSTRUCTION_STRATA):
                task_id = (
                    f"task-{family_index}-{resume_index}-{stratum_index}"
                )
                tasks.append(
                    {
                        "schema_version": 1,
                        "task_id": task_id,
                        "presentation_id": task_id,
                        "role_family": family,
                        "requirement": (
                            f"Requirement {task_id} needs capability "
                            f"{stratum_index} in production systems."
                        ),
                        "candidates": [
                            {
                                "candidate_id": f"{task_id}-c{index}",
                                "evidence": (
                                    f"Built distinct system {task_id} "
                                    f"component {index} with measured results."
                                ),
                            }
                            for index in range(4)
                        ],
                        "source_resume_hash": resume_hash,
                        "source_job_hash": f"job-{task_id}",
                        "source_dataset": "djinni-operational-v3",
                        "source_dataset_revision": "revision",
                        "operational_resume_group": resume_hash,
                        "construction_stratum": stratum,
                        "taxonomy_reference": ["O*NET 30.3", "ESCO 1.2.1"],
                        "profile_role_family": family,
                        "pairing_type": "aligned",
                        "hidden_repeat_of": None,
                        "blind_duplicate_of": None,
                    }
                )
    return tasks


def _no_support_state(task: dict[str, object]) -> dict[str, object]:
    candidates = list(task["candidates"])  # type: ignore[arg-type]
    return {
        "action": "label",
        "support_label": "No Support",
        "selected_candidate_id": None,
        "candidate_labels": {
            str(candidate["candidate_id"]): "No Support"
            for candidate in candidates
        },
    }


def test_djinni_profile_expansion_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested_columns: list[list[str]] = []

    def fake_read_parquet(
        _path: Path,
        *,
        columns: list[str],
    ) -> object:
        requested_columns.append(columns)
        return real_development_sources.pd.DataFrame(columns=columns)

    monkeypatch.setattr(
        real_development_sources.pd,
        "read_parquet",
        fake_read_parquet,
    )
    real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=1,
    )
    real_development_sources.load_djinni_profiles(
        Path("unused.parquet"),
        set(),
        random_state=1,
        include_supplemental_profile_text=True,
        prefer_specialized_role=True,
        minimum_evidence=4,
    )

    assert requested_columns == [
        ["id", "Position", "Primary Keyword", "CV"],
        [
            "id",
            "Position",
            "Primary Keyword",
            "CV",
            "Highlights",
            "Moreinfo",
            "Looking For",
        ],
    ]


def test_v3_evaluator_reports_only_aggregate_failure_slices() -> None:
    evaluator = _load_script("evaluate_operational_development_v3.py")
    tasks = [
        {
            "task_id": "supported",
            "role_family": "Data",
            "construction_stratum": "semantic_aligned",
        },
        {
            "task_id": "unsupported",
            "role_family": "Data",
            "construction_stratum": "same_role_adjacent",
        },
        {
            "task_id": "passed",
            "role_family": "ML",
            "construction_stratum": "explicit_aligned",
        },
    ]
    report = evaluator._slice_report(
        tasks,
        {"supported": False, "unsupported": False, "passed": True},
        {
            "supported": "support_reject",
            "unsupported": "false_accept",
        },
    )

    assert report["failure_counts"] == {
        "support_reject": 1,
        "false_accept": 1,
    }
    assert report["by_role_family"]["Data"] == {
        "tasks": 2,
        "accuracy": 0.0,
        "failure_counts": {
            "support_reject": 1,
            "false_accept": 1,
        },
    }
    assert "task_id" not in json.dumps(report)
    assert "requirement" not in json.dumps(report)


def test_v3_contract_has_24_groups_120_tasks_and_balanced_strata() -> None:
    tasks = _tasks()
    report = validate_v3_contract(
        tasks,
        isolation_index=build_development_isolation_index([], []),
    )
    assert report["tasks"] == 120
    assert report["resume_groups"] == 24
    assert report["candidate_count"] == 480
    assert report["role_counts"] == {
        family: 30 for family in ROLE_FAMILIES
    }
    assert report["construction_stratum_counts"] == {
        stratum: 24 for stratum in CONSTRUCTION_STRATA
    }


def test_reviewer_packet_has_12_blind_repeats_without_hidden_metadata() -> None:
    tasks = _tasks()
    packet, repeat_map = build_single_reviewer_packet(
        tasks,
        repeat_count=12,
        random_state=7,
    )
    validate_reviewer_packet(packet)
    assert len(packet) == 132
    assert len(repeat_map) == 12
    assert all(not (set(task) & REVIEWER_FORBIDDEN_FIELDS) for task in packet)
    by_id = {str(task["task_id"]): task for task in packet}
    for repeat in repeat_map:
        original = by_id[repeat["original_presentation_id"]]
        repeated = by_id[repeat["presentation_id"]]
        assert [
            candidate["candidate_id"]
            for candidate in original["candidates"]
        ] != [
            candidate["candidate_id"]
            for candidate in repeated["candidates"]
        ]


def test_independent_blind_reviewer_queue_changes_every_candidate_order() -> None:
    tasks = _tasks()[:24]
    packet = build_blind_reviewer_queue(
        tasks,
        reviewer_id="reviewer_b",
        random_state=19,
    )
    validate_reviewer_packet(packet)
    source_by_id = {str(task["task_id"]): task for task in tasks}

    assert {str(task["task_id"]) for task in packet} == set(source_by_id)
    assert all(not (set(task) & REVIEWER_FORBIDDEN_FIELDS) for task in packet)
    assert all(
        [
            candidate["candidate_id"] for candidate in task["candidates"]
        ]
        != [
            candidate["candidate_id"]
            for candidate in source_by_id[str(task["task_id"])]["candidates"]
        ]
        for task in packet
    )


def test_repeat_agreement_freezes_kappa_and_builds_content_only_queue() -> None:
    tasks = _tasks()
    packet, repeat_map = build_single_reviewer_packet(
        tasks,
        repeat_count=12,
        random_state=11,
    )
    states = {
        str(presentation["task_id"]): _no_support_state(presentation)
        for presentation in packet
    }
    report = repeat_agreement_report(tasks, repeat_map, states)
    assert report["candidate_cohen_kappa"] == 1.0
    assert report["task_exact_agreement_rate"] == 1.0
    assert report["agreement_gate_passed"] is True

    changed = repeat_map[0]
    repeated_id = changed["presentation_id"]
    candidate_id = str(packet[0]["candidates"][0]["candidate_id"])
    canonical = next(
        task
        for task in tasks
        if task["task_id"] == changed["canonical_task_id"]
    )
    candidate_id = str(canonical["candidates"][0]["candidate_id"])
    states[repeated_id] = {
        "action": "label",
        "support_label": "Direct",
        "selected_candidate_id": candidate_id,
        "candidate_labels": {
            str(candidate["candidate_id"]): (
                "Direct"
                if str(candidate["candidate_id"]) == candidate_id
                else "No Support"
            )
            for candidate in canonical["candidates"]
        },
    }
    conflict = repeat_agreement_report(tasks, repeat_map, states)
    assert conflict["disagreement_tasks"] == 1
    queue = build_self_adjudication_queue(
        tasks,
        conflict["disagreement_task_ids"],
        random_state=13,
    )
    assert len(queue) == 1
    assert not (set(queue[0]) & REVIEWER_FORBIDDEN_FIELDS)


def test_sufficiency_gate_uses_natural_labels_without_selection() -> None:
    tasks = _tasks()
    for family in ROLE_FAMILIES:
        family_tasks = [task for task in tasks if task["role_family"] == family]
        for index, task in enumerate(family_tasks):
            task["support_label"] = (
                "Direct"
                if index < 5
                else "Partial"
                if index < 15
                else "No Support"
            )
    report = sufficiency_report(tasks)
    assert report["dataset_sufficiency_gate_passed"] is True
    assert report["post_label_selection_performed"] is False
    changed = 0
    for task in tasks:
        if task["support_label"] == "Direct" and changed < 6:
            task["support_label"] = "Partial"
            changed += 1
    blocked = sufficiency_report(tasks)
    assert blocked["checks"]["direct_tasks_at_least_15"] is False


def test_v3_artifact_is_explicitly_offline_and_not_imported_by_dashboard() -> None:
    evaluator = (
        PROJECT_ROOT
        / "scripts"
        / "ml"
        / "evaluate_operational_development_v3.py"
    ).read_text(encoding="utf-8")
    assert '"product_integration_allowed": False' in evaluator
    assert '"demo_integration_allowed": False' in evaluator
    assert '"application_loadable": False' in evaluator
    dashboard_sources = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (PROJECT_ROOT / "src").glob("dashboard*.py")
    )
    assert "operational_development_v3" not in dashboard_sources


def test_finalizer_keeps_120_unique_tasks_and_passes_quality_gates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tasks = _tasks()
    packet, repeat_map = build_single_reviewer_packet(
        tasks,
        repeat_count=12,
        random_state=17,
    )
    canonical_by_presentation = {
        str(task["task_id"]): str(task["task_id"]) for task in tasks
    }
    canonical_by_presentation.update(
        {
            row["presentation_id"]: row["canonical_task_id"]
            for row in repeat_map
        }
    )
    task_by_id = {str(task["task_id"]): task for task in tasks}
    family_positions = {
        family: {
            str(task["task_id"]): index
            for index, task in enumerate(
                row for row in tasks if row["role_family"] == family
            )
        }
        for family in ROLE_FAMILIES
    }
    events: list[dict[str, object]] = []
    for presentation in packet:
        presentation_id = str(presentation["task_id"])
        canonical_id = canonical_by_presentation[presentation_id]
        canonical = task_by_id[canonical_id]
        index = family_positions[str(canonical["role_family"])][canonical_id]
        label = (
            "Direct"
            if index < 5
            else "Partial"
            if index < 15
            else "No Support"
        )
        selected = (
            str(canonical["candidates"][0]["candidate_id"])
            if label != "No Support"
            else None
        )
        events.append(
            {
                "task_id": presentation_id,
                "action": "label",
                "support_label": label,
                "selected_candidate_id": selected,
                "candidate_labels": {
                    str(candidate["candidate_id"]): (
                        label
                        if str(candidate["candidate_id"]) == selected
                        else "No Support"
                    )
                    for candidate in canonical["candidates"]
                },
            }
        )
    states = {
        str(event["task_id"]): event for event in events
    }
    repeat_report = repeat_agreement_report(tasks, repeat_map, states)
    annotation_dir = tmp_path / "annotations"
    output_dir = tmp_path / "processed"
    annotation_dir.mkdir()
    write_queue(tasks, annotation_dir / "queue.jsonl")
    (annotation_dir / "manifest.json").write_text(
        json.dumps({"task_sha256": sha256_bytes(jsonl_bytes(tasks))}),
        encoding="utf-8",
    )
    (annotation_dir / "reviewer_decisions.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    (annotation_dir / "repeat_agreement.json").write_text(
        json.dumps(repeat_report),
        encoding="utf-8",
    )
    module = _load_script("finalize_operational_development_v3.py")

    class Args:
        random_state = 17

        def __init__(self) -> None:
            self.annotation_dir = annotation_dir
            self.output_dir = output_dir

    monkeypatch.setattr(module, "parse_args", lambda: Args())  # type: ignore[attr-defined]
    module.main()
    gold = (output_dir / "annotated_tasks.jsonl").read_text().splitlines()
    pairs = (output_dir / "training_pairs.jsonl").read_text().splitlines()
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert len(gold) == 120
    assert len(pairs) == 480
    assert manifest["agreement_gate_passed"] is True
    assert manifest["dataset_sufficiency_gate_passed"] is True
    assert manifest["eligible_for_model_comparison"] is True
    assert manifest["product_integration_allowed"] is False


def test_v3_inner_candidate_crossfit_excludes_test_resume_groups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_script("evaluate_operational_development_v3.py")
    tasks = [
        {
            "task_id": f"task-{group}-{index}",
            "source_resume_hash": f"resume-{group}",
            "requirement": f"requirement-{group}-{index}",
            "support_label": "Direct" if group % 2 else "No Support",
            "selected_candidate_id": "a" if group % 2 else None,
            "candidates": [
                {
                    "candidate_id": candidate,
                    "evidence": f"evidence-{group}-{index}-{candidate}",
                    "support_label": (
                        "Direct"
                        if group % 2 and candidate == "a"
                        else "No Support"
                    ),
                }
                for candidate in ("a", "b")
            ],
        }
        for group in range(6)
        for index in range(2)
    ]

    class FakeModel:
        def __init__(self, requirements: set[str]) -> None:
            self.requirements = requirements

    def fake_fit(
        rows: list[tuple[str, str, str]],
        *,
        random_state: int,
    ) -> FakeModel:
        del random_state
        return FakeModel({row[0] for row in rows})

    def fake_predict(
        model: FakeModel,
        test_tasks: list[dict[str, object]],
    ) -> dict[str, list[list[float]]]:
        training_groups = {
            requirement.split("-")[1]
            for requirement in model.requirements
            if requirement.startswith("requirement-")
        }
        assert all(
            str(task["source_resume_hash"]).split("-")[1]
            not in training_groups
            for task in test_tasks
        )
        return {
            str(task["task_id"]): [
                [0.7, 0.2, 0.1],
                [0.1, 0.2, 0.7],
            ]
            for task in test_tasks
        }

    monkeypatch.setattr(module, "fit_candidate", fake_fit)
    monkeypatch.setattr(module, "predict_tasks", fake_predict)
    probabilities = module._inner_oof_candidate_probabilities(
        tasks,
        [("base requirement", "base evidence", "No Support")],
        random_state=19,
        n_splits=3,
    )
    assert set(probabilities) == {
        str(task["task_id"]) for task in tasks
    }
